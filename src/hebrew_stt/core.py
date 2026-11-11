"""One media file in, transcript (and optional summary) out. Shared by the CLI and the window."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

from .summarize import DEFAULT_STYLE, SummarySettings, style, summarize
from .transcribe import MEDIA_EXTENSIONS, OutputPaths, ProgressCallback, Transcriber, output_paths

log = logging.getLogger(__name__)
T = TypeVar("T")


class Cancelled(Exception):
    """The user stopped the file."""


@dataclass
class Result:
    media: Path
    paths: OutputPaths
    duration: float
    summarized: bool = False
    summary_error: Optional[str] = None
    summary_cancelled: bool = False


def _write(path: Path, text: str) -> None:
    """UTF-8 with LF line endings on every OS, so results are identical everywhere."""
    path.write_text(text, encoding="utf-8", newline="\n")


def is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def collect_media(paths: Iterable[str | Path]) -> list[Path]:
    """Keep supported media files; a folder adds the media files directly inside it."""
    found: list[Path] = []
    seen: set[Path] = set()
    for path in map(Path, paths):
        candidates = sorted(path.iterdir()) if path.is_dir() else [path]
        for candidate in candidates:
            key = candidate.resolve()
            if candidate.is_file() and is_media(candidate) and key not in seen:
                seen.add(key)
                found.append(candidate)
    return found


def _until_stopped(task: Callable[[], T], stop: threading.Event) -> T:
    """Run a blocking call (a network request) but stop waiting for it once `stop` is set.

    The abandoned call finishes in its own thread and its result is dropped.
    """
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["value"] = task()
        except BaseException as exc:  # handed back to the caller below
            outcome["error"] = exc

    thread = threading.Thread(target=target, name="hebrew-stt-summary", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(0.2)
        if stop.is_set():
            raise Cancelled()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def process_file(
    media: str | Path,
    transcriber: Transcriber,
    settings: Optional[SummarySettings] = None,
    output_dir: Optional[str | Path] = None,
    progress: Optional[ProgressCallback] = None,
    on_status: Optional[Callable[[str], None]] = None,
    stop: Optional[threading.Event] = None,
    style_name: str = DEFAULT_STYLE,
    language: str = "he",
    terms: str = "",
) -> Result:
    """Transcribe one file and write the results. A failed summary does not fail the file.

    Setting `stop` during transcription raises Cancelled and writes nothing. Setting it
    during the summary keeps the transcript and drops the summary.
    """
    summary_style = style(style_name)  # an unknown style fails before the long transcription
    media = Path(media)
    paths = output_paths(media, Path(output_dir) if output_dir else None)

    def report(fraction: float, position: float, total: float) -> None:
        if stop is not None and stop.is_set():
            raise Cancelled()
        if progress:
            progress(fraction, position, total)

    if stop is not None and stop.is_set():
        raise Cancelled()
    paths.text.parent.mkdir(parents=True, exist_ok=True)
    transcript = transcriber.transcribe(media, report, terms=terms)
    if stop is not None and stop.is_set():
        raise Cancelled()
    _write(paths.text, transcript.text + "\n")
    _write(paths.timestamps, transcript.timestamped_text() + "\n")
    result = Result(media, paths, transcript.duration)

    if settings is None or not settings.enabled:
        return result
    if not transcript.text.strip():
        result.summary_error = "No speech was found, so there is nothing to summarize."
        return result
    if on_status:
        on_status(f"Summarizing with {settings.model}...")
    def run_summary() -> str:
        source = transcript.timestamped_text() if summary_style.timestamps else transcript.text
        return summarize(
            source, settings, title=media.stem, style_name=style_name, language=language, terms=terms,
        )

    try:
        summary = run_summary() if stop is None else _until_stopped(run_summary, stop)
        _write(paths.summary, summary)
        result.summarized = True
    except Cancelled:
        result.summary_cancelled = True
    except Exception as exc:  # network, auth, quota... keep the transcript either way
        log.debug("Summary failed", exc_info=True)
        result.summary_error = f"{type(exc).__name__}: {exc}"
    return result
