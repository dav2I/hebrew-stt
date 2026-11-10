"""One media file in, transcript (and optional summary) out. Shared by the CLI and the window."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .summarize import SummarySettings, summarize
from .transcribe import MEDIA_EXTENSIONS, OutputPaths, ProgressCallback, Transcriber, output_paths

log = logging.getLogger(__name__)


@dataclass
class Result:
    media: Path
    paths: OutputPaths
    duration: float
    summarized: bool = False
    summary_error: Optional[str] = None


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


def process_file(
    media: str | Path,
    transcriber: Transcriber,
    settings: Optional[SummarySettings] = None,
    output_dir: Optional[str | Path] = None,
    progress: Optional[ProgressCallback] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> Result:
    """Transcribe one file and write the results. A failed summary does not fail the file."""
    media = Path(media)
    paths = output_paths(media, Path(output_dir) if output_dir else None)
    paths.text.parent.mkdir(parents=True, exist_ok=True)

    transcript = transcriber.transcribe(media, progress)
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
    try:
        _write(paths.summary, summarize(transcript.text, settings, title=media.stem))
        result.summarized = True
    except Exception as exc:  # network, auth, quota... keep the transcript either way
        log.debug("Summary failed", exc_info=True)
        result.summary_error = f"{type(exc).__name__}: {exc}"
    return result
