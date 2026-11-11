"""Command line: `hebrew-stt FILE...`. Without files it opens the page in the browser."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .core import collect_media, process_file
from .summarize import DEFAULT_STYLE, STYLES, load_settings
from .transcribe import DEFAULT_MODEL, MEDIA_EXTENSIONS, Transcriber, format_timestamp

log = logging.getLogger("hebrew_stt")


class _ProgressBar:
    def __init__(self, width: int = 30):
        self.width = width
        self.done = False

    def __call__(self, fraction: float, current: float, total: float) -> None:
        if self.done:
            return
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        print(
            f"\r  [{bar}] {fraction:4.0%}  {format_timestamp(current)} / {format_timestamp(total)}",
            end="", file=sys.stderr, flush=True,
        )
        if fraction >= 1.0:
            self.finish()

    def finish(self) -> None:
        if not self.done:
            print(file=sys.stderr)
            self.done = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hebrew-stt",
        description=(
            "Transcribe Hebrew audio/video locally, with an optional AI summary. "
            "Run without files to use it in the browser."
        ),
    )
    parser.add_argument("files", nargs="*", help="audio or video files, or folders that contain them")
    parser.add_argument("--no-summary", action="store_true", help="only transcribe, even if an API key is set")
    parser.add_argument(
        "--style", choices=list(STYLES), default=DEFAULT_STYLE, metavar="STYLE",
        help="summary style: " + ", ".join(f"{key} ({s.label})" for key, s in STYLES.items())
        + f". Default: {DEFAULT_STYLE}",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="where to save results (default: next to each file; in the browser: Documents/hebrew-stt)",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="default: auto")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model in CTranslate2 format (default: {DEFAULT_MODEL})")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "openai", "faster_whisper", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_console()

    if not args.files:
        from .web import serve

        return serve(model=args.model, device=args.device, output_dir=args.output_dir)

    missing = [name for name in args.files if not Path(name).exists()]
    for name in missing:
        log.error("Not found: %s", name)
    media = collect_media(f for f in args.files if f not in missing)
    if not media:
        log.error("No supported audio or video files. Supported: %s", " ".join(sorted(MEDIA_EXTENSIONS)))
        return 2

    settings = None if args.no_summary else load_settings()
    if settings is not None and not settings.enabled:
        log.info("No summary API key is set, so only transcribing (see README, 'AI summary').")
        settings = None

    transcriber = Transcriber(model=args.model, device=args.device)
    errors = len(missing)
    for i, path in enumerate(media, start=1):
        log.info("[%d/%d] %s", i, len(media), path.name)
        bar = _ProgressBar()
        try:
            result = process_file(
                path, transcriber, settings, args.output_dir, progress=bar, on_status=log.info,
                style_name=args.style,
            )
        except Exception as exc:
            bar.finish()
            log.error("  Failed: %s", exc)
            errors += 1
            continue
        bar.finish()
        log.info("  Duration %s. Saved:", format_timestamp(result.duration))
        log.info("    %s", result.paths.text)
        log.info("    %s", result.paths.timestamps)
        if result.summarized:
            log.info("    %s", result.paths.summary)
        elif result.summary_error:
            log.warning("  Summary failed: %s", result.summary_error)
            errors += 1

    if len(media) > 1:
        log.info("Done: %d file(s), %d problem(s).", len(media), errors)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
