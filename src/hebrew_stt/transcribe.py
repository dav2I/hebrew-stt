"""Local transcription with faster-whisper and the ivrit-ai Hebrew model."""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
LANGUAGE = "he"

MEDIA_EXTENSIONS = frozenset({
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
})

# progress(fraction 0..1, current_seconds, total_seconds)
ProgressCallback = Callable[[float, float, float], None]

log = logging.getLogger(__name__)

if sys.platform == "win32":
    # Hugging Face cache symlinks need admin rights or developer mode on Windows.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# Libraries CTranslate2 loads at runtime for GPU inference (CUDA 12, cuDNN 9).
_CUDA_LIBRARIES = {
    "win32": ("cublas64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"),
    "linux": ("libcublas.so.12", "libcudnn_ops.so.9", "libcudnn_cnn.so.9"),
}


# ---------------------------------------------------------------- formatting

def format_timestamp(seconds: float, with_hours: bool = False) -> str:
    """Format seconds as MM:SS, or HH:MM:SS when asked or when over an hour."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if with_hours or hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


_INVALID_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
}


def safe_stem(name: str, max_length: int = 150) -> str:
    """Turn a file stem into a name that is valid on Windows, macOS and Linux."""
    stem = _INVALID_CHARS.sub("", name)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    stem = stem[:max_length].rstrip(" .")
    if not stem:
        return "transcript"
    if stem.split(".")[0].upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    return stem


@dataclass(frozen=True)
class OutputPaths:
    text: Path
    timestamps: Path
    summary: Path


def output_paths(media: Path, output_dir: Optional[Path] = None) -> OutputPaths:
    """Where the results for `media` go: next to it, unless output_dir is given."""
    folder = Path(output_dir) if output_dir else Path(media).parent
    stem = safe_stem(Path(media).stem)
    return OutputPaths(
        text=folder / f"{stem}.txt",
        timestamps=folder / f"{stem}_timestamps.txt",
        summary=folder / f"{stem}_summary.md",
    )


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment]
    duration: float

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments)

    def timestamped_text(self) -> str:
        with_hours = self.duration >= 3600
        return "\n".join(
            f"[{format_timestamp(s.start, with_hours)} -> "
            f"{format_timestamp(s.end, with_hours)}] {s.text}"
            for s in self.segments
        )


# ------------------------------------------------------------------ hardware

def _add_windows_cuda_dll_dirs() -> None:
    """Make DLLs from the optional nvidia-* pip packages loadable on Windows."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for root in spec.submodule_search_locations:
        for bin_dir in Path(root).glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                continue
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def _cuda_libraries_present() -> bool:
    names = _CUDA_LIBRARIES.get(sys.platform)
    if not names:
        return False
    for name in names:
        try:
            ctypes.CDLL(name)
        except OSError:
            log.debug("CUDA library not found: %s", name)
            return False
    return True


def detect_device(requested: str = "auto") -> tuple[str, str]:
    """Pick (device, compute_type): NVIDIA GPU in float16 if usable, else CPU in int8."""
    if requested == "cpu":
        return "cpu", "int8"
    if sys.platform == "win32":
        _add_windows_cuda_dll_dirs()
    if requested == "cuda":
        return "cuda", "float16"
    if _cuda_device_count() > 0:
        if _cuda_libraries_present():
            return "cuda", "float16"
        log.warning(
            "An NVIDIA GPU was found, but the CUDA libraries are missing, so the CPU "
            "will be used. For GPU speed install the extra: pip install \"hebrew-stt[cuda]\""
        )
    return "cpu", "int8"


def _is_cuda_error(exc: Exception) -> bool:
    return any(word in str(exc).lower() for word in ("cuda", "cublas", "cudnn"))


# --------------------------------------------------------------- transcriber

class Transcriber:
    """Loads the Whisper model once and transcribes files with it."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto"):
        self.model_name = model
        self.requested_device = device
        self.device: Optional[str] = None
        self._model = None

    def _model_path(self) -> str:
        """Local model folder; downloads from Hugging Face only if not cached yet."""
        if Path(self.model_name).is_dir():
            return self.model_name
        from faster_whisper.utils import download_model

        try:
            return download_model(self.model_name, local_files_only=True)
        except Exception:
            log.info("Downloading %s (about 1.6 GB, first run only)...", self.model_name)
            return download_model(self.model_name)

    def _load(self, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        path = self._model_path()
        log.info("Loading the model on the %s (%s)...", "GPU" if device == "cuda" else "CPU", compute_type)
        cpu_threads = min(8, os.cpu_count() or 4) if device == "cpu" else 0
        self._model = WhisperModel(
            path, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )
        self.device = device

    def _can_fall_back(self) -> bool:
        return self.device == "cuda" and self.requested_device == "auto"

    def load(self) -> None:
        if self._model is not None:
            return
        device, compute_type = detect_device(self.requested_device)
        try:
            self._load(device, compute_type)
        except Exception as exc:
            if device != "cuda" or self.requested_device != "auto":
                raise
            log.warning("Could not load the model on the GPU (%s). Using the CPU.", exc)
            self._load("cpu", "int8")

    def transcribe(self, media: Path, progress: Optional[ProgressCallback] = None) -> Transcript:
        self.load()
        try:
            return self._run(media, progress)
        except RuntimeError as exc:
            if not (self._can_fall_back() and _is_cuda_error(exc)):
                raise
            log.warning("GPU transcription failed (%s). Retrying on the CPU.", exc)
            self._model = None
            self._load("cpu", "int8")
            return self._run(media, progress)

    def _run(self, media: Path, progress: Optional[ProgressCallback]) -> Transcript:
        segments, info = self._model.transcribe(
            str(media),
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        total = float(info.duration or 0.0)
        result: list[Segment] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                result.append(Segment(seg.start, seg.end, text))
            if progress and total > 0:
                progress(min(1.0, seg.end / total), seg.end, total)
        if progress:
            progress(1.0, total, total)
        return Transcript(result, total)
