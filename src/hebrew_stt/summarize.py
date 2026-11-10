"""Optional AI summary through any OpenAI-compatible chat API.

Only this step sends data off the computer: the transcript text goes to the
provider you configure. Without an API key it is skipped entirely.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

APP_NAME = "hebrew-stt"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
# Characters per request. Fits comfortably in any modern 128k-token model;
# longer transcripts are summarized part by part and then merged.
CHUNK_CHARS = 100_000

KEY_VAR, URL_VAR, MODEL_VAR = "SUMMARY_API_KEY", "SUMMARY_BASE_URL", "SUMMARY_MODEL"
_VARS = (KEY_VAR, URL_VAR, MODEL_VAR)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ settings

@dataclass
class SummarySettings:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.strip())


def config_dir() -> Path:
    """Per-user settings folder, outside any project or repository."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_NAME


def settings_file() -> Path:
    return config_dir() / "settings.env"


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    from dotenv import dotenv_values

    return {k: v.strip() for k, v in dotenv_values(path).items() if k in _VARS and v and v.strip()}


def _to_settings(values: Mapping[str, str]) -> SummarySettings:
    return SummarySettings(
        api_key=values.get(KEY_VAR, ""),
        base_url=values.get(URL_VAR) or DEFAULT_BASE_URL,
        model=values.get(MODEL_VAR) or DEFAULT_MODEL,
    )


def read_saved_settings(path: Optional[Path] = None) -> SummarySettings:
    """Only what was saved from the app's Settings window."""
    return _to_settings(_read_env_file(path or settings_file()))


def load_settings(
    environ: Optional[Mapping[str, str]] = None,
    dotenv_path: Optional[Path] = None,
    saved_path: Optional[Path] = None,
) -> SummarySettings:
    """Effective settings. Precedence: environment, then ./.env, then saved settings."""
    environ = os.environ if environ is None else environ
    values = _read_env_file(saved_path or settings_file())
    values.update(_read_env_file(dotenv_path or Path.cwd() / ".env"))
    values.update({k: environ[k].strip() for k in _VARS if environ.get(k, "").strip()})
    return _to_settings(values)


def _quote(value: str) -> str:
    # Keys, URLs and model names never need these; rejecting them avoids escaping.
    if any(ch in value for ch in "\"'\\\r\n"):
        raise ValueError("Settings values cannot contain quotes, backslashes or line breaks.")
    return f'"{value}"'


def save_settings(settings: SummarySettings, path: Optional[Path] = None) -> Path:
    """Save to the per-user settings file, readable only by the current user."""
    path = path or settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        f"{name}={_quote(value.strip())}\n"
        for name, value in (
            (KEY_VAR, settings.api_key),
            (URL_VAR, settings.base_url),
            (MODEL_VAR, settings.model),
        )
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    if os.name == "posix":
        os.chmod(path, 0o600)
    return path


# ------------------------------------------------------------------ chunking

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _split_words(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:max_chars])
            word = word[max_chars:]
        if not word:
            continue
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split text into chunks of at most max_chars characters.

    Breaks between lines where possible, then between sentences, then between
    words; a single word longer than max_chars is cut.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            pieces.append(line)
            continue
        for sentence in _SENTENCE_END.split(line):
            if len(sentence) <= max_chars:
                pieces.append(sentence)
            else:
                pieces.extend(_split_words(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------------- prompts

SYSTEM_PROMPT = """אתה מומחה לניתוח וסיכום של הקלטות אודיו ווידאו.

שלב 1 - זהה את סוג התוכן: הרצאה או שיעור, פגישת עבודה, פודקאסט או ראיון, \
הדרכה או וובינר, הקלטה אישית, או דיון מקצועי.

שלב 2 - כתוב סיכום שמותאם לסוג התוכן:
- הרצאה או שיעור: תמצית, מושגי מפתח והגדרות, הנושאים לפי הסדר, דגשים חשובים.
- פגישת עבודה: מטרה, נושאים שנדונו, החלטות, משימות (עם אחראים ומועדים אם צוינו), שאלות פתוחות.
- פודקאסט או ראיון: המסר המרכזי, תובנות מפתח, רעיונות בולטים, מסקנות.
- הדרכה או וובינר: מה נלמד, שלבים מעשיים, המלצות וטעויות נפוצות.
- הקלטה אישית או אחר: תמצית, עיקרי הדברים, צעדים הבאים.

כתוב בעברית ברורה, בפורמט Markdown. בראש הסיכום ציין את סוג התוכן שזיהית. \
אל תוסיף מידע שלא מופיע בהקלטה."""

PART_PROMPT = """אתה מקבל חלק אחד מתוך תמלול ארוך של הקלטה.
כתוב בעברית סיכום ביניים מפורט של החלק הזה: הנושאים, הטיעונים, המושגים, \
ההחלטות והפרטים החשובים, לפי סדר הופעתם. אל תוסיף מידע שלא מופיע בתמלול. \
סיכום הביניים ישמש לבניית סיכום כולל של כל ההקלטה."""


# ----------------------------------------------------------------- summarize

def _chat(client, model: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("The summary provider returned an empty answer.")
    return content.strip()


def summarize(
    text: str,
    settings: SummarySettings,
    title: Optional[str] = None,
    client=None,
    max_chars: int = CHUNK_CHARS,
) -> str:
    """Summarize a transcript. Long transcripts are summarized in parts, then merged."""
    if not settings.enabled:
        raise ValueError("No API key is configured for the summary.")
    chunks = split_text(text, max_chars)
    if not chunks:
        raise ValueError("The transcript is empty.")
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    if len(chunks) == 1:
        intro, material = "להלן תמלול ההקלטה", chunks[0]
    else:
        notes = []
        for i, chunk in enumerate(chunks, start=1):
            log.info("Summarizing part %d of %d...", i, len(chunks))
            notes.append(_chat(client, settings.model, PART_PROMPT, f"חלק {i} מתוך {len(chunks)}:\n\n{chunk}"))
        intro = "להלן סיכומי ביניים של חלקי ההקלטה, לפי הסדר"
        material = "\n\n".join(f"## חלק {i}\n{note}" for i, note in enumerate(notes, start=1))

    named = f" ({title})" if title else ""
    body = _chat(client, settings.model, SYSTEM_PROMPT, f"{intro}{named}:\n\n{material}")
    header = f"# סיכום: {title}\n\n" if title else ""
    return f"{header}{body}\n"
