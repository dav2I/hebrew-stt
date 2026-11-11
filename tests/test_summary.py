import threading
from types import SimpleNamespace

import pytest

from hebrew_stt import core
from hebrew_stt.cli import main
from hebrew_stt.summarize import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SummarySettings,
    load_settings,
    read_saved_settings,
    save_settings,
    summarize,
)
from hebrew_stt.transcribe import Segment, Transcript


class FakeClient:
    """Stands in for openai.OpenAI; records requests and never touches the network."""

    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = f"answer {len(self.requests)}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


KEYED = SummarySettings(api_key="test-key")


# ------------------------------------------------------------------ settings

def test_defaults_when_nothing_is_configured(tmp_path):
    s = load_settings(environ={}, dotenv_path=tmp_path / ".env", saved_path=tmp_path / "saved.env")
    assert s == SummarySettings("", DEFAULT_BASE_URL, DEFAULT_MODEL)
    assert not s.enabled


def test_precedence_env_then_dotenv_then_saved(tmp_path):
    saved = tmp_path / "saved.env"
    save_settings(SummarySettings("saved-key", "https://saved.example", "saved-model"), saved)
    dotenv = tmp_path / ".env"
    dotenv.write_text('SUMMARY_MODEL=dotenv-model\nSUMMARY_BASE_URL=""\nOTHER=x\n', encoding="utf-8")

    s = load_settings(environ={"SUMMARY_API_KEY": "env-key"}, dotenv_path=dotenv, saved_path=saved)
    assert s.api_key == "env-key"
    assert s.model == "dotenv-model"
    assert s.base_url == "https://saved.example"  # empty values never override


def test_save_and_read_round_trip(tmp_path):
    path = tmp_path / "sub" / "settings.env"
    original = SummarySettings("sk-abc_123", "https://api.example.com/v1", "some-model")
    save_settings(original, path)
    assert read_saved_settings(path) == original


def test_save_rejects_values_that_need_escaping(tmp_path):
    with pytest.raises(ValueError):
        save_settings(SummarySettings('bad"key'), tmp_path / "settings.env")


# ----------------------------------------------------------------- summarize

def test_short_transcript_is_one_request():
    client = FakeClient()
    result = summarize("שלום עולם", KEYED, title="בדיקה", client=client)
    assert len(client.requests) == 1
    assert client.requests[0]["model"] == DEFAULT_MODEL
    assert "שלום עולם" in client.requests[0]["messages"][1]["content"]
    assert result == "# סיכום: בדיקה\n\nanswer 1\n"


def test_long_transcript_is_summarized_in_parts_then_merged():
    client = FakeClient()
    text = "\n".join(f"שורה {i}" for i in range(100))
    summarize(text, KEYED, client=client, max_chars=200)
    parts = len(client.requests) - 1
    assert parts > 1
    final = client.requests[-1]["messages"][1]["content"]
    assert all(f"answer {i}" in final for i in range(1, parts + 1))


def test_summary_needs_key_and_text():
    with pytest.raises(ValueError):
        summarize("text", SummarySettings(), client=FakeClient())
    with pytest.raises(ValueError):
        summarize("   ", KEYED, client=FakeClient())


# --------------------------------------------------------------- processing

class FakeTranscriber:
    def transcribe(self, media, progress=None):
        if progress:
            progress(1.0, 3.0, 3.0)
        return Transcript([Segment(0, 3, "שלום לכולם")], 3.0)


def test_process_file_without_summary(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"")
    result = core.process_file(media, FakeTranscriber())
    assert result.paths.text.read_text(encoding="utf-8") == "שלום לכולם\n"
    assert result.paths.timestamps.read_text(encoding="utf-8") == "[00:00 -> 00:03] שלום לכולם\n"
    assert not result.paths.summary.exists()
    assert not result.summarized and result.summary_error is None


def test_failed_summary_keeps_transcript(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(core, "summarize", broken)
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"")
    result = core.process_file(media, FakeTranscriber(), KEYED, output_dir=tmp_path / "out")
    assert result.paths.text.exists()
    assert result.summary_error == "ConnectionError: offline"


def test_stop_during_transcription_writes_nothing(tmp_path):
    stop = threading.Event()

    class StoppedMidway:
        def transcribe(self, media, progress=None):
            progress(0.5, 1.5, 3.0)  # first segment done
            stop.set()  # the user clicks Stop
            progress(1.0, 3.0, 3.0)  # the next segment notices
            raise AssertionError("transcription went on after Stop")

    media = tmp_path / "talk.mp3"
    media.write_bytes(b"")
    with pytest.raises(core.Cancelled):
        core.process_file(media, StoppedMidway(), output_dir=tmp_path / "out", stop=stop)
    assert not (tmp_path / "out" / "talk.txt").exists()


def test_stop_during_summary_keeps_the_transcript(tmp_path, monkeypatch):
    stop, started, release = threading.Event(), threading.Event(), threading.Event()

    def slow_summary(*_args, **_kwargs):  # a provider that takes its time
        started.set()
        release.wait(10)
        return "too late"

    monkeypatch.setattr(core, "summarize", slow_summary)
    threading.Thread(target=lambda: started.wait(10) and stop.set(), daemon=True).start()
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"")
    try:
        result = core.process_file(media, FakeTranscriber(), KEYED, output_dir=tmp_path / "out", stop=stop)
    finally:
        release.set()
    assert result.summary_cancelled and not result.summarized
    assert result.paths.text.exists()
    assert not result.paths.summary.exists()


def test_cli_rejects_missing_files(tmp_path):
    assert main([str(tmp_path / "missing.mp3")]) == 2
