from types import SimpleNamespace

import pytest

from hebrew_stt.transcribe import (
    LANGUAGES,
    MAX_TERMS_CHARS,
    Transcriber,
    TranscriberPool,
    clean_terms,
    guess_language,
)


class FakeWhisper:
    """Stands in for faster_whisper.WhisperModel and records how it was called."""

    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        segments = [SimpleNamespace(start=0.0, end=1.5, text=" hello ")]
        return iter(segments), SimpleNamespace(duration=1.5)


def test_each_language_has_its_own_model_and_settings():
    for code in ("he", "en"):
        transcriber = Transcriber(language=code)
        transcriber._model = FakeWhisper()
        transcript = transcriber.transcribe("talk.mp3")
        call = transcriber._model.calls[0]
        assert transcriber.model_name == LANGUAGES[code].model
        assert call["language"] == code
        assert call["condition_on_previous_text"] is (code == "he")
        assert transcript.text == "hello"


def test_guess_language_from_the_letters():
    assert guess_language("שלום לכולם, today") == "he"
    assert guess_language("Hello everyone, שלום") == "en"
    assert guess_language("") == "he"


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError, match="Unknown language: fr"):
        Transcriber(language="fr")


def test_pool_keeps_only_the_last_language_loaded():
    pool = TranscriberPool(models={"en": "distil-small.en"})
    hebrew = pool("he")
    hebrew._model = FakeWhisper()
    english = pool("en")
    assert english.language == "en" and english.model_name == "distil-small.en"
    assert hebrew.model_name == LANGUAGES["he"].model
    assert not hebrew.loaded
    assert pool("he") is hebrew


def test_names_and_terms_are_given_to_the_model_as_hotwords():
    transcriber = Transcriber()
    transcriber._model = FakeWhisper()
    transcriber.transcribe("talk.mp3", terms="  Dr. Cohen,\n Dijkstra ")
    transcriber.transcribe("talk.mp3")
    first, second = transcriber._model.calls
    assert first["hotwords"] == "Dr. Cohen, Dijkstra"
    assert second["hotwords"] is None


def test_names_and_terms_have_a_length_limit():
    assert clean_terms("x" * MAX_TERMS_CHARS) == "x" * MAX_TERMS_CHARS
    with pytest.raises(ValueError, match="limited to 300 characters"):
        clean_terms("x" * (MAX_TERMS_CHARS + 1))
