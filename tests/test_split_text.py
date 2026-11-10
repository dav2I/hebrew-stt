import pytest

from hebrew_stt.summarize import split_text

SAMPLE = "\n".join(f"זו שורה מספר {i}. ובה עוד משפט קצר!" for i in range(300))


def test_empty_text():
    assert split_text("") == []
    assert split_text("  \n\n ") == []


def test_short_text_is_one_chunk():
    assert split_text("  שלום עולם  ", 100) == ["שלום עולם"]


@pytest.mark.parametrize("max_chars", [50, 200, 1000, 5000])
def test_chunks_respect_limit_and_keep_every_word(max_chars):
    chunks = split_text(SAMPLE, max_chars)
    assert len(chunks) > 1
    assert all(0 < len(c) <= max_chars for c in chunks)
    assert " ".join(chunks).split() == SAMPLE.split()


def test_prefers_line_boundaries():
    chunks = split_text(SAMPLE, 1000)
    assert all(c.endswith("!") for c in chunks)


def test_long_line_splits_between_sentences():
    line = " ".join(f"משפט {i}." for i in range(100))
    chunks = split_text(line, 40)
    assert all(len(c) <= 40 for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert " ".join(chunks).split() == line.split()


def test_word_longer_than_limit_is_cut():
    word = "x" * 95
    chunks = split_text(f"start {word} end", 30)
    assert all(len(c) <= 30 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == f"start{word}end"


def test_invalid_limit():
    with pytest.raises(ValueError):
        split_text("abc", 0)
