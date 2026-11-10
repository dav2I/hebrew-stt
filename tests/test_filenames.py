from pathlib import Path

import pytest

from hebrew_stt.core import collect_media
from hebrew_stt.transcribe import output_paths, safe_stem


@pytest.mark.parametrize(
    "name, expected",
    [
        ("lecture 1", "lecture 1"),
        ("הרצאה 3 - מבוא", "הרצאה 3 - מבוא"),
        ('a:b*c?d"e<f>g|h', "abcdefgh"),
        ("back\\slash/forward", "backslashforward"),
        ("tabs\tand   spaces", "tabs and spaces"),
        ("  trailing dots... ", "trailing dots"),
        ("", "transcript"),
        ("???", "transcript"),
        ("CON", "_CON"),
        ("nul.backup", "_nul.backup"),
        ("com1", "_com1"),
        ("console", "console"),
    ],
)
def test_safe_stem(name, expected):
    assert safe_stem(name) == expected


def test_safe_stem_limits_length():
    result = safe_stem("א" * 400)
    assert len(result) == 150


def test_output_paths_next_to_source():
    paths = output_paths(Path("recordings") / "meeting: notes.mp3")
    assert paths.text == Path("recordings") / "meeting notes.txt"
    assert paths.timestamps == Path("recordings") / "meeting notes_timestamps.txt"
    assert paths.summary == Path("recordings") / "meeting notes_summary.md"


def test_output_paths_custom_folder():
    paths = output_paths(Path("in") / "talk.wav", Path("out"))
    assert paths.text == Path("out") / "talk.txt"


def test_collect_media_filters_and_expands_folders(tmp_path):
    for name in ("a.mp3", "b.TXT", "c.WAV", "d.mp4"):
        (tmp_path / name).write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.mp3").write_bytes(b"")

    found = collect_media([tmp_path, tmp_path / "a.mp3"])
    assert [p.name for p in found] == ["a.mp3", "c.WAV", "d.mp4"]
