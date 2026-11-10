import pytest

from hebrew_stt.transcribe import Segment, Transcript, format_timestamp


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00"),
        (5.4, "00:05"),
        (59.99, "00:59"),
        (61, "01:01"),
        (3599, "59:59"),
        (3600, "01:00:00"),
        (3725.7, "01:02:05"),
        (36000, "10:00:00"),
        (-3, "00:00"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_timestamp_forced_hours():
    assert format_timestamp(5, with_hours=True) == "00:00:05"


def _transcript(duration):
    return Transcript(
        segments=[Segment(0.0, 2.5, "שלום"), Segment(62.0, 65.2, "מה שלומך?")],
        duration=duration,
    )


def test_timestamped_text_short_recording():
    assert _transcript(70).timestamped_text() == (
        "[00:00 -> 00:02] שלום\n"
        "[01:02 -> 01:05] מה שלומך?"
    )


def test_timestamped_text_uses_hours_for_long_recordings():
    lines = _transcript(4000).timestamped_text().splitlines()
    assert lines[0] == "[00:00:00 -> 00:00:02] שלום"
    assert lines[1].startswith("[00:01:02 -> 00:01:05]")


def test_plain_text_is_one_segment_per_line():
    assert _transcript(70).text == "שלום\nמה שלומך?"
