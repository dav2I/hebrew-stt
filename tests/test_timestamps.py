import pytest

from hebrew_stt.transcribe import Segment, Transcript, format_timestamp, parse_srt, srt_time


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


def test_srt_times_have_milliseconds_and_hours():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(61.2345) == "00:01:01,234"
    assert srt_time(3725.5) == "01:02:05,500"
    assert srt_time(-1) == "00:00:00,000"


def test_subtitles_round_trip():
    transcript = Transcript([Segment(0.0, 2.5, "שלום לכולם"), Segment(2.5, 3661.25, "ועכשיו נושא חדש")], 3662)
    srt = transcript.srt()
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nשלום לכולם\n\n2\n00:00:02,500 --> 01:01:01,250\n")
    assert parse_srt(srt) == transcript.segments
    assert parse_srt(srt.replace("\n", "\r\n")) == transcript.segments


def test_subtitles_from_other_programs():
    text = "1\n00:00:01.000 --> 00:00:02.000\nfirst line\nsecond line\n\n2\n00:00:03,000 --> 00:00:04,000\n\n"
    assert parse_srt(text) == [Segment(1.0, 2.0, "first line second line")]
    assert Transcript([], 0).srt() == "" and parse_srt("") == []
