"""The page shows file names and paths as isolated left-to-right text.

Laid out right-to-left (dir="auto" or a bare <bdi>), a name that starts with Hebrew
shows its extension on the wrong side: "mp3.שיעור" instead of "שיעור.mp3".
These tests check the page source; they do not render it.
"""
import re
from html.parser import HTMLParser
from importlib import resources

PAGE = resources.files("hebrew_stt").joinpath("page.html").read_text(encoding="utf-8")
SCRIPT = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]


class ElementsById(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found = {}
        self.auto_direction = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.found[attrs["id"]] = (tag, attrs)
        if attrs.get("dir") == "auto":
            self.auto_direction.append(tag)


def test_file_name_helper_is_an_ltr_isolate():
    helper = re.search(r"function fileName\(text\) \{(.*?)\n\}", SCRIPT, re.S)
    assert helper, "page.html has no fileName() helper"
    assert 'el("bdi", { dir: "ltr", textContent: text })' in helper.group(1)


def test_output_folder_is_an_ltr_isolate():
    parser = ElementsById()
    parser.feed(PAGE)
    tag, attrs = parser.found["out-dir"]
    assert (tag, attrs.get("dir")) == ("bdi", "ltr")


def test_names_and_paths_are_shown_through_the_helper():
    assert 'el("span", { className: "job-name" }, fileName(name))' in SCRIPT
    assert "fileName(s.path)" in SCRIPT
    assert not re.search(r"textContent:\s*(name|file\.name|job\.name)\b", SCRIPT)
    assert "${s.path}" not in SCRIPT


def test_only_content_picks_its_own_direction():
    # Transcripts, summaries, questions and answers are content, so the browser picks
    # right-to-left for Hebrew. Names never do (they go through fileName()).
    automatic = re.findall(r'el\("\w+", \{[^}]*\bdir: "auto"[^}]*\}', SCRIPT)
    assert 'el("pre", { dir: "auto", textContent: "Loading..." }' in automatic
    assert 'el("span", { dir: "auto", textContent: seg.text }' in automatic
    assert not [found for found in automatic if re.search(r"\bname\b|job-name|fileName", found)]
    parser = ElementsById()
    parser.feed(PAGE)
    assert parser.auto_direction == []
