import http.client
import json
import os
import threading
import time
from urllib.parse import quote

import pytest

from hebrew_stt import core, web
from hebrew_stt.transcribe import Segment, Transcript


class FakeTranscriber:
    loaded = True

    def load(self):
        pass

    def transcribe(self, media, progress=None, terms=""):
        if progress:
            progress(1.0, 2.0, 2.0)
        return Transcript([Segment(0, 2, "שלום")], 2.0)


class FakePool:
    """Stands in for TranscriberPool: one transcriber for every language, and a log of languages."""

    def __init__(self):
        self.transcriber = FakeTranscriber()
        self.languages = []

    def __call__(self, language):
        self.languages.append(language)
        return self.transcriber


@pytest.fixture
def server(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for var in ("APPDATA", "XDG_CONFIG_HOME", "HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    for var in ("SUMMARY_API_KEY", "SUMMARY_BASE_URL", "SUMMARY_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    app = web.WebApp(FakePool(), tmp_path / "out")
    srv = web.make_server(app)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, srv.server_address[1]
    srv.shutdown()
    srv.server_close()
    app.close()


def call(port, method, path, body=None, token=None, host=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    if token:
        conn.putheader("X-Token", token)
    for name, value in (headers or {}).items():
        conn.putheader(name, value)
    if body is not None:
        conn.putheader("Content-Length", str(len(body)))
    conn.endheaders(body)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, response.headers, data


def upload(port, token, name, body=b"fake audio", summary=False, **params):
    extra = "".join(f"&{key}={quote(value)}" for key, value in params.items())
    path = f"/api/upload?name={quote(name)}&summary={int(summary)}{extra}"
    status, _, data = call(port, "PUT", path, body, token)
    return status, json.loads(data)


def cancel(port, token, job_id):
    status, _, data = call(port, "POST", f"/api/jobs/{job_id}/cancel", b"", token)
    return status, json.loads(data)


def wait_state(port, token, job_id, states):
    for _ in range(200):
        _, _, data = call(port, "GET", "/api/jobs", token=token)
        job = next(j for j in json.loads(data)["jobs"] if j["id"] == job_id)
        if job["state"] in states:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job never reached {states}")


def wait_done(port, token, job_id):
    return wait_state(port, token, job_id, ("done", "failed", "cancelled"))


class SlowTranscriber(FakeTranscriber):
    """Runs until the test releases it, reporting progress like a long recording."""

    def __init__(self):
        self.release = threading.Event()

    def transcribe(self, media, progress=None, terms=""):
        while not self.release.wait(0.02):
            progress(0.5, 1.0, 2.0)
        return super().transcribe(media, progress)


def test_page_needs_the_token(server):
    app, port = server
    assert call(port, "GET", "/")[0] == 403
    assert call(port, "GET", "/?t=wrong")[0] == 403
    status, headers, body = call(port, "GET", f"/?t={app.token}")
    assert status == 200
    assert "media-src blob:" in headers["Content-Security-Policy"]  # the player plays local files
    assert b"<title>Hebrew STT</title>" in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_other_hosts_are_rejected(server):
    app, port = server
    status, _, _ = call(port, "GET", "/api/jobs", token=app.token, host=f"attacker.example:{port}")
    assert status == 403


def test_upload_transcribe_and_download(server, tmp_path):
    app, port = server
    status, job = upload(port, app.token, "הרצאה 1.mp3")
    assert status == 201
    job = wait_done(port, app.token, job["id"])
    assert job["state"] == "done"
    assert job["files"] == {
        "text": "הרצאה 1.txt", "timestamps": "הרצאה 1_timestamps.txt", "subtitles": "הרצאה 1.srt",
    }

    assert job["result"] == "הרצאה 1"
    base = f"/api/results/{quote(job['result'])}"
    status, _, body = call(port, "GET", f"{base}/text", token=app.token)
    assert status == 200 and body.decode("utf-8") == "שלום\n"
    status, headers, _ = call(port, "GET", f"{base}/timestamps?download=1", token=app.token)
    assert "attachment" in headers["Content-Disposition"]
    status, headers, body = call(port, "GET", f"{base}/subtitles", token=app.token)
    assert headers["Content-Type"].startswith("application/x-subrip")
    assert body.decode("utf-8") == "1\n00:00:00,000 --> 00:00:02,000\nשלום\n"
    assert (tmp_path / "out" / "הרצאה 1.txt").is_file()
    assert not list(app.upload_dir.iterdir())  # the uploaded copy is deleted afterwards


def test_same_name_does_not_overwrite_earlier_results(server):
    app, port = server
    first = wait_done(port, app.token, upload(port, app.token, "talk.wav")[1]["id"])
    second = wait_done(port, app.token, upload(port, app.token, "talk.wav")[1]["id"])
    assert first["files"]["text"] == "talk.txt"
    assert second["files"]["text"] == "talk (2).txt"


def test_unsupported_upload_is_rejected(server):
    app, port = server
    status, data = upload(port, app.token, "הערות.txt")
    # The page shows this error on the file's card, so the name is not repeated in it.
    assert (status, data["error"]) == (400, "Not an audio or video file.")


def test_a_rejected_large_upload_still_gets_its_answer(server):
    # The server must read what was sent before answering: on Windows, closing a connection
    # with unread data resets it, and the page would never see the error.
    app, port = server
    for _ in range(3):
        status, data = upload(port, app.token, "talk.mp3", body=b"x" * 8_000_000, style="poem")
        assert status == 400 and data["error"].startswith("Unknown summary style")
        status, _, data = call(port, "POST", "/api/settings", b"{" + b" " * 2_000_000 + b"}", app.token)
        assert status == 400 and json.loads(data)["error"] == "Request too large."
        assert call(port, "PUT", "/api/upload?name=a.mp3", b"x" * 8_000_000, "wrong-token")[0] == 403


def test_stop_a_running_and_a_queued_file(server, tmp_path):
    app, port = server
    slow = app.transcribers.transcriber = SlowTranscriber()
    running = upload(port, app.token, "long.mp3")[1]
    queued = upload(port, app.token, "next.mp3")[1]
    wait_state(port, app.token, running["id"], ("transcribing",))

    assert cancel(port, app.token, queued["id"]) == (200, {**queued, "state": "cancelled"})
    assert cancel(port, app.token, running["id"])[1]["state"] == "cancelled"

    # The worker drops both, so the next file runs, and the stopped name is free again.
    slow.release.set()
    again = wait_done(port, app.token, upload(port, app.token, "long.mp3")[1]["id"])
    assert again["files"]["text"] == "long.txt"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["long.srt", "long.txt", "long_timestamps.txt"]
    for job in (running, queued):
        assert wait_done(port, app.token, job["id"])["state"] == "cancelled"


def test_skip_summary_keeps_the_transcript(server, monkeypatch):
    app, port = server
    monkeypatch.setenv("SUMMARY_API_KEY", "test-key")
    started, release = threading.Event(), threading.Event()

    def slow_summary(*_args, **_kwargs):
        started.set()
        release.wait(10)
        return "too late"

    monkeypatch.setattr(core, "summarize", slow_summary)
    job = upload(port, app.token, "talk.mp3", summary=True)[1]
    try:
        assert started.wait(10)
        assert cancel(port, app.token, job["id"])[1]["state"] == "summarizing"
        job = wait_done(port, app.token, job["id"])
    finally:
        release.set()
    assert job["state"] == "done" and job["summary_cancelled"]
    assert job["files"] == {"text": "talk.txt", "timestamps": "talk_timestamps.txt", "subtitles": "talk.srt"}


def test_stop_leaves_finished_and_unknown_files_alone(server):
    app, port = server
    job = wait_done(port, app.token, upload(port, app.token, "talk.wav")[1]["id"])
    assert cancel(port, app.token, job["id"]) == (200, job)
    assert call(port, "POST", "/api/jobs/nope/cancel", b"", app.token)[0] == 404


def test_upload_takes_a_summary_style(server):
    app, port = server
    status, job = upload(port, app.token, "talk.mp3", style="lecture")
    assert status == 201 and app.jobs[job["id"]].style == "lecture"
    status, data = upload(port, app.token, "talk.mp3", style="poem")
    assert status == 400 and data["error"].startswith("Unknown summary style: poem.")
    _, _, body = call(port, "GET", "/api/settings", token=app.token)
    styles = json.loads(body)["styles"]
    assert styles[0] == {"id": "general", "label": "General"} and len(styles) > 5


def test_upload_takes_a_language(server):
    app, port = server
    job = wait_done(port, app.token, upload(port, app.token, "talk.mp3", language="en")[1]["id"])
    assert job["state"] == "done" and app.transcribers.languages == ["en"]
    status, data = upload(port, app.token, "talk.mp3", language="fr")
    assert status == 400 and data["error"].startswith("Unknown language: fr.")
    _, _, body = call(port, "GET", "/api/settings", token=app.token)
    settings = json.loads(body)
    assert settings["languages"] == [{"id": "he", "label": "Hebrew"}, {"id": "en", "label": "English"}]
    assert settings["default_language"] == "he"


def test_upload_takes_names_and_terms(server):
    app, port = server
    status, job = upload(port, app.token, "talk.mp3", terms="Dr. Cohen,  Dijkstra")
    assert status == 201 and app.jobs[job["id"]].terms == "Dr. Cohen, Dijkstra"
    status, data = upload(port, app.token, "talk.mp3", terms="x" * 301)
    assert status == 400 and "limited to 300 characters" in data["error"]


def test_past_results_are_listed_newest_first(server, tmp_path):
    app, port = server
    out = tmp_path / "out"
    out.mkdir()
    for i, stem in enumerate(["old lecture", "שיחה"]):
        (out / f"{stem}.txt").write_text("text", encoding="utf-8")
        (out / f"{stem}_timestamps.txt").write_text("[00:00 -> 00:01] text", encoding="utf-8")
        os.utime(out / f"{stem}.txt", (1_700_000_000 + i, 1_700_000_000 + i))
    (out / "שיחה_summary.md").write_text("# סיכום", encoding="utf-8")
    _, _, body = call(port, "GET", "/api/results", token=app.token)
    results = json.loads(body)["results"]
    assert [r["name"] for r in results] == ["שיחה", "old lecture"]
    assert results[0]["files"] == {"text": "שיחה.txt", "timestamps": "שיחה_timestamps.txt", "summary": "שיחה_summary.md"}


def test_results_are_served_only_from_the_output_folder(server, tmp_path):
    app, port = server
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    for name in ("missing", "..%2Fsecret", "..%5Csecret", "%2E%2E%2Fsecret"):
        assert call(port, "GET", f"/api/results/{name}/text", token=app.token)[0] == 404
    assert call(port, "GET", "/api/results/x/summarize", token=app.token)[0] == 404


def test_summarize_a_result_again_in_another_style(server, tmp_path, monkeypatch):
    app, port = server
    job = wait_done(port, app.token, upload(port, app.token, "talk.mp3")[1]["id"])
    seen = {}

    def fake_summary(text, settings, **kwargs):
        seen.update(text=text, **kwargs)
        return f"# {kwargs['style_name']}\n"

    monkeypatch.setattr(web, "summarize", fake_summary)

    def again(style, terms=""):
        body = json.dumps({"style": style, "terms": terms}).encode()
        status, _, data = call(port, "POST", f"/api/results/{job['result']}/summarize", body, app.token,
                               headers={"Content-Type": "application/json"})
        return status, json.loads(data)

    status, data = again("lecture")
    assert status == 400 and "API key" in data["error"]  # no key yet
    monkeypatch.setenv("SUMMARY_API_KEY", "test-key")
    status, data = again("chapters", terms="Dr. Cohen")
    assert status == 200 and data["files"]["summary"] == "talk_summary.md"
    assert (tmp_path / "out" / "talk_summary.md").read_text(encoding="utf-8") == "# chapters\n"
    assert seen["text"] == "[00:00 -> 00:02] שלום\n"  # chapters reads the times
    assert seen["language"] == "he" and seen["terms"] == "Dr. Cohen" and seen["title"] == "talk"
    assert again("poem")[0] == 400


def test_segments_come_from_the_subtitles_or_the_timestamps(server, tmp_path):
    app, port = server
    job = wait_done(port, app.token, upload(port, app.token, "talk.mp3")[1]["id"])
    _, _, body = call(port, "GET", f"/api/results/{job['result']}/segments", token=app.token)
    assert json.loads(body) == {"segments": [{"start": 0.0, "end": 2.0, "text": "שלום"}]}
    (tmp_path / "out" / "older.txt").write_text("hi", encoding="utf-8")  # made before subtitles existed
    (tmp_path / "out" / "older_timestamps.txt").write_text("[00:05 -> 00:09] hi\n", encoding="utf-8")
    _, _, body = call(port, "GET", "/api/results/older/segments", token=app.token)
    assert json.loads(body)["segments"] == [{"start": 5.0, "end": 9.0, "text": "hi"}]
    assert call(port, "GET", "/api/results/missing/segments", token=app.token)[0] == 404


def test_ask_about_a_result(server, monkeypatch):
    app, port = server
    job = wait_done(port, app.token, upload(port, app.token, "talk.mp3")[1]["id"])
    seen = {}

    def fake_ask(transcript, question, settings, history=(), client=None):
        if not settings.enabled:
            raise ValueError("Add an API key in Settings to ask questions.")
        seen.update(transcript=transcript, question=question, history=history)
        return "At [00:01] they said hello."

    monkeypatch.setattr(web, "ask", fake_ask)

    def post(payload, name=job["result"]):
        status, _, data = call(port, "POST", f"/api/results/{name}/ask", json.dumps(payload).encode(),
                               app.token, headers={"Content-Type": "application/json"})
        return status, json.loads(data)

    assert post({"question": "What was said?"})[0] == 400  # no key yet
    monkeypatch.setenv("SUMMARY_API_KEY", "test-key")
    history = [{"question": "Who?", "answer": "Someone."}]
    assert post({"question": "What was said?", "history": history}) == (200, {"answer": "At [00:01] they said hello."})
    assert seen["transcript"] == "[00:00 -> 00:02] שלום\n"  # with times, to point at moments
    assert seen["history"] == [("Who?", "Someone.")]
    assert post({"question": "x", "history": "nope"})[0] == 400
    assert post({"question": "x"}, name="missing")[0] == 404


def test_settings_never_send_the_key_back(server):
    app, port = server
    key = "sk-test-1234567890"

    def post(payload):
        status, _, body = call(port, "POST", "/api/settings", json.dumps(payload).encode(), app.token,
                               headers={"Content-Type": "application/json"})
        assert status == 200
        assert key.encode() not in body
        return json.loads(body)

    saved = post({"api_key": key, "base_url": "", "model": "m1"})
    assert saved["enabled"] and saved["saved"]["key_hint"] == "7890"
    assert saved["model"] == "m1" and saved["host"] == "api.deepseek.com"

    kept = post({"api_key": None, "base_url": "https://example.com/v1", "model": "m2"})
    assert kept["saved"]["has_key"] and kept["host"] == "example.com"

    removed = post({"api_key": "", "base_url": "", "model": ""})
    assert not removed["enabled"] and not removed["saved"]["has_key"]
