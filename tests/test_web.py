import http.client
import json
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

    def transcribe(self, media, progress=None):
        if progress:
            progress(1.0, 2.0, 2.0)
        return Transcript([Segment(0, 2, "שלום")], 2.0)


@pytest.fixture
def server(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for var in ("APPDATA", "XDG_CONFIG_HOME", "HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    for var in ("SUMMARY_API_KEY", "SUMMARY_BASE_URL", "SUMMARY_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    app = web.WebApp(FakeTranscriber(), tmp_path / "out")
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


def upload(port, token, name, body=b"fake audio", summary=False):
    path = f"/api/upload?name={quote(name)}&summary={int(summary)}"
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

    def transcribe(self, media, progress=None):
        while not self.release.wait(0.02):
            progress(0.5, 1.0, 2.0)
        return super().transcribe(media, progress)


def test_page_needs_the_token(server):
    app, port = server
    assert call(port, "GET", "/")[0] == 403
    assert call(port, "GET", "/?t=wrong")[0] == 403
    status, headers, body = call(port, "GET", f"/?t={app.token}")
    assert status == 200
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
    assert job["files"] == {"text": "הרצאה 1.txt", "timestamps": "הרצאה 1_timestamps.txt"}

    status, _, body = call(port, "GET", f"/api/jobs/{job['id']}/text", token=app.token)
    assert status == 200 and body.decode("utf-8") == "שלום\n"
    status, headers, _ = call(port, "GET", f"/api/jobs/{job['id']}/timestamps?download=1", token=app.token)
    assert "attachment" in headers["Content-Disposition"]
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


def test_stop_a_running_and_a_queued_file(server, tmp_path):
    app, port = server
    app.transcriber = SlowTranscriber()
    running = upload(port, app.token, "long.mp3")[1]
    queued = upload(port, app.token, "next.mp3")[1]
    wait_state(port, app.token, running["id"], ("transcribing",))

    assert cancel(port, app.token, queued["id"]) == (200, {**queued, "state": "cancelled"})
    assert cancel(port, app.token, running["id"])[1]["state"] == "cancelled"

    # The worker drops both, so the next file runs, and the stopped name is free again.
    app.transcriber.release.set()
    again = wait_done(port, app.token, upload(port, app.token, "long.mp3")[1]["id"])
    assert again["files"]["text"] == "long.txt"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["long.txt", "long_timestamps.txt"]
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
    assert job["files"] == {"text": "talk.txt", "timestamps": "talk_timestamps.txt"}


def test_stop_leaves_finished_and_unknown_files_alone(server):
    app, port = server
    job = wait_done(port, app.token, upload(port, app.token, "talk.wav")[1]["id"])
    assert cancel(port, app.token, job["id"]) == (200, job)
    assert call(port, "POST", "/api/jobs/nope/cancel", b"", app.token)[0] == 404


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
