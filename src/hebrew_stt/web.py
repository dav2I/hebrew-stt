"""Browser UI: a small server on 127.0.0.1 that only this computer can reach.

Every request must carry a random token (printed in the terminal and put in the
URL that opens automatically), and the Host header must be 127.0.0.1/localhost,
so other websites open in the browser cannot talk to it.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import BinaryIO, Optional
from urllib.parse import parse_qs, quote, urlparse

from .core import Cancelled, process_file
from .summarize import SummarySettings, load_settings, read_saved_settings, save_settings, settings_file
from .transcribe import MEDIA_EXTENSIONS, OutputPaths, Transcriber, safe_stem

log = logging.getLogger(__name__)

_CHUNK = 1 << 20
_RESULT_KINDS = ("text", "timestamps", "summary")
_FINISHED = ("done", "failed", "cancelled")
_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
_FORBIDDEN = (
    "<!doctype html><meta charset=utf-8><title>hebrew-stt</title>"
    "<p style='font-family:system-ui;margin:40px'>This link is missing its access key. "
    "Open the full link that <b>hebrew-stt</b> printed in its terminal window.</p>"
).encode()


def default_output_dir() -> Path:
    documents = Path.home() / "Documents"
    return (documents if documents.is_dir() else Path.home()) / "hebrew-stt"


def open_folder(folder: Path) -> None:
    if sys.platform == "win32":
        os.startfile(folder)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])


@dataclass
class Job:
    id: str
    name: str
    media: Path
    summarize: bool
    state: str = "queued"  # queued, loading, transcribing, summarizing, done, failed, cancelled
    progress: float = 0.0
    position: float = 0.0
    duration: float = 0.0
    error: str = ""
    summary_error: str = ""
    summary_cancelled: bool = False
    paths: Optional[OutputPaths] = None
    stop: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    holds_name: bool = field(default=True, repr=False)  # its result name is still reserved

    def to_json(self) -> dict:
        files = {}
        if self.paths and self.state == "done":
            for kind in _RESULT_KINDS:
                path = getattr(self.paths, kind)
                if path.is_file():
                    files[kind] = path.name
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "progress": round(self.progress, 4),
            "position": self.position,
            "duration": self.duration,
            "error": self.error,
            "summary_error": self.summary_error,
            "summary_cancelled": self.summary_cancelled,
            "files": files,
        }


class WebApp:
    """Upload queue and one worker thread, so files are transcribed one at a time."""

    def __init__(self, transcriber: Transcriber, output_dir: Path):
        self.transcriber = transcriber
        self.output_dir = Path(output_dir)
        self.token = secrets.token_urlsafe(24)
        self.upload_dir = Path(tempfile.mkdtemp(prefix="hebrew-stt-"))
        self.jobs: dict[str, Job] = {}
        self._reserved: set[str] = set()
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Job]" = queue.Queue()
        threading.Thread(target=self._worker, name="hebrew-stt-worker", daemon=True).start()

    # ------------------------------------------------------------- uploads

    def _reserve_stem(self, name: str) -> str:
        """A result name that does not overwrite earlier results in the output folder."""
        base = safe_stem(Path(name).stem)
        stem, n = base, 2
        while stem.lower() in self._reserved or (self.output_dir / f"{stem}.txt").exists():
            stem, n = f"{base} ({n})", n + 1
        self._reserved.add(stem.lower())
        return stem

    def _release_name(self, job: Job) -> None:
        """Called with the lock held. Once released, the files on disk decide whether the name is taken."""
        if job.holds_name:
            self._reserved.discard(job.media.stem.lower())
            job.holds_name = False

    def add_upload(self, name: str, stream: BinaryIO, length: int, summarize: bool) -> Job:
        suffix = Path(name).suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            # No name in the message: the page shows it on the file's card, which already has the name.
            raise ValueError("Not an audio or video file.")
        with self._lock:
            stem = self._reserve_stem(name)
        job_id = uuid.uuid4().hex[:12]
        folder = self.upload_dir / job_id
        folder.mkdir()
        media = folder / f"{stem}{suffix}"
        try:
            remaining = length
            with open(media, "wb") as fh:
                while remaining > 0:
                    chunk = stream.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise ConnectionError("The upload was interrupted.")
                    fh.write(chunk)
                    remaining -= len(chunk)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            with self._lock:
                self._reserved.discard(stem.lower())
            raise
        job = Job(job_id, name, media, summarize)
        with self._lock:
            self.jobs[job_id] = job
        self._queue.put(job)
        return job

    def cancel(self, job_id: str) -> Optional[Job]:
        """Stop a job. Before its transcript is written nothing is saved; during the
        summary the transcript is kept and the summary is dropped."""
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None or job.state in _FINISHED:
                return job
            job.stop.set()
            if job.state != "summarizing":
                job.state = "cancelled"  # the worker notices at the next segment
                self._release_name(job)  # so the same file can be dropped again right away
            return job

    # -------------------------------------------------------------- worker

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._run(job)
            except Cancelled:
                job.state = "cancelled"
            except Exception as exc:
                log.exception("Failed: %s", job.name)
                job.error = str(exc) or type(exc).__name__
                job.state = "failed"
            finally:
                shutil.rmtree(job.media.parent, ignore_errors=True)
                with self._lock:
                    self._release_name(job)

    def _set_state(self, job: Job, state: str) -> None:
        with self._lock:
            if not job.stop.is_set():
                job.state = state

    def _run(self, job: Job) -> None:
        if job.stop.is_set():
            raise Cancelled()
        if not self.transcriber.loaded:
            self._set_state(job, "loading")
            self.transcriber.load()
        self._set_state(job, "transcribing")
        settings = load_settings() if job.summarize else None

        def progress(fraction: float, position: float, total: float) -> None:
            job.progress, job.position, job.duration = fraction, position, total

        def status(_text: str) -> None:
            self._set_state(job, "summarizing")

        result = process_file(job.media, self.transcriber, settings, self.output_dir, progress, status, job.stop)
        with self._lock:
            job.paths = result.paths
            job.duration = result.duration
            job.summary_error = result.summary_error or ""
            job.summary_cancelled = result.summary_cancelled
            if job.state != "cancelled":  # Stop that landed while the files were being written
                job.state = "done"

    # ------------------------------------------------------------ settings

    def settings_payload(self) -> dict:
        effective, saved = load_settings(), read_saved_settings()
        key = saved.api_key
        return {
            "enabled": effective.enabled,
            "model": effective.model,
            "host": urlparse(effective.base_url).netloc or effective.base_url,
            "saved": {
                "has_key": bool(key),
                "key_hint": key[-4:] if len(key) >= 8 else "",
                "base_url": saved.base_url,
                "model": saved.model,
            },
            "external": effective != saved,
            "path": str(settings_file()),
        }

    def update_settings(self, data: dict) -> dict:
        saved = read_saved_settings()
        key = data.get("api_key")
        defaults = SummarySettings()
        save_settings(SummarySettings(
            api_key=saved.api_key if key is None else str(key).strip(),
            base_url=str(data.get("base_url") or "").strip() or defaults.base_url,
            model=str(data.get("model") or "").strip() or defaults.model,
        ))
        return self.settings_payload()

    def close(self) -> None:
        shutil.rmtree(self.upload_dir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "hebrew-stt"
    page: bytes = b""

    @property
    def app(self) -> WebApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        log.debug("%s %s", self.address_string(), format % args)

    # ------------------------------------------------------------- helpers

    def _allowed(self) -> bool:
        port = self.server.server_address[1]
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return False
        token = self.headers.get("X-Token") or parse_qs(urlparse(self.path).query).get("t", [""])[0]
        return secrets.compare_digest(token.encode(), self.app.token.encode())

    def _send(self, status: int, body: bytes, content_type: str, extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _guard(self) -> bool:
        if self._allowed():
            return True
        self._send(403, _FORBIDDEN, "text/html; charset=utf-8")
        return False

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 64_000:
            raise ValueError("Request too large.")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    # -------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        url = urlparse(self.path)
        if url.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if not self._guard():
            return
        if url.path == "/":
            return self._send(200, self.page, "text/html; charset=utf-8", {"Content-Security-Policy": _CSP})
        if url.path == "/api/jobs":
            jobs = [job.to_json() for job in list(self.app.jobs.values())]
            return self._json({"jobs": jobs, "output_dir": str(self.app.output_dir)})
        if url.path == "/api/settings":
            return self._json(self.app.settings_payload())
        parts = url.path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
            return self._result(parts[2], parts[3], "download" in parse_qs(url.query))
        self._error(404, "Not found.")

    def _result(self, job_id: str, kind: str, download: bool) -> None:
        job = self.app.jobs.get(job_id)
        if job is None or job.paths is None or kind not in _RESULT_KINDS:
            return self._error(404, "Not found.")
        path: Path = getattr(job.paths, kind)
        if not path.is_file():
            return self._error(404, "Not found.")
        content_type = "text/markdown; charset=utf-8" if path.suffix == ".md" else "text/plain; charset=utf-8"
        extra = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(path.name)}"} if download else None
        self._send(200, path.read_bytes(), content_type, extra)

    def do_PUT(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if not self._guard():
            return
        if url.path != "/api/upload":
            return self._error(404, "Not found.")
        query = parse_qs(url.query)
        name = Path(query.get("name", [""])[0]).name
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._error(411, "Missing Content-Length.")
        try:
            job = self.app.add_upload(name, self.rfile, length, query.get("summary", ["0"])[0] == "1")
        except ValueError as exc:
            return self._error(400, str(exc))
        except (ConnectionError, OSError) as exc:
            log.warning("Upload of %s failed: %s", name, exc)
            return
        self._json(job.to_json(), 201)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if not self._guard():
            return
        if url.path == "/api/settings":
            try:
                return self._json(self.app.update_settings(self._read_json()))
            except (ValueError, OSError) as exc:
                return self._error(400, str(exc))
        if url.path == "/api/open-folder":
            self.app.output_dir.mkdir(parents=True, exist_ok=True)
            open_folder(self.app.output_dir)
            return self._json({"ok": True})
        parts = url.path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
            job = self.app.cancel(parts[2])
            return self._json(job.to_json()) if job else self._error(404, "Not found.")
        self._error(404, "Not found.")


def make_server(app: WebApp, port: int = 0) -> ThreadingHTTPServer:
    Handler.page = resources.files("hebrew_stt").joinpath("page.html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.app = app  # type: ignore[attr-defined]
    return server


def serve(
    model: str,
    device: str = "auto",
    output_dir: Optional[Path] = None,
    open_browser: bool = True,
) -> int:
    app = WebApp(Transcriber(model=model, device=device), Path(output_dir) if output_dir else default_output_dir())
    server = make_server(app)
    url = f"http://127.0.0.1:{server.server_address[1]}/?t={app.token}"
    print(
        f"\nhebrew-stt is running at:\n  {url}\n\n"
        f"Results are saved in: {app.output_dir}\n"
        "Keep this window open while you use the page. Press Ctrl+C to stop.\n",
        flush=True,
    )
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.server_close()
        app.close()
    return 0
