"""The window: drop audio or video files, press Start."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional
from urllib.parse import urlparse

from .core import collect_media, process_file
from .summarize import SummarySettings, load_settings, read_saved_settings, save_settings, settings_file
from .transcribe import DEFAULT_MODEL, MEDIA_EXTENSIONS, Transcriber, format_timestamp

log = logging.getLogger(__name__)

WAITING, WORKING, DONE, FAILED = "Waiting", "Working...", "Done", "Failed"


def _open_folder(folder: Path) -> None:
    if sys.platform == "win32":
        os.startfile(folder)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])


def _enable_drag_and_drop(root: tk.Tk) -> bool:
    try:
        from tkinterdnd2 import TkinterDnD

        TkinterDnD.require(root)
        return True
    except Exception as exc:  # missing package, unsupported platform or Tk version
        log.info("Drag and drop is not available (%s); use the file chooser.", exc)
        return False


class SettingsDialog(tk.Toplevel):
    """Edit the summary provider settings saved on this computer."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(False, False)
        self.transient(parent)
        self.saved = False

        current = read_saved_settings()
        self.key = tk.StringVar(value=current.api_key)
        self.url = tk.StringVar(value=current.base_url)
        self.model = tk.StringVar(value=current.model)
        self.show_key = tk.BooleanVar(value=False)

        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="AI summary (optional)", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(body, text="API key").grid(row=1, column=0, sticky="w", pady=3)
        self.key_entry = ttk.Entry(body, textvariable=self.key, width=48, show="•")
        self.key_entry.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Checkbutton(body, text="Show key", variable=self.show_key, command=self._toggle_key).grid(
            row=2, column=1, sticky="w")

        ttk.Label(body, text="Base URL").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(body, textvariable=self.url, width=48).grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Label(body, text="Model").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(body, textvariable=self.model, width=48).grid(row=4, column=1, sticky="ew", pady=3)

        note = (
            "Works with any OpenAI-compatible provider. Leave the key empty to only transcribe.\n"
            f"Saved only on this computer: {settings_file()}\n"
            "Environment variables and a .env file in the current folder take precedence."
        )
        ttk.Label(body, text=note, foreground="gray", wraplength=460, justify="left").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 12))

        buttons = ttk.Frame(body)
        buttons.grid(row=6, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", command=self._save).pack(side="right", padx=(0, 8))

        self.bind("<Escape>", lambda _e: self.destroy())
        self.grab_set()
        self.key_entry.focus_set()

    def _toggle_key(self) -> None:
        self.key_entry.configure(show="" if self.show_key.get() else "•")

    def _save(self) -> None:
        settings = SummarySettings(
            api_key=self.key.get().strip(),
            base_url=self.url.get().strip() or SummarySettings().base_url,
            model=self.model.get().strip() or SummarySettings().model,
        )
        try:
            save_settings(settings)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Settings", f"Could not save the settings:\n{exc}", parent=self)
            return
        self.saved = True
        self.destroy()


class App:
    def __init__(self, root: tk.Tk, model: str, device: str):
        self.root = root
        self.transcriber = Transcriber(model=model, device=device)
        self.files: list[Path] = []
        self.states: list[str] = []
        self.last_folder: Optional[Path] = None
        self.current = ""
        self.events: queue.Queue = queue.Queue()
        self.busy = False

        root.title("Hebrew STT")
        root.geometry("640x520")
        root.minsize(520, 420)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)

        self.drop = tk.Label(
            frame, relief="groove", borderwidth=2, cursor="hand2", pady=26,
            font=("TkDefaultFont", 12),
        )
        self.drop.pack(fill="x")
        self.drop.bind("<Button-1>", lambda _e: self._browse())

        self.table = ttk.Treeview(frame, columns=("file", "status"), show="headings", height=8)
        self.table.heading("file", text="File")
        self.table.heading("status", text="Status")
        self.table.column("file", width=420)
        self.table.column("status", width=140, anchor="w")
        self.table.pack(fill="both", expand=True, pady=(12, 8))

        if _enable_drag_and_drop(root):
            from tkinterdnd2 import DND_FILES

            self.drop.configure(text="Drop audio or video files here\nor click to choose")
            for widget in (self.drop, self.table):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.drop.configure(text="Click to choose audio or video files")

        options = ttk.Frame(frame)
        options.pack(fill="x")
        self.summarize = tk.BooleanVar(value=False)
        self.summary_check = ttk.Checkbutton(options, variable=self.summarize)
        self.summary_check.pack(side="left")
        ttk.Button(options, text="Settings...", command=self._settings).pack(side="right")

        self.status = ttk.Label(frame, text="Add files to start.")
        self.status.pack(fill="x", pady=(10, 4))
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(12, 0))
        self.clear_button = ttk.Button(buttons, text="Clear", command=self._clear)
        self.clear_button.pack(side="left")
        ttk.Button(buttons, text="Open folder", command=self._open_output).pack(side="left", padx=8)
        self.start_button = ttk.Button(buttons, text="Start", command=self._start)
        self.start_button.pack(side="right")

        self._refresh_summary_option(initial=True)
        self.root.after(100, self._poll)

    # ------------------------------------------------------------ file list

    def _browse(self) -> None:
        if self.busy:
            return
        patterns = " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS))
        chosen = filedialog.askopenfilenames(
            parent=self.root, title="Choose audio or video files",
            filetypes=[("Audio and video", patterns), ("All files", "*.*")],
        )
        self._add(chosen)

    def _on_drop(self, event) -> str:
        if not self.busy:
            self._add(self.root.tk.splitlist(event.data))
        return event.action

    def _add(self, paths) -> None:
        added = 0
        for path in collect_media(paths):
            if path in self.files:
                continue
            self.files.append(path)
            self.states.append(WAITING)
            self.table.insert("", "end", iid=str(len(self.files) - 1), values=(path.name, WAITING))
            added += 1
        if paths and not added:
            self.status.configure(text="No new audio or video files in what you added.")
        elif added:
            self.status.configure(text=f"{len(self.files)} file(s) ready. Press Start.")

    def _set_state(self, index: int, state: str) -> None:
        self.states[index] = state
        self.table.set(str(index), "status", state)

    def _clear(self) -> None:
        if self.busy:
            return
        self.files.clear()
        self.states.clear()
        self.table.delete(*self.table.get_children())
        self.progress["value"] = 0
        self.status.configure(text="Add files to start.")

    def _open_output(self) -> None:
        selected = self.table.selection()
        if selected:
            folder = self.files[int(selected[0])].parent
        else:
            folder = self.last_folder or (self.files[0].parent if self.files else None)
        if folder and folder.is_dir():
            _open_folder(folder)

    # -------------------------------------------------------------- summary

    def _refresh_summary_option(self, initial: bool = False) -> None:
        settings = load_settings()
        if settings.enabled:
            host = urlparse(settings.base_url).netloc or settings.base_url
            self.summary_check.configure(
                state="normal", text=f"Summarize with AI ({settings.model}; sends the text to {host})")
            if initial:
                self.summarize.set(True)
        else:
            self.summarize.set(False)
            self.summary_check.configure(
                state="disabled", text="Summarize with AI (add an API key in Settings)")

    def _settings(self) -> None:
        dialog = SettingsDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.saved:
            was_enabled = str(self.summary_check.cget("state")) == "normal"
            self._refresh_summary_option()
            if not was_enabled and load_settings().enabled:
                self.summarize.set(True)

    # ----------------------------------------------------------- processing

    def _start(self) -> None:
        todo = [i for i, state in enumerate(self.states) if state != DONE]
        if self.busy or not todo:
            if not self.files:
                self.status.configure(text="Add files first: drop them above or click to choose.")
            return
        settings = load_settings() if self.summarize.get() else None
        self.busy = True
        self.start_button.state(["disabled"])
        self.clear_button.state(["disabled"])
        self.summary_check.state(["disabled"])
        threading.Thread(target=self._work, args=(todo, settings), daemon=True).start()

    def _work(self, todo: list[int], settings: Optional[SummarySettings]) -> None:
        put = self.events.put
        failed = 0
        for n, index in enumerate(todo, start=1):
            path = self.files[index]
            put(("state", index, WORKING))
            put(("status", f"[{n}/{len(todo)}] Transcribing {path.name}..."))
            try:
                result = process_file(
                    path, self.transcriber, settings,
                    progress=lambda f, cur, tot: put(("progress", f, cur, tot)),
                    on_status=lambda text: put(("status", text)),
                )
            except Exception as exc:
                log.exception("Failed: %s", path)
                failed += 1
                put(("state", index, FAILED))
                put(("status", f"{path.name}: {exc}"))
                continue
            self.last_folder = result.paths.text.parent
            if result.summary_error:
                failed += 1
                put(("state", index, "Done, no summary"))
                put(("status", f"Summary failed for {path.name}: {result.summary_error}"))
            else:
                put(("state", index, DONE))
        put(("finished", len(todo), failed))

    def _poll(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "state":
                    self._set_state(event[1], event[2])
                    if event[2] == WORKING:
                        self.current = self.files[event[1]].name
                        self.progress["value"] = 0
                elif kind == "status":
                    self.status.configure(text=event[1])
                elif kind == "progress":
                    _, fraction, current, total = event
                    self.progress["value"] = fraction * 100
                    self.status.configure(
                        text=f"Transcribing {self.current}... {fraction:.0%}  "
                             f"({format_timestamp(current)} / {format_timestamp(total)})")
                elif kind == "finished":
                    _, count, failed = event
                    self._finish(count, failed)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finish(self, count: int, failed: int) -> None:
        self.busy = False
        self.start_button.state(["!disabled"])
        self.clear_button.state(["!disabled"])
        self._refresh_summary_option()
        self.progress["value"] = 100
        if failed:
            self.status.configure(text=f"Finished with problems: {failed} of {count} file(s). See the Status column.")
        else:
            self.status.configure(text=f"Finished {count} file(s). Results are saved next to each file.")
        self.root.bell()

    def _on_close(self) -> None:
        if self.busy and not messagebox.askyesno(
            "Quit", "A file is still being transcribed. Quit anyway?", parent=self.root
        ):
            return
        self.root.destroy()


def run(model: str = DEFAULT_MODEL, device: str = "auto") -> int:
    if sys.platform == "win32":
        try:  # sharp text on high-DPI screens instead of a blurry, scaled window
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    App(root, model=model, device=device)
    root.mainloop()
    return 0
