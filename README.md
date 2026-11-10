# hebrew-stt

Transcribe Hebrew audio and video on your own computer, with an optional AI summary.

- **Local transcription** with Whisper fine-tuned for Hebrew by [ivrit-ai](https://huggingface.co/ivrit-ai) (`ivrit-ai/whisper-large-v3-turbo-ct2`), run by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- **Any computer**: uses an NVIDIA GPU when available, otherwise the CPU. Windows, macOS and Linux.
- **A simple window** with drag and drop, and a **command line** that does the same thing.
- **Optional summary** through any OpenAI-compatible API (DeepSeek by default). Without an API key, nothing leaves your computer.

## Quick start

You need Python 3.10 or newer.

```bash
pipx install git+https://github.com/dav2I/hebrew-stt
```

```bash
hebrew-stt
```

That opens the window: drop files (or click to choose), then press **Start**. Or use the command line:

```bash
hebrew-stt lecture.mp3 meeting.m4a
```

The first run downloads the model (about 1.6 GB) from Hugging Face. After that it works offline.

No pipx? `pip install git+https://github.com/dav2I/hebrew-stt` works too, or from a clone: `pip install -e .`

### Faster with an NVIDIA GPU (optional)

```bash
pipx install "hebrew-stt[cuda] @ git+https://github.com/dav2I/hebrew-stt"
```

This adds the CUDA 12 libraries (cuBLAS, cuDNN 9) from pip. On Windows they are found automatically. On Linux, either install CUDA 12 and cuDNN 9 system-wide or add the pip packages to `LD_LIBRARY_PATH` ([details](https://github.com/SYSTRAN/faster-whisper#gpu)). If no usable GPU is found, hebrew-stt says so and uses the CPU, which is slower but gives the same result. On macOS it always uses the CPU.

### Linux and macOS: the window needs Tk

The command line works everywhere. For the window you may need Tk: `sudo apt install python3-tk` on Debian/Ubuntu, or `brew install python-tk` for Homebrew's Python.

## Usage

```
hebrew-stt [FILE_OR_FOLDER ...] [--no-summary] [-o DIR] [--device auto|cuda|cpu] [--model NAME]
```

| Option | What it does |
|---|---|
| *(no files)* | Opens the window |
| `FILE_OR_FOLDER` | Audio/video files, or folders containing them |
| `--no-summary` | Only transcribe, even if an API key is set |
| `-o, --output-dir` | Save results here instead of next to each file |
| `--device` | `auto` (default), `cuda` or `cpu` |
| `--model` | Another Whisper model in CTranslate2 format |

Results are saved next to each input file:

| File | Content |
|---|---|
| `name.txt` | The transcript, one segment per line |
| `name_timestamps.txt` | The transcript with times: `[01:02 -> 01:05] ...` |
| `name_summary.md` | The AI summary, only when enabled |

Supported formats: mp3, m4a, wav, flac, ogg, opus, aac, wma, mp4, mkv, webm, mov, avi. No separate FFmpeg install is needed.

## AI summary (optional)

The summary first detects what the recording is (a lecture, a meeting, a podcast or interview, a tutorial, a personal note) and then writes a Hebrew summary in a matching structure: key concepts for a lecture, decisions and action items for a meeting, and so on. Long recordings are summarized in parts and then merged.

To turn it on, open **Settings...** in the window and enter an API key. Or set these in your environment or in a `.env` file in the folder you run from (see [`.env.example`](.env.example)):

```bash
SUMMARY_API_KEY=your-key
SUMMARY_BASE_URL=https://api.deepseek.com
SUMMARY_MODEL=deepseek-flash
```

Any OpenAI-compatible chat API works. Set the base URL and model of your provider, for example `https://api.openai.com/v1`, or a local server such as Ollama at `http://localhost:11434/v1` (any non-empty key).

Where settings come from, highest priority first: environment variables, then `.env` in the current folder, then what you saved in **Settings**. The Settings file is kept only on your computer:

- Windows: `%APPDATA%\hebrew-stt\settings.env`
- macOS: `~/Library/Application Support/hebrew-stt/settings.env`
- Linux: `~/.config/hebrew-stt/settings.env`

On macOS and Linux the file is readable only by your user.

## Privacy

- **Your audio never leaves your computer.** Transcription runs locally.
- The only network access is:
  1. downloading the model from Hugging Face on the first run, and
  2. if you turn on the summary, sending the **transcript text** (not the audio) to the provider you configured. Their privacy policy applies to that text.
- Your API key is stored only on your computer, in the Settings file or your own `.env`.

## Development

```bash
git clone https://github.com/dav2I/hebrew-stt
cd hebrew-stt
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then:

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The tests need no model download and no network.

## Credits

- [ivrit-ai](https://huggingface.co/ivrit-ai) for the Hebrew Whisper models
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [tkinterdnd2](https://github.com/Eliav2/tkinterdnd2) for drag and drop

## License

MIT. See [LICENSE](LICENSE).
