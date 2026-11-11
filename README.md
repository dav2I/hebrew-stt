# hebrew-stt

Transcribe Hebrew audio and video on your own computer, with an optional AI summary.

![Dropping a Hebrew recording into hebrew-stt and reading the transcript](docs/demo.gif)

- **Local transcription** with Whisper fine-tuned for Hebrew by [ivrit-ai](https://huggingface.co/ivrit-ai) (`ivrit-ai/whisper-large-v3-turbo-ct2`), run by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- **Any computer**: uses an NVIDIA GPU when available, otherwise the CPU. Windows, macOS and Linux.
- **A simple page in your browser**: drop files, follow the progress (or stop a file), read, copy or download the result. A **command line** does the same thing.
- **Optional summary** through any OpenAI-compatible API (DeepSeek by default). Without an API key, nothing leaves your computer.

## Quick start

You need Python 3.10 or newer.

```bash
pipx install git+https://github.com/dav2I/hebrew-stt
```

```bash
hebrew-stt
```

That opens hebrew-stt in your browser. Drop audio or video files on the page and the transcript appears when it is ready. Keep the terminal window open while you use the page; press Ctrl+C there to stop.

Or use the command line:

```bash
hebrew-stt lecture.mp3 meeting.m4a
```

The first run downloads the model (about 1.6 GB) from Hugging Face. After that it works offline.

No pipx? `pip install git+https://github.com/dav2I/hebrew-stt` works too, or from a clone: `pip install -e .`

To try it on the included sample (a 40-second Hebrew reading, see [examples](examples/)), run this from a clone:

```bash
hebrew-stt examples/sample.mp3
```

### Faster with an NVIDIA GPU (optional)

```bash
pipx install "hebrew-stt[cuda] @ git+https://github.com/dav2I/hebrew-stt"
```

This adds the CUDA 12 libraries (cuBLAS, cuDNN 9) from pip. On Windows they are found automatically. On Linux, either install CUDA 12 and cuDNN 9 system-wide or add the pip packages to `LD_LIBRARY_PATH` ([details](https://github.com/SYSTRAN/faster-whisper#gpu)). If no usable GPU is found, hebrew-stt says so and uses the CPU, which is slower but gives the same result. On macOS it always uses the CPU.

## Usage

```
hebrew-stt [FILE_OR_FOLDER ...] [--no-summary] [-o DIR] [--device auto|cuda|cpu] [--model NAME]
```

| Option | What it does |
|---|---|
| *(no files)* | Opens the page in your browser |
| `FILE_OR_FOLDER` | Audio/video files, or folders containing them |
| `--no-summary` | Only transcribe, even if an API key is set |
| `-o, --output-dir` | Where to save results (see below) |
| `--device` | `auto` (default), `cuda` or `cpu` |
| `--model` | Another Whisper model in CTranslate2 format |

Each file produces:

| File | Content |
|---|---|
| `name.txt` | The transcript, one segment per line |
| `name_timestamps.txt` | The transcript with times: `[01:02 -> 01:05] ...` |
| `name_summary.md` | The AI summary, only when enabled |

From the command line they are saved next to each input file. From the browser they are shown on the page and also saved in `Documents/hebrew-stt` (the page has an **Open folder** button). `-o` changes either location.

Supported formats: mp3, m4a, wav, flac, ogg, opus, aac, wma, mp4, mkv, webm, mov, avi. No separate FFmpeg install is needed.

## AI summary (optional)

The summary first detects what the recording is (a lecture, a meeting, a podcast or interview, a tutorial, a personal note) and then writes a Hebrew summary in a matching structure: key concepts for a lecture, decisions and action items for a meeting, and so on. Long recordings are summarized in parts and then merged.

To turn it on, click **Settings** on the page and enter an API key. Or set these in your environment or in a `.env` file in the folder you run from (see [`.env.example`](.env.example)):

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

On macOS and Linux the file is readable only by your user. The page never sends the saved key back to the browser.

## Privacy

- **Your audio never leaves your computer.** Transcription runs locally.
- The page is served only to this computer (`127.0.0.1`), and every request needs the random key that is in the address hebrew-stt opens. Files you drop are copied to a temporary folder for transcription and deleted afterwards.
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
- The sample recording is by Ariel Palmon, from Wikimedia Commons, under CC BY 3.0 ([details](examples/README.md))

## License

MIT. See [LICENSE](LICENSE). The sample recording in `examples/` keeps its own CC BY 3.0 license.
