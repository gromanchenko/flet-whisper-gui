# Huisper - Voice Transcription App

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Privacy](https://img.shields.io/badge/privacy-100%25%20offline-purple)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Win%20%7C%20Linux-lightgrey)

## TL;DR
**It** is a local-first, privacy-focused voice transcription app designed to run offline.
> **Note**: Internet access is required **only** for the initial model download or when switching to a new model size. Once downloaded, the app works 100% offline.
*   **Install**: `pip install -r requirements.txt` (requires Python 3.10+)
*   **Run (GUI)**: `python app.py`
*   **Run (CLI)**: `python app.py --mode live` (mic to stdout) or `python app.py --mode batch --source-audio-file input.wav`

## Dependencies
```
Huisper
├── flet (UI Framework)
├── faster-whisper (Inference Engine)
│   ├── ctranslate2
│   └── tokenizers
├── sounddevice (Audio Capture)
│   └── PortAudio
├── webrtcvad (Voice Activity Detection)
├── resemblyzer (Speaker Verification)
│   └── torch
└── toml (Configuration)
```

## Overview
A cross-platform, thread-safe GUI for faster-whisper. Implements real-time VAD, speaker embeddings (Resemblyzer), and hardware acceleration in a clean Python codebase.

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application (GUI)
python app.py
```

## Architecture
Transcriber uses a **Dual-Threaded Producer-Consumer** architecture to ensure zero-latency recording:
1.  **VAD Thread (Producer)**: Captures audio and detects voice activity in real-time (30ms frames).
2.  **Inference Thread (Consumer)**: Verifies speaker identity and runs Whisper transcription on buffered speech.


## Operation Modes

### 1. GUI Mode (Default)
Standard graphical interface for desktop use.
```bash
python app.py
# or explicitly:
python app.py --gui
```

### 2. Live Mode (Headless / Persistent)
Listens to the microphone indefinitely and streams text to `stdout`.
Status messages and logs are sent to `stderr`, making `stdout` clean for piping.
```bash
python app.py --mode live
# Output example:
# (stderr) Status: Listening...
# (stdout) Hello world
```

### 3. Batch Mode (Headless / Discrete)
Process a single audio file and exit. Useful for scripting.
```bash
python app.py --mode batch --source-audio-file /path/to/meeting.wav
# Optional: Specify output file
python app.py --mode batch --source-audio-file in.wav --dst-text-file out.txt
```

## CLI Arguments

| Section | Argument | Overrides | Description |
|---|---|---|---|
| **[Execution]** | `--mode` | `mode` | Execution mode: `gui`, `live`, or `batch`. |
| | `--source-audio-file` | `source_audio_file` | Input file (Required for `batch` mode). |
| | `--dst-text-file` | `dst_text_file` | Output file (Optional for `batch` mode). |
| **[whisper]** | `--model-size` | `model_size` | Whisper model size (tiny, base, small, medium, large, large-v2, large-v3). |
| | `--model-path` | `model_path` | Path to local model folder. |
| | `--language` | `language` | Language code ("auto" or ISO code like "en", "fr"). |
| | `--initial-prompt` | `initial_prompt` | Context prompt for the model. |
| **[vad]** | `--min-silence-duration-ms` | `min_silence_duration_ms` | Silence threshold (ms) to cut a segment. |
| | `--max-speech-duration-s` | `max_speech_duration_s` | Max speech duration (s) before forcing a cut. |
| | `--min-speech-duration-ms` | `min_speech_duration_ms` | Min duration (ms) to consider as speech. |
| **[storage]** | `--transcript-dir` | `transcript_dir` | Directory to save JSONL logs. |
| | `--audio-dir` | `audio_dir` | Directory to save audio recordings. |
| | `--audio-format` | `audio_format` | Format (wav, mp3). |
| | `--ffmpeg-path` | `ffmpeg_path` | Path to ffmpeg executable (for MP3 conversion). |
| **[speaker]** | `--strict-mode` | `strict_mode` | Enforce speaker verification (true/false). |
| | `--min-enroll-seconds` | `min_enroll_seconds` | Min seconds required for enrollment. |
| | `--similarity-threshold` | `similarity_threshold` | Cosine similarity threshold (0.0-1.0). |
| | `--ui-only-my-voice` | `ui_only_my_voice` | Only listen to my voice (true/false). |

## Cross-Platform Support (Windows, Linux, macOS)
Designed to run on all major operating systems.
- **Windows**: Supports NVIDIA GPU acceleration (CUDA) automatically.
- **Linux**: Supports NVIDIA GPU acceleration (CUDA).
- **macOS**: Optimized for Apple Silicon (M-series).

### Building Executables
You can build standalone executables using GitHub Actions (included in `.github/workflows/build.yml`) or locally:
```bash
# Install packaging tools
pip install flet

# Build (Windows)
flet pack app.py --icon assets/icon.ico --name Huisper

# Build (Linux/macOS)
flet pack app.py --icon assets/icon.png --name Huisper
```

## Configuration (`config.toml`)

| Section | Key | Default | Description |
## Configuration Reference

You can configure Huisper via the `config.toml` file or CLI arguments. CLI arguments override config file values.

| Category | Variable / Config Key | CLI Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Whisper** | `model_size` | `--model-size` | `str` | `"small"` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| | `model_path` | `--model-path` | `str` | `""` | Path to a local CTranslate2 model directory (overrides size). |
| | `language` | `--language` | `str` | `"auto"` | ISO language code (e.g., `en`, `es`) or `auto`. |
| | `initial_prompt` | `--initial-prompt` | `str` | `""` | Context hint to improve accuracy (e.g. spelling of names). |
| **VAD** | `min_silence_duration_ms` | `--min-silence-duration-ms` | `int` | `500` | Silence duration required to split segments. |
| | `max_speech_duration_s` | `--max-speech-duration-s` | `int` | `10` | Force split segment after N seconds. |
| | `min_speech_duration_ms` | `--min-speech-duration-ms` | `int` | `250` | Ignore speech segments shorter than this. |
| | `speech_pad_ms` | `--speech-pad-ms` | `int` | `30` | Milliseconds of audio padding around speech. |
| **Storage** | `transcript_dir` | `--transcript-dir` | `str` | `./Transcripts` | Directory to save JSONL transcriptions. |
| | `audio_dir` | `--audio-dir` | `str` | `./Audio` | Directory to save recording audio files. |
| | `audio_format` | `--audio-format` | `str` | `"wav"` | Output format: `"wav"` (default) or `"mp3"`. |
| | `ffmpeg_path` | `--ffmpeg-path` | `str` | `"ffmpeg"` | Path to ffmpeg executable (required for mp3). |
| **Speaker** | `strict_mode` | `--strict-mode` | `bool` | `true` | Reject audio if verification confidence is low. |
| | `min_enroll_seconds` | `--min-enroll-seconds` | `int` | `60` | Minimum audio duration required to enroll voice. |
| | `similarity_threshold` | `--similarity-threshold` | `float` | `0.75` | Cosine similarity threshold (0.0-1.0) for match. |
| | `ui_only_my_voice` | `--ui-only-my-voice` | `bool` | `false` | Initial state of "Only Listen to Me" toggle. |
| **Modes** | `mode` | `--mode` | `str` | `"live"` | Execution mode: `gui`, `live` (headless mic), `batch`. |
| | `source_audio_file` | `--source-audio-file` | `str` | `None` | Input file path for batch mode. |

### Configuration (`config.toml`)
The `config.toml` file is located in the app root. It is automatically created if it does not exist.
```toml
[whisper]
model_size = "small"
# ... (see table above)
```

## Optimized Configuration Example
Here is a configuration tuned for a **technical, single-user English environment** on macOS Silicon.

### `config.toml`
```toml
[whisper]
# "medium" balances high precision with acceptable speed on M1/M2/M3 chips.
model_size = "medium"
# Hardcoding "en" skips language detection (saving ~30s load time).
language = "en"
# Prompts the model with context to ensure proper capitalization and punctuation for technical terms.
initial_prompt = "Hello. This is a technical transcription covering software engineering, Python development, and system administration. It uses proper capitalization and punctuation."

[vad]
# Slightly aggressive silence detection for cleaner segmentation.
min_silence_duration_ms = 500

[speaker]
# "Strict mode" ensures high confidence before transcribing.
strict_mode = true
# Higher threshold for single-user security/precision.
similarity_threshold = 0.75
```

### Explanation of Settings
*   **`model_size = "medium"`**: The sweet spot for Apple Silicon. Significant accuracy jump over `small` without the massive latency of `large`.
*   **`language = "en"`**: forcing English prevents the model from hallucinating other languages during silence and skips the initialization delay of the detector.
*   **`initial_prompt`**: This is critical for "steering" the model's style. 
    *   Starts with "Hello." to seed sentence-casing.
    *   Lists keywords ("software engineering", "Python") so the model recognizes jargon.
    *   Explicitly requests proper punctuation.

## Custom Model Path

The `model_path` setting allows using a locally-stored CTranslate2-converted Whisper model.

### Default Cache Location
When using a model size name (e.g., `medium`), models are downloaded to:
```
~/.cache/huggingface/hub/models--Systran--faster-whisper-{size}/snapshots/{hash}/
```

### Required Files
A valid model directory must contain:
```
model.bin        # CTranslate2 model weights
config.json      # Model configuration
tokenizer.json   # Tokenizer data
vocabulary.txt   # Token vocabulary
```

### Using a Custom Path
1. Download or convert a model to CTranslate2 format
2. Set `model_path` to the directory containing the above files
3. The `model_size` setting is ignored when `model_path` is set

Example:
```toml
[whisper]
model_path = "/path/to/my-custom-whisper-model/"
```

## Validation
To validate:
```bash
pytest tests/
```
