# Persian-English Transcriber

A Python command-line tool that takes a YouTube URL, transcribes the audio with Whisper, and labels every word as **Persian** or **English**. Results are stored in a SQLite database, with optional CSV export.

I built it for research on code-switching in Persian-language podcasts, where speakers mix Persian and English within a single sentence. General-purpose transcription tools handle this poorly, and this tool is my attempt at a workable pipeline for studying it.

## What it does

1. Downloads the audio of a YouTube video (`yt-dlp`).
2. Transcribes it with word-level timestamps (`faster-whisper`).
3. Classifies each word by script: Persian (Arabic script), English (Latin script), mixed, or other.
4. Saves videos, segments, and words to SQLite, with ready-made views `persian_words` and `english_words`.
5. Prints word counts per language and the number of Persian/English switches.

## Requirements

- Python 3.10 or newer
- [ffmpeg](https://ffmpeg.org/)
- [deno](https://deno.com/) (YouTube downloads now need a JavaScript runtime)

```
pip install -U "yt-dlp[default]" faster-whisper
```

## Usage

```
python yt_codeswitch_transcriber.py "https://www.youtube.com/watch?v=VIDEO_ID" --model small --export-csv out
```

Useful options:

| Option | Meaning |
|---|---|
| `--model` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--language` | Forced language code (default `fa`), or `auto` |
| `--multilingual` | Per-segment language detection (needs faster-whisper 1.1.0+) |
| `--loanwords FILE` | Text file of Persian-script spellings of English words to relabel as English |
| `--db FILE` | SQLite file (default `corpus.db`) |
| `--export-csv DIR` | Write `persian_words.csv` and `english_words.csv` |

On a computer without a GPU, start with a short video and a small model.

## Database layout

- `videos`: one row per video (title, channel, duration, model used)
- `segments`: timestamped transcript segments
- `words`: every word with its language label, timestamps, and confidence
- views `persian_words` and `english_words`

## Known limitations

- Whisper handles mixed-language speech imperfectly, and English words are sometimes spelled in Persian script. The `--loanwords` option helps with known cases.
- Classification is by script only, so Persian written in Latin letters ("khoob") is labelled English.
- Results are saved only after transcription finishes, so an interrupted run saves nothing.

## Planned improvements

- Save segments incrementally while transcribing
- Options to transcribe only part of a video
- Semantic search over the stored transcripts
- A simple web interface and visualizations of code-switching patterns
