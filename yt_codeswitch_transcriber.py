#!/usr/bin/env python3
"""
yt_codeswitch_transcriber.py

YouTube URL -> audio -> Whisper transcription (word-level timestamps)
-> every word labelled Persian ("fa") or English ("en") -> SQLite database.

Install (once):
    pip install yt-dlp faster-whisper
    # ffmpeg must also be installed on your system (e.g. `brew install ffmpeg`,
    # `sudo apt install ffmpeg`, or the Windows build from ffmpeg.org)

Usage examples:
    python yt_codeswitch_transcriber.py "https://www.youtube.com/watch?v=XXXX"
    python yt_codeswitch_transcriber.py URL --model medium --db corpus.db --export-csv out/
    python yt_codeswitch_transcriber.py URL --multilingual          # per-segment language detection
    python yt_codeswitch_transcriber.py URL --loanwords loanwords.txt

How the Persian/English split works
-----------------------------------
Whisper writes Persian words in Persian script and (usually) English words in
Latin script. So each word is classified by the Unicode script of its letters:
    Arabic-script letters only -> "fa"
    Latin letters only         -> "en"
    digits / punctuation only  -> "other"
    both scripts in one token  -> "mixed"
Known limitation: Whisper sometimes transliterates English words into Persian
script (e.g. "استارتاپ" for "startup"). Those look Persian to a script check.
Pass --loanwords with a text file (one Persian-script word per line) to relabel
such words as English; they are stored with translit=1 so you can review them.
"""
from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import string
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# Punctuation to strip from word edges (ASCII + common Persian/typographic marks)
STRIP_CHARS = string.punctuation + "،؛؟«»…–—“”‘’٪٬٫"

# A mixed-language hint: nudges Whisper to keep English words in Latin script.
DEFAULT_PROMPT = (
    "این یک پادکست درباره digital innovation و startup است، "
    "با کلمات انگلیسی مثل product، team و feedback."
)


# --------------------------------------------------------------------------
# 1. Download audio
# --------------------------------------------------------------------------
def download_audio(url: str, out_dir: Path) -> tuple[Path, dict]:
    """Download the audio track of a YouTube video as 16 kHz mono WAV."""
    import yt_dlp  # imported here so the rest of the file works without it

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    audio_path = out_dir / f"{info['id']}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Expected audio file not found: {audio_path}")
    return audio_path, info


# --------------------------------------------------------------------------
# 2. Transcribe
# --------------------------------------------------------------------------
def transcribe(
    audio_path: Path,
    model_name: str = "large-v3",
    language: str | None = "fa",
    prompt: str | None = DEFAULT_PROMPT,
    multilingual: bool = False,
    device: str = "auto",
    compute_type: str = "auto",
) -> list[dict]:
    """Return a list of segments: {start, end, text, words: [{word,start,end,prob}]}."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    kwargs = dict(
        language=language,
        initial_prompt=prompt,
        word_timestamps=True,
        vad_filter=True,   # skip silence/music: fewer hallucinations
        beam_size=5,
        # condition_on_previous_text stays at its default (True) on purpose:
        # in faster-whisper, turning it off also drops the initial prompt
        # after the first 30-second window.
    )
    if multilingual:
        # Needs faster-whisper >= 1.1.0: detects the language per segment,
        # which can help when speakers switch languages mid-conversation.
        kwargs["language"] = None
        kwargs["multilingual"] = True

    segments_iter, info = model.transcribe(str(audio_path), **kwargs)
    print(f"Transcribing... (language setting: {kwargs['language'] or 'auto'}, "
          f"detected: {info.language} p={info.language_probability:.2f})")

    segments = []
    for seg in segments_iter:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                for w in (seg.words or [])
            ],
        })
        print(f"  [{seg.start:7.1f}s] {seg.text.strip()[:80]}")
    return segments


# --------------------------------------------------------------------------
# 3. Classify words by script
# --------------------------------------------------------------------------
def _script_of(ch: str) -> str | None:
    """'persian' for Arabic-script letters, 'latin' for Latin letters, else None."""
    if not ch.isalpha():          # digits, punctuation, ZWNJ, diacritics -> ignored
        return None
    name = unicodedata.name(ch, "")
    if name.startswith("ARABIC"):
        return "persian"
    if name.startswith("LATIN"):
        return "latin"
    return "other"


def normalize_word(raw: str) -> str:
    """Strip edge punctuation; unify Arabic Yeh/Kaf to Persian forms; lowercase Latin."""
    w = raw.strip().strip(STRIP_CHARS)
    w = w.replace("ي", "ی").replace("ك", "ک")
    return w.lower() if any(_script_of(c) == "latin" for c in w) else w


def classify_word(raw: str, loanwords: set[str] | None = None) -> tuple[str, str, int]:
    """Return (normalized_word, lang, translit_flag). lang in fa/en/mixed/other."""
    norm = normalize_word(raw)
    scripts = {_script_of(c) for c in norm} - {None}
    has_fa, has_en = "persian" in scripts, "latin" in scripts

    if has_fa and has_en:
        return norm, "mixed", 0
    if has_en:
        return norm, "en", 0
    if has_fa:
        if loanwords and norm in loanwords:
            return norm, "en", 1          # Persian-script spelling of an English word
        return norm, "fa", 0
    return norm, "other", 0


def load_loanwords(path: Path | None) -> set[str]:
    if not path:
        return set()
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(normalize_word(line))
    return words


# --------------------------------------------------------------------------
# 4. Database
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id       TEXT PRIMARY KEY,
    url            TEXT,
    title          TEXT,
    channel        TEXT,
    duration_s     REAL,
    model          TEXT,
    transcribed_at TEXT
);
CREATE TABLE IF NOT EXISTS segments (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    t_start  REAL,
    t_end    REAL,
    text     TEXT
);
CREATE TABLE IF NOT EXISTS words (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    TEXT NOT NULL REFERENCES videos(video_id),
    segment_id  INTEGER REFERENCES segments(id),
    position    INTEGER,          -- running word index within the video
    word        TEXT,             -- as Whisper wrote it
    norm        TEXT,             -- cleaned/normalised form
    lang        TEXT,             -- fa / en / mixed / other
    translit    INTEGER DEFAULT 0,-- 1 = Persian-script spelling relabelled as English
    t_start     REAL,
    t_end       REAL,
    probability REAL
);
CREATE INDEX IF NOT EXISTS idx_words_video_lang ON words(video_id, lang);

-- Convenience views: the "separate databases" you asked for, as separate tables.
CREATE VIEW IF NOT EXISTS persian_words AS SELECT * FROM words WHERE lang = 'fa';
CREATE VIEW IF NOT EXISTS english_words AS SELECT * FROM words WHERE lang = 'en';
"""


def save_to_db(db_path: Path, url: str, info: dict, model_name: str,
               segments: list[dict], loanwords: set[str]) -> dict:
    """Store everything; re-running on the same video replaces its old rows."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    vid = info["id"]

    with conn:  # one transaction
        for table in ("words", "segments", "videos"):
            conn.execute(f"DELETE FROM {table} WHERE video_id = ?", (vid,))
        conn.execute(
            "INSERT INTO videos VALUES (?,?,?,?,?,?,?)",
            (vid, url, info.get("title"), info.get("channel") or info.get("uploader"),
             info.get("duration"), model_name,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        position = 0
        for seg in segments:
            cur = conn.execute(
                "INSERT INTO segments (video_id, t_start, t_end, text) VALUES (?,?,?,?)",
                (vid, seg["start"], seg["end"], seg["text"]),
            )
            seg_id = cur.lastrowid
            for w in seg["words"]:
                norm, lang, translit = classify_word(w["word"], loanwords)
                if not norm:          # token was pure punctuation
                    continue
                conn.execute(
                    "INSERT INTO words (video_id, segment_id, position, word, norm, lang,"
                    " translit, t_start, t_end, probability) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (vid, seg_id, position, w["word"].strip(), norm, lang, translit,
                     w["start"], w["end"], w["prob"]),
                )
                position += 1

    stats = summarize(conn, vid)
    conn.close()
    return stats


def summarize(conn: sqlite3.Connection, video_id: str) -> dict:
    """Word counts per language and the number of fa<->en switches."""
    counts = dict(conn.execute(
        "SELECT lang, COUNT(*) FROM words WHERE video_id=? GROUP BY lang", (video_id,)))
    langs = [r[0] for r in conn.execute(
        "SELECT lang FROM words WHERE video_id=? AND lang IN ('fa','en') ORDER BY position",
        (video_id,))]
    switches = sum(1 for a, b in zip(langs, langs[1:]) if a != b)
    return {"counts": counts, "switches": switches}


def export_csv(db_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    for view in ("persian_words", "english_words"):
        cur = conn.execute(f"SELECT * FROM {view}")
        with open(out_dir / f"{view}.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([d[0] for d in cur.description])
            writer.writerows(cur)
    conn.close()
    print(f"CSV files written to {out_dir}/")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Transcribe a YouTube video and split Persian/English words.")
    p.add_argument("url", help="YouTube video URL")
    p.add_argument("--db", type=Path, default=Path("corpus.db"), help="SQLite file (default: corpus.db)")
    p.add_argument("--audio-dir", type=Path, default=Path("audio"), help="where audio is saved")
    p.add_argument("--model", default="large-v3",
                   help="Whisper model: tiny/base/small/medium/large-v3 (use small/medium on CPU)")
    p.add_argument("--language", default="fa", help="forced language code, or 'auto' (default: fa)")
    p.add_argument("--multilingual", action="store_true", help="per-segment language detection")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="initial prompt (mixed fa/en hint)")
    p.add_argument("--loanwords", type=Path, help="text file of Persian-script words to count as English")
    p.add_argument("--device", default="auto", help="auto / cpu / cuda")
    p.add_argument("--compute-type", default="auto", help="auto / int8 / float16 ...")
    p.add_argument("--keep-audio", action="store_true", help="do not delete the WAV afterwards")
    p.add_argument("--export-csv", type=Path, help="folder for persian_words.csv / english_words.csv")
    args = p.parse_args(argv)

    print("Downloading audio...")
    audio_path, info = download_audio(args.url, args.audio_dir)
    print(f"Got: {info.get('title')}  ({info.get('duration')} s)")

    segments = transcribe(
        audio_path,
        model_name=args.model,
        language=None if args.language == "auto" else args.language,
        prompt=args.prompt or None,
        multilingual=args.multilingual,
        device=args.device,
        compute_type=args.compute_type,
    )

    stats = save_to_db(args.db, args.url, info, args.model, segments, load_loanwords(args.loanwords))
    print("\nDone.")
    print(f"  Word counts by language: {stats['counts']}")
    print(f"  Persian<->English switches: {stats['switches']}")
    print(f"  Database: {args.db}  (tables: videos, segments, words; views: persian_words, english_words)")

    if args.export_csv:
        export_csv(args.db, args.export_csv)
    if not args.keep_audio:
        audio_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
