"""Analysis pipeline: audio -> transcript (word timestamps) -> AI highlights -> speaker framing."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import httpx
from pydantic import BaseModel, Field

MOCK = os.getenv("MOCK_AI", "0") == "1"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CHUNK_SECONDS = 20 * 60  # Whisper accepts files up to 25 MB; 20 min of 32 kbps mono is ~5 MB


# ---------------------------------------------------------------- media helpers

def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), {})
    return {"duration": float(info["format"]["duration"]), "width": v.get("width", 0), "height": v.get("height", 0),
            "has_audio": any(s.get("codec_type") == "audio" for s in info["streams"])}


def extract_audio(src: Path, dst_dir: Path, duration: float) -> list[tuple[Path, float]]:
    """Mono 16 kHz MP3 chunks with their start offsets."""
    chunks = []
    for i in range(max(1, math.ceil(duration / CHUNK_SECONDS))):
        start = i * CHUNK_SECONDS
        out = dst_dir / f"audio_{i}.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(CHUNK_SECONDS), "-i", str(src),
                        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(out)], check=True)
        chunks.append((out, start))
    return chunks


# ---------------------------------------------------------------- transcription

def transcribe(chunks: list[tuple[Path, float]], duration: float) -> list[dict]:
    """Returns [{"w": word, "s": start, "e": end}] across the whole video."""
    if MOCK:
        return _mock_words(duration)
    provider = os.getenv("TRANSCRIBE_PROVIDER", "openai")
    words: list[dict] = []
    for path, offset in chunks:
        part = _deepgram(path) if provider == "deepgram" else _whisper(path)
        words += [{"w": w["w"], "s": round(w["s"] + offset, 2), "e": round(w["e"] + offset, 2)} for w in part]
    return words


def _whisper(path: Path) -> list[dict]:
    with path.open("rb") as f:
        r = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            data={"model": "whisper-1", "response_format": "verbose_json", "timestamp_granularities[]": "word"},
            files={"file": (path.name, f, "audio/mpeg")}, timeout=600)
    r.raise_for_status()
    return [{"w": w["word"].strip(), "s": w["start"], "e": w["end"]} for w in r.json().get("words", [])]


def _deepgram(path: Path) -> list[dict]:
    r = httpx.post(
        "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true&detect_language=true",
        headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": "audio/mpeg"},
        content=path.read_bytes(), timeout=600)
    r.raise_for_status()
    alt = r.json()["results"]["channels"][0]["alternatives"][0]
    return [{"w": w.get("punctuated_word", w["word"]), "s": w["start"], "e": w["end"]} for w in alt["words"]]


def _mock_words(duration: float) -> list[dict]:
    text = ("ini contoh transkrip palsu untuk menguji aliran kerja tanpa kunci API sebenar "
            "setiap perkataan diberi masa supaya caption dan pemotongan boleh diuji").split()
    words, t, i = [], 0.3, 0
    while t < duration - 0.4:
        words.append({"w": text[i % len(text)], "s": round(t, 2), "e": round(t + 0.32, 2)})
        t += 0.4 if (i + 1) % 9 else 1.0
        i += 1
    return words


# ---------------------------------------------------------------- highlights

class Highlight(BaseModel):
    start: float = Field(description="Clip start in seconds, taken from a line timestamp")
    end: float = Field(description="Clip end in seconds")
    title: str = Field(description="Short catchy title in the video's language")
    hook: str = Field(description="The opening line that grabs attention")
    score: int = Field(description="Viral potential 0-100")
    reason: str = Field(description="One sentence on why this moment works, in Malay")


class Highlights(BaseModel):
    clips: list[Highlight]


def transcript_lines(words: list[dict], max_len: float = 8.0) -> list[dict]:
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if w["e"] - cur[0]["s"] >= max_len or w["w"].endswith((".", "?", "!")):
            lines.append({"s": cur[0]["s"], "e": cur[-1]["e"], "text": " ".join(x["w"] for x in cur)})
            cur = []
    if cur:
        lines.append({"s": cur[0]["s"], "e": cur[-1]["e"], "text": " ".join(x["w"] for x in cur)})
    return lines


def find_highlights(words: list[dict], duration: float, min_len=15, max_len=60, count=8) -> list[dict]:
    if MOCK or not words:
        n = max(1, min(count, int(duration // (min_len + 5))))
        step = duration / n
        return [{"start": round(i * step, 2), "end": round(min(duration, i * step + min(max_len, step - 1, 20)), 2),
                 "title": f"Klip contoh {i + 1}", "hook": "", "score": 90 - i * 7,
                 "reason": "Mod contoh: klip dipilih sama rata tanpa AI."} for i in range(n)]

    import anthropic

    lines = "\n".join(f"[{l['s']:.1f}-{l['e']:.1f}] {l['text']}" for l in transcript_lines(words))
    prompt = f"""You are an expert short-form video editor (TikTok, Reels, Shorts).
Below is a timestamped transcript of a {duration / 60:.0f}-minute video, often a podcast with a host and a guest.

Pick up to {count} self-contained moments that would perform well as vertical short clips.
Rules:
- Each clip is {min_len}-{max_len} seconds and starts and ends on line timestamps below.
- It must make sense without the rest of the video: a clear hook in the first 3 seconds, a complete thought, a satisfying ending.
- Prefer strong opinions, surprising facts, emotional stories, humour, practical tips, and quotable lines.
- Clips must not overlap. Score honestly; most moments are not 90+.
- Write title and hook in the transcript's language.

Transcript:
{lines}"""
    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Highlights,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError("AI tidak memulangkan senarai klip.")
    out = []
    for h in resp.parsed_output.clips:
        s, e = snap(words, h.start, h.end)
        if e - s >= 3:
            out.append({**h.model_dump(), "start": s, "end": e})
    return sorted(out, key=lambda c: -c["score"])


def snap(words: list[dict], s: float, e: float) -> tuple[float, float]:
    """Snap to word boundaries so clips never cut a word in half."""
    starts = [w["s"] for w in words if w["s"] >= s - 0.5]
    ends = [w["e"] for w in words if w["e"] <= e + 0.5]
    s2 = max(0.0, (min(starts) if starts else s) - 0.15)
    e2 = (max(ends) if ends else e) + 0.25
    return round(s2, 2), round(max(e2, s2 + 1), 2)


# ---------------------------------------------------------------- framing (who to follow)

_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def frame_for_clip(src: Path, start: float, end: float, width: int) -> dict:
    """Pick a horizontal crop centre for the clip.

    Detects faces a few times per second, groups them by horizontal position (one group per
    person in a fixed podcast shot) and follows the person whose mouth area moves the most,
    which is usually the one talking. Falls back to the centre when no face is found.
    """
    cap = cv2.VideoCapture(str(src))
    groups: list[dict] = []  # {"x": mean centre x, "n": hits, "motion": total, "y": mean centre y, "h": mean face h}
    t = start
    step = max(0.5, (end - start) / 40)
    while t < end:
        a, b = _grab(cap, t), _grab(cap, t + 0.2)
        t += step
        if a is None or b is None:
            continue
        gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        scale = 640 / gray.shape[1]
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        faces = _face.detectMultiScale(small, 1.1, 5, minSize=(30, 30))
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in faces:
            x, y, w, h = (int(v / scale) for v in (x, y, w, h))
            mouth = (slice(y + h * 2 // 3, y + h), slice(x + w // 4, x + w * 3 // 4))
            motion = float(cv2.absdiff(gray[mouth], gb[mouth]).mean()) if gray[mouth].size else 0.0
            cx = x + w / 2
            g = next((g for g in groups if abs(g["x"] - cx) < width * 0.15), None)
            if g is None:
                g = {"x": cx, "n": 0, "motion": 0.0, "y": y + h / 2, "h": h}
                groups.append(g)
            g["x"] = (g["x"] * g["n"] + cx) / (g["n"] + 1)
            g["y"] = (g["y"] * g["n"] + y + h / 2) / (g["n"] + 1)
            g["h"] = (g["h"] * g["n"] + h) / (g["n"] + 1)
            g["n"] += 1
            g["motion"] += motion
    cap.release()
    groups = [g for g in groups if g["n"] >= 2]
    if not groups:
        return {"cx": 0.5, "cy": 0.5, "z": 1.0, "faces": 0}
    best = max(groups, key=lambda g: g["motion"] / g["n"] * min(g["n"], 10))
    return {"cx": round(best["x"] / width, 4), "cy": 0.5, "z": 1.0, "faces": len(groups),
            "people": sorted(round(g["x"] / width, 4) for g in groups)}


def _grab(cap, t: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    return frame if ok else None
