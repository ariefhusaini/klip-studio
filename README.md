# Klip Studio

Upload a long video (podcast, interview, livestream). AI finds the best moments, frames the person who is talking, adds word-by-word captions, and renders vertical shorts for TikTok, Reels and Shorts.

## What Phase 1 does

1. **Upload** a video in the browser.
2. **Transcribe** the audio with word timestamps (OpenAI Whisper or Deepgram).
3. **Find highlights** with Claude: each clip gets a title, hook, a 0–100 score and a reason.
4. **Frame the speaker**: OpenCV finds faces, groups them per person and follows the one whose mouth moves most. Every clip remembers its own frame, so a podcast can cut from host to guest.
5. **Edit**: trim, drag the picture to reframe, switch person, zoom, pick a caption style, fix caption spelling, add or delete clips.
6. **Render** with ffmpeg to 1080×1920 (or 1:1, 4:5, 16:9) MP4 with captions burned in.

Not in Phase 1: importing from a YouTube link, scheduling, and auto-posting.

## Run locally

Needs Python 3.11+ and ffmpeg.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the keys
set -a; . ./.env; set +a
uvicorn app.main:app --reload
```

Open http://localhost:8000.

To try the whole flow without any API keys, set `MOCK_AI=1`: the transcript is fake and clips are spaced evenly, but face framing and rendering are real.

## Run with Docker

```bash
docker build -t klip-studio .
docker run -p 8000:8000 --env-file .env -v $PWD/data:/data klip-studio
```

## Layout

- `app/main.py`: HTTP API and background jobs
- `app/pipeline.py`: audio extraction, transcription, highlight picking, speaker framing
- `app/render.py`: crop, scale and caption burn-in (ASS subtitles via libass)
- `app/static/index.html`: the editor

Projects are stored as folders under `DATA_DIR` (video, `project.json`, renders). There are no user accounts yet, so do not expose the server publicly without putting it behind a login.
