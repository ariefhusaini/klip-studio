"""Klip Studio API: upload a video, let AI find clips, edit them, render vertical shorts."""
from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline, render

DATA = Path(os.getenv("DATA_DIR", "./data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).parent / "static"
_lock = threading.Lock()

app = FastAPI(title="Klip Studio")


# ---------------------------------------------------------------- storage

def pdir(pid: str) -> Path:
    if not pid.isalnum():
        raise HTTPException(404)
    d = DATA / pid
    if not d.exists():
        raise HTTPException(404, "Projek tidak dijumpai")
    return d


def load(pid: str) -> dict:
    return json.loads((pdir(pid) / "project.json").read_text())


def save(pid: str, proj: dict) -> None:
    with _lock:
        tmp = DATA / pid / "project.json.tmp"
        tmp.write_text(json.dumps(proj, ensure_ascii=False))
        tmp.replace(DATA / pid / "project.json")


def update(pid: str, **kw) -> dict:
    with _lock:
        proj = json.loads((DATA / pid / "project.json").read_text())
        proj.update(kw)
        (DATA / pid / "project.json").write_text(json.dumps(proj, ensure_ascii=False))
    return proj


# ---------------------------------------------------------------- analysis job

def analyse(pid: str) -> None:
    d = DATA / pid
    proj = load(pid)
    src = d / proj["source"]
    try:
        update(pid, status="analysing", step="Membaca video")
        info = pipeline.probe(src)
        update(pid, **info)
        words: list[dict] = []
        if info["has_audio"]:
            update(pid, step="Menukar suara jadi teks")
            chunks = pipeline.extract_audio(src, d, info["duration"])
            words = pipeline.transcribe(chunks, info["duration"])
            for c, _ in chunks:
                c.unlink(missing_ok=True)
        update(pid, words=words, step="AI mencari detik terbaik")
        found = pipeline.find_highlights(words, info["duration"])
        clips = []
        for i, h in enumerate(found):
            update(pid, step=f"Mengesan muka untuk klip {i + 1} daripada {len(found)}")
            crop = pipeline.frame_for_clip(src, h["start"], h["end"], info["width"])
            clips.append({**h, "id": uuid.uuid4().hex[:8], "crop": crop, "captions": True,
                          "captionStyle": "kuning", "captionPos": 0.78, "render": None})
        update(pid, clips=clips, status="ready", step="")
    except Exception as e:  # surface the failure to the editor instead of hanging
        traceback.print_exc()
        update(pid, status="error", step=f"Gagal: {e}")


def do_render(pid: str, cid: str) -> None:
    proj = load(pid)
    clip = next(c for c in proj["clips"] if c["id"] == cid)
    out = DATA / pid / "renders" / f"{cid}.mp4"
    out.parent.mkdir(exist_ok=True)
    try:
        render.render_clip(DATA / pid / proj["source"], out, clip, proj.get("words", []),
                           proj["width"], proj["height"], proj.get("aspect", "9:16"))
        _set_render(pid, cid, {"status": "done", "url": f"/media/{pid}/renders/{cid}.mp4"})
    except Exception as e:
        traceback.print_exc()
        _set_render(pid, cid, {"status": "error", "error": str(e)})


def _set_render(pid: str, cid: str, value: dict | None) -> None:
    with _lock:
        path = DATA / pid / "project.json"
        proj = json.loads(path.read_text())
        for c in proj["clips"]:
            if c["id"] == cid:
                c["render"] = value
        path.write_text(json.dumps(proj, ensure_ascii=False))


# ---------------------------------------------------------------- API

@app.post("/api/projects")
async def create_project(file: UploadFile, background: BackgroundTasks):
    pid = uuid.uuid4().hex[:12]
    d = DATA / pid
    d.mkdir()
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    with (d / f"source{ext}").open("wb") as f:
        shutil.copyfileobj(file.file, f, 1024 * 1024)
    save(pid, {"id": pid, "name": file.filename, "source": f"source{ext}", "status": "queued",
               "step": "Dalam giliran", "aspect": "9:16", "clips": [], "words": []})
    background.add_task(analyse, pid)
    return {"id": pid}


@app.get("/api/projects")
def list_projects():
    out = []
    for d in sorted(DATA.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (d / "project.json").exists():
            p = json.loads((d / "project.json").read_text())
            out.append({k: p.get(k) for k in ("id", "name", "status", "duration")} | {"clips": len(p.get("clips", []))})
    return out


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = load(pid)
    p["video"] = f"/media/{pid}/{p['source']}"
    return p


class ClipsIn(BaseModel):
    clips: list[dict]
    aspect: str | None = None


@app.put("/api/projects/{pid}/clips")
def put_clips(pid: str, body: ClipsIn):
    with _lock:
        path = pdir(pid) / "project.json"
        proj = json.loads(path.read_text())
        old = {c["id"]: c.get("render") for c in proj["clips"]}
        for c in body.clips:
            c.setdefault("id", uuid.uuid4().hex[:8])
            c["render"] = old.get(c["id"])  # render state is owned by the server
        proj["clips"] = body.clips
        if body.aspect in render.OUTPUT_SIZES:
            proj["aspect"] = body.aspect
        path.write_text(json.dumps(proj, ensure_ascii=False))
    return {"ok": True}


class WordEdit(BaseModel):
    index: int
    text: str


@app.put("/api/projects/{pid}/words")
def edit_word(pid: str, body: WordEdit):
    with _lock:
        path = pdir(pid) / "project.json"
        proj = json.loads(path.read_text())
        if not 0 <= body.index < len(proj["words"]):
            raise HTTPException(400, "Perkataan tidak wujud")
        proj["words"][body.index]["w"] = body.text.strip()
        path.write_text(json.dumps(proj, ensure_ascii=False))
    return {"ok": True}


@app.post("/api/projects/{pid}/clips/{cid}/render")
def start_render(pid: str, cid: str, background: BackgroundTasks):
    proj = load(pid)
    if not any(c["id"] == cid for c in proj["clips"]):
        raise HTTPException(404, "Klip tidak dijumpai")
    _set_render(pid, cid, {"status": "rendering"})
    background.add_task(do_render, pid, cid)
    return {"ok": True}


@app.get("/api/styles")
def styles():
    return {"captions": {k: v["label"] for k, v in render.CAPTION_STYLES.items()},
            "aspects": list(render.OUTPUT_SIZES)}


@app.get("/media/{pid}/{path:path}")
def media(pid: str, path: str):
    base = pdir(pid).resolve()
    f = (base / path).resolve()
    if base not in f.parents or not f.is_file() or f.suffix in (".json", ".ass", ".tmp"):
        raise HTTPException(404)
    return FileResponse(f)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
