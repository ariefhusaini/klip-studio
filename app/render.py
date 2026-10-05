"""Render one clip: crop to the chosen frame, scale, burn word-by-word captions."""
from __future__ import annotations

import subprocess
from pathlib import Path

OUTPUT_SIZES = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350), "16:9": (1920, 1080)}

# ASS colours are &HAABBGGRR
CAPTION_STYLES = {
    "kuning": {"label": "Bold kuning", "font": "DejaVu Sans", "bold": 1, "size": 0.062, "upper": True,
               "base": "&H00FFFFFF", "hi": "&H0000E1FF", "outline": "&H00000000", "border": 1, "out_w": 0.006,
               "shadow": 0.002, "words": 3},
    "kotak": {"label": "Kotak hitam", "font": "DejaVu Sans", "bold": 1, "size": 0.05, "upper": False,
              "base": "&H00FFFFFF", "hi": "&H0066D9FF", "outline": "&H66000000", "border": 3, "out_w": 0.012,
              "shadow": 0, "words": 4},
    "minimal": {"label": "Minimal", "font": "DejaVu Sans", "bold": 0, "size": 0.045, "upper": False,
                "base": "&H00FFFFFF", "hi": "&H00FFFFFF", "outline": "&H00000000", "border": 1, "out_w": 0.003,
                "shadow": 0.002, "words": 6},
}


def _ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def build_ass(words: list[dict], clip_start: float, clip_end: float, w: int, h: int, style: str,
              position: float = 0.72) -> str:
    st = CAPTION_STYLES.get(style, CAPTION_STYLES["kuning"])
    size = round(h * st["size"]) if h > w else round(h * st["size"] * 1.5)
    margin_v = round(h * (1 - position))
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{size},{st['base']},{st['base']},{st['outline']},&H80000000,{-1 if st['bold'] else 0},0,0,0,100,100,0,0,{st['border']},{max(1, round(h * st['out_w']))},{round(h * st['shadow'])},2,{round(w * 0.08)},{round(w * 0.08)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    clip_words = [x for x in words if x["e"] > clip_start and x["s"] < clip_end and x["w"].strip()]
    lines = []
    n = st["words"]
    for i in range(0, len(clip_words), n):
        group = clip_words[i:i + n]
        for j, cur in enumerate(group):
            s = cur["s"] - clip_start
            e = (group[j + 1]["s"] if j + 1 < len(group) else cur["e"] + 0.15) - clip_start
            if i + n < len(clip_words) and j + 1 == len(group):
                e = min(e, clip_words[i + n]["s"] - clip_start)
            parts = []
            for k, wd in enumerate(group):
                txt = _esc(wd["w"].upper() if st["upper"] else wd["w"])
                parts.append(f"{{\\c{st['hi']}}}{txt}{{\\c{st['base']}}}" if k == j else txt)
            lines.append(f"Dialogue: 0,{_ts(s)},{_ts(max(e, s + 0.05))},Cap,,0,0,0,,{' '.join(parts)}")
    return head + "\n".join(lines) + "\n"


def render_clip(src: Path, out: Path, clip: dict, words: list[dict], src_w: int, src_h: int,
                aspect: str = "9:16") -> None:
    ow, oh = OUTPUT_SIZES.get(aspect, OUTPUT_SIZES["9:16"])
    ar = ow / oh
    crop = clip.get("crop") or {"cx": 0.5, "cy": 0.5, "z": 1.0}
    if src_w / src_h > ar:
        bh, bw = src_h, src_h * ar
    else:
        bw, bh = src_w, src_w / ar
    bw, bh = bw / crop.get("z", 1), bh / crop.get("z", 1)
    x = min(max(crop["cx"] * src_w - bw / 2, 0), src_w - bw)
    y = min(max(crop["cy"] * src_h - bh / 2, 0), src_h - bh)
    bw, bh, x, y = (int(round(v)) // 2 * 2 for v in (bw, bh, x, y))

    vf = f"crop={bw}:{bh}:{x}:{y},scale={ow}:{oh}:flags=lanczos,setsar=1"
    ass_path = out.with_suffix(".ass")
    if clip.get("captions", True) and words:
        ass_path.write_text(build_ass(words, clip["start"], clip["end"], ow, oh,
                                      clip.get("captionStyle", "kuning"), clip.get("captionPos", 0.72)),
                            encoding="utf-8")
        vf += f",ass={ass_path.as_posix()}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{clip['start']:.3f}", "-to", f"{clip['end']:.3f}",
                    "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)],
                   check=True)
