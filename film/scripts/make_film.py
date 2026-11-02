#!/usr/bin/env python3
"""
Pakistan: A History of Five Thousand Years
------------------------------------------
Builds a ~5 minute landscape (1920x1080) documentary from:
  * AI-generated stills  (assets_raw/)
  * Urdu narration       (voice/, TTS)
  * original score       (scripts/gen_music.py)
  * title cards burned with libass (Urdu in Noto Nastaliq Urdu)

steps: prep -> clips -> audio -> ass -> final

usage: make_film.py [--step prep|clips|audio|ass|final|all] [--scenes 0-9]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
if not os.path.isdir(FONT_DIR):
    FONT_DIR = "/home/user/fonts/ttf"
FF = os.environ.get("FFMPEG") or __import__("imageio_ffmpeg").get_ffmpeg_exe()

W, H = 1920, 1080
PREP_W, PREP_H = 2240, 1260
FPS = 24
XFADE = 0.9
SR = 48000


def log(*a):
    print(*a, flush=True)


def run(cmd, **kw):
    log("  $", " ".join(cmd[:3]), "...")
    return subprocess.run(cmd, check=True, **kw)


def probe_duration(path):
    out = subprocess.run([FF, "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    for line in out.stderr.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"no duration for {path}")


# ---------------------------------------------------------------- scene table
def load_scenes():
    scenes = json.load(open(os.path.join(ROOT, "data", "scenes.json")))
    for s in scenes:
        vo_path = os.path.join(ROOT, "voice", s["vo"])
        s["vo_dur"] = (probe_duration(vo_path) if os.path.exists(vo_path)
                       else round(len(s.get("vo_text", "")) / 14.1, 2))
        s["dur"] = round(s["lead"] + s["vo_dur"] + s["tail"], 3)
        if s.get("endcard"):
            s["dur"] += s["endcard"]
    # timeline with crossfades
    t = 0.0
    for i, s in enumerate(scenes):
        s["start"] = round(t, 3)
        t += s["dur"] - (XFADE if i < len(scenes) - 1 else 0)
    total = round(t, 3)
    for s in scenes:
        s["vo_start"] = round(s["start"] + s["lead"], 3)
        s["card_in"] = round(s["start"] + 1.0, 3)
        s["card_out"] = round(s["start"] + min(s["dur"] - 0.6, 7.4), 3)
    return scenes, total


# ------------------------------------------------------------------ 1. images
def step_prep(scenes):
    outdir = os.path.join(ROOT, "build", "prep")
    os.makedirs(outdir, exist_ok=True)
    for s in scenes:
        src = os.path.join(ROOT, "assets_raw", s["img"])
        dst = os.path.join(outdir, s["id"] + ".png")
        im = Image.open(src).convert("RGB")
        # fill the 16:9 frame, upscale for the Ken Burns move
        ratio = max(PREP_W / im.width, PREP_H / im.height)
        im = im.resize((round(im.width * ratio), round(im.height * ratio)), Image.LANCZOS)
        left = (im.width - PREP_W) // 2
        top = (im.height - PREP_H) // 2
        im = im.crop((left, top, left + PREP_W, top + PREP_H))
        im = im.filter(ImageFilter.UnsharpMask(radius=2.2, percent=55, threshold=2))
        im.save(dst, optimize=False)
        log(f"  prep {s['id']}  {im.size}")


# ------------------------------------------------------------------- 2. clips
MOTION = {
    # z / x / y expressions, i = on/(N-1)
    "push_in":  ("1+0.15*(on/{n1})", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    "push_out": ("1.15-0.15*(on/{n1})", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    "pan_right": ("1.13", "(iw-iw/zoom)*(on/{n1})", "(ih-ih/zoom)/2"),
    "pan_left":  ("1.13", "(iw-iw/zoom)*(1-on/{n1})", "(ih-ih/zoom)/2"),
}


def step_clips(scenes):
    outdir = os.path.join(ROOT, "build", "clips")
    os.makedirs(outdir, exist_ok=True)
    for s in scenes:
        src = os.path.join(ROOT, "build", "prep", s["id"] + ".png")
        dst = os.path.join(outdir, s["id"] + ".mp4")
        frames = round(s["dur"] * FPS)
        z, x, y = MOTION[s["motion"]]
        n1 = max(1, frames - 1)
        z, x, y = z.format(n1=n1), x.format(n1=n1), y.format(n1=n1)
        vf = (
            f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},"
            f"eq=contrast=1.05:saturation=1.05:gamma=0.99:brightness=-0.01,"
            f"vignette=angle=PI/4.2,"
            f"noise=alls=5:allf=t,"
            f"format=yuv420p"
        )
        run([FF, "-y", "-loglevel", "error", "-i", src, "-vf", vf,
             "-frames:v", str(frames), "-r", str(FPS),
             "-c:v", "libx264", "-preset", "superfast", "-crf", "13",
             "-pix_fmt", "yuv420p", dst])
        log(f"  clip {s['id']}  {s['dur']:.2f}s  {frames}f  {s['motion']}")


# ------------------------------------------------------------------- 3. audio
def decode_voice(s):
    os.makedirs(os.path.join(ROOT, "build", "vo"), exist_ok=True)
    dst = os.path.join(ROOT, "build", "vo", s["id"] + ".wav")
    if not os.path.exists(dst):
        run([FF, "-y", "-loglevel", "error", "-i", os.path.join(ROOT, "voice", s["vo"]),
             "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", dst])
    return dst


def read_wav_mono(path):
    with wave.open(path) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    return raw


def step_audio(scenes, total):
    build = os.path.join(ROOT, "build")
    n = int(total * SR)
    narr = np.zeros(n)
    for s in scenes:
        path = decode_voice(s)
        seg = read_wav_mono(path)
        # 25 ms de-click fades
        f = int(0.025 * SR)
        seg[:f] *= np.linspace(0, 1, f)
        seg[-f:] *= np.linspace(1, 0, f)
        start = int(s["vo_start"] * SR)
        end = min(start + len(seg), n)
        narr[start:end] += seg[: end - start]
        s["vo_dur"] = round(len(seg) / SR, 3)

    narr_path = os.path.join(build, "narration.wav")
    write_wav(narr_path, np.stack([narr, narr], axis=1))

    chapters = [{"id": s["id"], "start": s["start"]} for s in scenes]
    chap_path = os.path.join(build, "chapters.json")
    json.dump(chapters, open(chap_path, "w"), indent=1)

    music_path = os.path.join(build, "music.wav")
    if not os.path.exists(music_path) or os.environ.get("FORCE_MUSIC") == "1":
        run([sys.executable, os.path.join(ROOT, "scripts", "gen_music.py"),
             music_path, f"{total:.3f}", chap_path, narr_path])

    music = read_wav_stereo(music_path)
    if len(music) < n:
        music = np.pad(music, ((0, n - len(music)), (0, 0)))

    # narration sits on top: gentle bus compression feel via tanh, music ducked in the score
    mixed = np.stack([narr, narr], axis=1) * 1.0 + music[:n] * 0.30
    peak = np.abs(mixed).max()
    mixed = mixed / peak * 0.89
    mixed = np.tanh(mixed * 1.1) / np.tanh(1.1)
    mix_path = os.path.join(build, "mix.wav")
    write_wav(mix_path, mixed)
    log(f"  mix {total:.2f}s -> build/mix.wav")
    return mix_path


def read_wav_stereo(path):
    with wave.open(path) as w:
        ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    return raw.reshape(-1, ch)


def write_wav(path, stereo):
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "w") as w:
        w.setnchannels(stereo.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------- 4. ass
def ts(t):
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
Title: Pakistan - A History of Five Thousand Years
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: EnTitle,Cinzel,68,&H00F6EFE4,&H000000FF,&H00201A14,&H8C000000,0,0,0,0,100,100,15,0,1,0,2,7,128,128,40,1
Style: EnDate,EB Garamond,43,&H00D2AC6A,&H000000FF,&H00201A14,&H8C000000,0,0,0,0,100,100,7,0,1,0,1.6,7,128,128,40,1
Style: UrLine,Noto Nastaliq Urdu,144,&H00F6EFE4,&H000000FF,&H00201A14,&H8C000000,0,0,0,0,100,100,0,0,1,0,2,7,128,128,40,1
Style: UrSmall,Noto Nastaliq Urdu,92,&H00E2D3BA,&H000000FF,&H00141010,&H8C000000,0,0,0,0,100,100,0,0,1,0,1.6,5,60,60,60,1
Style: HeroEn,Cinzel,116,&H00F8F2E7,&H000000FF,&H00100C08,&H96000000,0,0,0,0,100,100,34,0,1,0,3,5,60,60,60,1
Style: HeroUr,Noto Nastaliq Urdu,205,&H00F8F2E7,&H000000FF,&H00100C08,&H96000000,0,0,0,0,100,100,0,0,1,0,3,5,60,60,60,1
Style: Tag,EB Garamond,36,&H00DCC188,&H000000FF,&H00100C08,&H96000000,0,0,0,0,100,100,11,0,1,0,1.4,5,60,60,60,1
Style: Cred,EB Garamond,27,&H00C3B7A6,&H000000FF,&H000D0B09,&H96000000,0,0,0,0,100,100,5,0,1,0,1,5,60,60,60,1
Style: Panel,EB Garamond,40,&H00000000,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

RULE = (r"{\p1\c&H00C0A063&\alpha&h30&\an7\pos(%d,%d)}"
        r"m 0 0 l %d 0 l %d 2 l 0 2{\p0}")


def panel(t0, t1, alpha="80"):
    """Full-frame darkening used behind the hero cards."""
    return (f"Dialogue: 0,{ts(t0)},{ts(t1)},Panel,,0,0,0,,"
            f"{{\\p1\\c&H000000&\\alpha&H{alpha}&\\an7\\pos(0,0)\\fad(700,700)}}"
            f"m 0 0 l 1920 0 l 1920 1080 l 0 1080{{\\p0}}")


def build_ass(scenes, total):
    ev = []
    # ---- opening title card
    t0, t1 = 1.4, 7.2
    ev.append(panel(t0, t1))
    ev.append(f"Dialogue: 0,{ts(t0)},{ts(t1)},HeroUr,,0,0,0,,"
              f"{{\\an5\\pos(960,430)\\fad(950,750)}}پاکستان")
    ev.append(f"Dialogue: 0,{ts(t0 + 0.5)},{ts(t1 + 0.4)},Tag,,0,0,0,,"
              + RULE % (830, 505, 260, 260))
    ev.append(f"Dialogue: 0,{ts(t0 + 0.9)},{ts(t1)},HeroEn,,0,0,0,,"
              f"{{\\an5\\pos(960,588)\\fad(850,750)}}PAKISTAN")
    ev.append(f"Dialogue: 0,{ts(t0 + 1.6)},{ts(t1 + 0.6)},Tag,,0,0,0,,"
              f"{{\\an5\\pos(960,676)\\fad(750,650)}}A HISTORY OF FIVE THOUSAND YEARS")
    ev.append(f"Dialogue: 0,{ts(t0 + 2.0)},{ts(t1 + 0.6)},UrSmall,,0,0,0,,"
              f"{{\\an5\\pos(960,760)\\fad(750,650)}}پانچ ہزار سال کی کہانی")

    # ---- chapter cards
    for s in scenes:
        if s.get("kind") in ("title", "closing"):
            continue
        a, b = s["card_in"], s["card_out"]
        ev.append(f"Dialogue: 0,{ts(a)},{ts(b)},EnDate,,0,0,0,,"
                  + RULE % (142, 655, 96, 96))
        ev.append(f"Dialogue: 0,{ts(a + 0.1)},{ts(b)},EnTitle,,0,0,0,,"
                  f"{{\\an1\\pos(144,920)\\fad(520,520)}}{s['en']}")
        if s.get("date"):
            ev.append(f"Dialogue: 0,{ts(a + 0.35)},{ts(b)},EnDate,,0,0,0,,"
                      f"{{\\an1\\pos(146,968)\\fad(520,520)}}{s['date']}")
        ev.append(f"Dialogue: 0,{ts(a + 0.55)},{ts(b)},UrLine,,0,0,0,,"
                  f"{{\\an1\\pos(140,805)\\fad(620,520)}}{s['ur']}")

    # ---- closing end card
    last = scenes[-1]
    if last.get("endcard"):
        ec = last["start"] + last["dur"] - last["endcard"]
        ev.append(panel(ec + 0.4, total - 0.2, "70"))
        ev.append(f"Dialogue: 0,{ts(ec + 0.8)},{ts(total - 0.3)},HeroUr,,0,0,0,,"
                  f"{{\\an5\\pos(960,440)\\fad(950,900)}}کہانی جاری ہے")
        ev.append(f"Dialogue: 0,{ts(ec + 1.4)},{ts(total - 0.3)},Tag,,0,0,0,,"
                  f"{{\\an5\\pos(960,620)\\fad(900,900)\\fs44\\fsp(18)}}THE STORY CONTINUES")
        ev.append(f"Dialogue: 0,{ts(ec + 2.0)},{ts(total - 0.3)},Cred,,0,0,0,,"
                  f"{{\\an5\\pos(960,712)\\fad(800,900)}}Urdu narration  ·  Original ambient score  ·  Imagery created for this film")
        ev.append(f"Dialogue: 0,{ts(ec + 2.0)},{ts(total - 0.3)},UrSmall,,0,0,0,,"
                  f"{{\\an5\\pos(960,760)\\fad(800,900)\\fs68\\c&H00C8BCA9&}}کراچی سے خیبر تک")

    path = os.path.join(ROOT, "build", "film.ass")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ASS_HEAD + "\n".join(ev) + "\n")
    log(f"  ass -> {path} ({len(ev)} events)")
    return path


def build_srt(scenes, total):
    """Urdu subtitle sidecar of the narration."""
    def st(t):
        h, r = divmod(t, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"
    lines = []
    for i, s in enumerate(scenes):
        text = s.get("vo_text", "")
        if not text:
            continue
        lines.append(f"{i+1}\n{st(s['vo_start'])} --> {st(s['vo_start'] + s['vo_dur'])}\n{text}\n")
    with open(os.path.join(ROOT, "out", "pakistan_history_urdu.srt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ------------------------------------------------------------------- 5. final
def make_scrim():
    path = os.path.join(ROOT, "build", "scrim.png")
    yy = np.arange(H).astype(np.float64) / H
    # bottom gradient for the titles, very light top gradient for the crown
    bottom = np.clip((yy - 0.46) / 0.54, 0, 1) ** 1.7 * 0.86
    top = np.clip((0.22 - yy) / 0.22, 0, 1) ** 2 * 0.30
    xx = np.arange(W).astype(np.float64) / W
    left_edge = np.clip((0.34 - xx) / 0.34, 0, 1) ** 1.6 * 0.55
    edge = left_edge[None, :] * np.clip((yy - 0.28) / 0.72, 0, 1)[:, None] ** 1.2 * 0.75
    alpha = np.clip(np.maximum(bottom[:, None], edge * 0.9), 0, 1)
    alpha = np.clip(alpha + top[:, None] * 0.6, 0, 1)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def step_final(scenes, total, mix_path, ass_path):
    scrim = make_scrim()
    inputs = []
    for s in scenes:
        inputs += ["-i", os.path.join(ROOT, "build", "clips", s["id"] + ".mp4")]
    inputs += ["-i", scrim, "-i", mix_path]
    scrim_idx = len(scenes)
    audio_idx = scrim_idx + 1

    fc = []
    for i, s in enumerate(scenes):
        fc.append(f"[{i}:v]fps={FPS},settb=AVTB,format=yuv420p[v{i}]")

    cur = "v0"
    offset = 0.0
    for i in range(1, len(scenes)):
        offset += scenes[i - 1]["dur"] - XFADE
        trans = "fadeblack" if scenes[i]["id"] in ("s09", "s10") else "fade"
        out = f"x{i}"
        fc.append(f"[{cur}][v{i}]xfade=transition={trans}:duration={XFADE}:offset={offset:.3f}[{out}]")
        cur = out

    fc.append(f"[{cur}][{scrim_idx}:v]overlay=0:0:format=auto[vs]")
    fc.append(f"[vs]ass=filename='{ass_path}':fontsdir='{FONT_DIR}'[vt]")
    fc.append(f"[vt]fade=t=in:st=0:d=1.4,fade=t=out:st={total - 2.4:.2f}:d=2.4,format=yuv420p[vout]")
    fc.append(f"[{audio_idx}:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")

    outdir = os.path.join(ROOT, "out")
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, "Pakistan_History_Urdu_1080p.mp4")
    cmd = [FF, "-y", "-loglevel", "error", "-stats", *inputs,
           "-filter_complex", ";".join(fc),
           "-map", "[vout]", "-map", "[aout]",
           "-t", f"{total:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-profile:v", "high", "-level", "4.1", "-maxrate", "9M", "-bufsize", "16M",
           "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-movflags", "+faststart", dst]
    run(cmd)
    log(f"  final -> {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all")
    ap.add_argument("--scenes", default=None, help="e.g. 0-3 for a partial build")
    args = ap.parse_args()

    scenes, total = load_scenes()
    if args.scenes:
        a, b = args.scenes.split("-")
        scenes = scenes[int(a): int(b) + 1]
        scenes, total = load_scenes_from(scenes)

    log(f"film: {len(scenes)} scenes, {total:.1f}s = {int(total // 60)}:{total % 60:04.1f}")
    for s in scenes:
        log(f"   {s['id']} start {s['start']:7.2f}  dur {s['dur']:6.2f}  vo {s['vo_dur']:5.2f}")

    ass_path = os.path.join(ROOT, "build", "film.ass")
    mix_path = os.path.join(ROOT, "build", "mix.wav")
    if args.step in ("prep", "all"):
        step_prep(scenes)
    if args.step in ("clips", "all"):
        step_clips(scenes)
    if args.step in ("audio", "all"):
        mix_path = step_audio(scenes, total)
    if args.step in ("ass", "all"):
        ass_path = build_ass(scenes, total)
        build_srt(scenes, total)
    if args.step in ("final", "all"):
        step_final(scenes, total, mix_path, ass_path)


def load_scenes_from(subset):
    t = 0.0
    for i, s in enumerate(subset):
        s["start"] = round(t, 3)
        s["vo_start"] = round(t + s["lead"], 3)
        s["card_in"] = round(t + 1.0, 3)
        s["card_out"] = round(t + min(s["dur"] - 0.6, 7.4), 3)
        t += s["dur"] - (XFADE if i < len(subset) - 1 else 0)
    return subset, round(t, 3)


if __name__ == "__main__":
    main()
