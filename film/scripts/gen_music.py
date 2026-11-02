#!/usr/bin/env python3
"""
Original ambient score for the film, synthesised from scratch with numpy/scipy.

A slow D-minor drone (tanpura-like) built from detuned voices with breathing
amplitude LFOs, filtered air, sparse plucked motifs and low impact hits on
chapter cuts. Everything is generated -- no samples, no external music.

usage: gen_music.py OUT.wav DURATION chapters.json [narration.wav]
"""
import sys
import json
import wave
import numpy as np
from scipy.signal import lfilter, butter, sosfilt

SR = 48000
RNG = np.random.default_rng(20261102)


def lowpass(x, cutoff, order=2):
    sos = butter(order, cutoff / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, x)


def slow_curve(n, rate_hz):
    """Smooth random automation curve in ~[0,1]."""
    k = max(4, int(n / SR * rate_hz) + 2)
    pts = RNG.random(k)
    curve = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, k), pts)
    win = int(SR * 1.2)
    if win > 3 and win < len(curve):
        kernel = np.hanning(win)
        kernel /= kernel.sum()
        curve = np.convolve(curve, kernel, mode="same")
    return np.clip(curve, 0, 1)


def drone(T, seed):
    rng = np.random.default_rng(seed)
    n = int(T * SR)
    t = np.arange(n) / SR
    base = 73.416  # D2
    voices = [
        (1.0, 0.50),   # D2
        (2.0, 0.32),   # D3
        (3.0, 0.18),   # A3
        (2.0 * 2 ** (3 / 12), 0.15),  # F3 minor third
        (4.0, 0.06),   # D4
        (1.5, 0.12),   # A2
    ]
    out = np.zeros(n)
    for i, (mult, amp) in enumerate(voices):
        f = base * mult
        drift = 1 + 0.0016 * np.sin(2 * np.pi * (0.010 + 0.004 * i) * t + rng.random() * 6.28)
        lfo_rate = 0.013 + 0.008 * rng.random()
        lfo = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * lfo_rate * t + rng.random() * 6.28))
        if i == 3:  # the minor third breathes: darker during the harsh chapters
            lfo = lfo * (0.25 + 0.75 * slow_curve(n, 0.018))
        phase = 2 * np.pi * f * np.cumsum(drift) / SR
        v = np.sin(phase) + 0.26 * np.sin(2 * phase) + 0.09 * np.sin(3 * phase)
        out += amp * lfo * v
    return out / 2.2


def air(T, seed):
    rng = np.random.default_rng(seed)
    n = int(T * SR)
    noise = lowpass(rng.standard_normal(n), 850, order=2)
    noise /= np.abs(noise).max() + 1e-9
    env = 0.25 + 0.5 * slow_curve(n, 0.05)
    return noise * env * 0.15


def impact(dur, f0, seed):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.6 * np.exp(-t * 1.4))
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(phase) * np.exp(-t * 2.4)
    click = lowpass(rng.standard_normal(n), 260, order=1) * np.exp(-t * 9) * 0.30
    return (body + click) / 1.35


def pluck(freq, dur, seed):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.0032 * np.sin(2 * np.pi * 5.1 * t)
    phase = 2 * np.pi * freq * np.cumsum(vib) / SR
    sig = sum(a * np.sin(h * phase) * np.exp(-t * (2.0 + h * 1.2))
              for h, a in [(1, 1.0), (2, 0.40), (3, 0.20), (4, 0.11), (5, 0.06)])
    return sig * np.clip(t / 0.012, 0, 1) / 1.9


def duck_envelope(path, T):
    """Follow the narration loudness so the score sits underneath it."""
    with wave.open(path) as w:
        sr_in, ch = w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    n_target = int(T * SR)
    if sr_in != SR:
        idx = np.clip((np.arange(int(len(raw) * SR / sr_in)) * sr_in / SR).astype(int), 0, len(raw) - 1)
        raw = raw[idx]
    raw = np.concatenate([raw, np.zeros(max(0, n_target - len(raw)))])[:n_target]
    win = int(SR * 0.05)
    m = len(raw) // win
    rms = np.sqrt(np.mean(raw[: m * win].reshape(m, win) ** 2, axis=1))
    env = np.clip(rms / (rms.max() + 1e-9) * 2.0, 0, 1)
    env = np.repeat(env, win)
    env = np.concatenate([env, np.zeros(len(raw) - len(env))])
    # fast attack, slow release
    att = np.exp(-1 / (SR * 0.03))
    rel = np.exp(-1 / (SR * 0.55))
    out = np.empty_like(env)
    acc = 0.0
    up = np.exp(-1 / (SR * 0.03))
    y = lfilter([1 - rel], [1, -rel], env)
    y2 = lfilter([1 - att], [1, -att], env)
    out = np.maximum(y * 0.35, y2)
    return np.clip(out, 0, 1)


def build(T, chapters, duck=None):
    n = int(T * SR)
    t = np.arange(n) / SR
    left = drone(T, 11) + air(T, 12)
    right = drone(T, 29) + air(T, 31)

    shape = np.clip(t / 6.0, 0, 1) * (0.92 + 0.08 * np.sin(2 * np.pi * 0.033 * t))
    shape *= np.clip((T - t) / 7.0, 0, 1)
    left *= shape
    right *= shape

    for i, ch in enumerate(chapters):
        start = ch["start"]
        if not (0 <= start < T - 4):
            continue
        hit = impact(4.5, 54.0 if i % 2 == 0 else 46.0, 100 + i)
        s = int(start * SR)
        seg = min(len(hit), n - s)
        gain = 0.30 if i else 0.45
        left[s:s + seg] += hit[:seg] * gain
        right[s:s + seg] += hit[:seg] * gain * 0.93
        if i % 2 == 1 and start < T - 8:
            note = [146.83, 174.61, 220.0, 293.66][(i // 2) % 4]
            p = pluck(note, 7.0, 200 + i)
            s2 = int((start + 0.5) * SR)
            seg2 = min(len(p), n - s2)
            left[s2:s2 + seg2] += p[:seg2] * 0.13
            right[s2:s2 + seg2] += p[:seg2] * 0.10

    stereo = np.stack([left, right], axis=1)
    if duck is not None:
        stereo *= (1.0 - 0.60 * duck)[:, None]

    peak = np.abs(stereo).max()
    if peak > 0:
        stereo = stereo / peak * 0.70
    stereo = np.tanh(stereo * 1.2) / np.tanh(1.2)
    return stereo


def main():
    out, T, chap_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    chapters = json.load(open(chap_path))
    duck = duck_envelope(sys.argv[4], T) if len(sys.argv) > 4 else None
    stereo = build(T, chapters, duck)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(out, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"music: {out}  {T:.2f}s  peak {np.abs(stereo).max():.3f}")


if __name__ == "__main__":
    main()
