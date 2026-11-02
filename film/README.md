# Pakistan — A History of Five Thousand Years

An original 5-minute cinematic documentary, **narrated in Urdu**, delivered in landscape
**1920×1080 / 24 fps**.

> **Final film:** `film/out/Pakistan_History_Urdu_1080p.mp4`
> **Subtitles:** `film/out/pakistan_history_urdu.srt` (Urdu narration, sidecar)

This folder is self-contained and independent of the HunyuanVideo model code at the repo
root — the model in this repository needs a multi-GPU cluster, so the film was built with a
compositing pipeline instead (generated stills + Ken Burns motion + TTS narration +
synthesised score + libass typography).

---

## Structure of the film

13 chapters, ~5:04 total, each chapter built as its own part and then joined with cross-dissolves.

| # | Chapter | Date on screen | Urdu title |
|---|---------|----------------|------------|
| — | Opening title card | — | پاکستان |
| 1 | Mohenjo-daro | 2500 BCE | موہنجو دڑو اور ہڑپہ |
| 2 | Taxila | 326 BCE | ٹیکسلا اور گندھارا کا فن |
| 3 | The Arrival | 711 CE | محمد بن قاسم کا سندھ |
| 4 | The Sufis | 11th–17th century | اولیا، قوالی اور محبت |
| 5 | The Mughals | 1526–1739 | مغلیہ دور میں لاہور |
| 6 | The British Raj | 1849–1947 | برطانوی راج اور 1857 |
| 7 | The Awakening | 1875–1930 | علی گڑھ اور اقبال کا خواب |
| 8 | The Resolution | 23 March 1940 | قراردادِ پاکستان |
| 9 | Freedom | 14 August 1947 | آزادی اور ہجرت |
| 10 | The Trials | 1948–1971 | آزمائشیں اور سبق |
| 11 | The Nation Today | 1992–today | آج کا پاکستان |
| — | End card | — | کہانی جاری ہے |

## How it is made

| Stage | Script | Output |
|-------|--------|--------|
| Still frames | (image generation) | `assets_raw/*.jpg` |
| Urdu narration | (TTS) | `voice/s00–s12.mp3` |
| Frame prep + Ken Burns | `scripts/make_film.py --step prep/clips` | `build/prep`, `build/clips` |
| Original score | `scripts/gen_music.py` | `build/music.wav` |
| Narration mix | `scripts/make_film.py --step audio` | `build/mix.wav` |
| Titles, subtitles | `scripts/make_film.py --step ass` | `build/film.ass` |
| Join + grade + mux | `scripts/make_film.py --step final` | `out/*.mp4` |

Full rebuild:

```bash
python3 scripts/make_film.py --step all      # prep → clips → audio → ass → final
```

### Craft notes

* **Motion** — every still gets a slow push-in / pull-out / lateral drift (max 15% scale over
  the shot) so the images breathe instead of feeling like a slideshow.
* **Grade** — mild contrast/saturation lift, vignette and temporal film grain on every shot;
  the 1947 and 1971 chapters cut through *fade-to-black* rather than a dissolve.
* **Typography** — Noto Nastaliq Urdu for the Urdu titles, Cinzel for the Latin chapter
  names, EB Garamond for dates and credits, burned with libass (proper Urdu shaping and RTL).
* **Score** — generated from scratch with numpy: a detuned D-minor drone whose minor third
  breathes across the film, filtered air, low impact hits on every chapter cut, sparse
  plucks, all ducked under the narration by a loudness-following envelope.
* **Audio master** — mixed to 48 kHz stereo, then `loudnorm I=-16 LUFS, TP=-1.5`.

## Urdu narration (script)

The complete narration is in `data/scenes.json` (`vo_text` field) and in the sidecar `.srt`.
