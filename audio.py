"""Sound design for the Rogue pitch video, synthesized and synced to pitch.html timings.

python audio.py  ->  soundtrack.wav (48 kHz stereo, 63 s)
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 48000
DUR = 63.0
N = int(SR * (DUR + 0.2))
rng = np.random.default_rng(42)

BUS = {k: np.zeros((2, N)) for k in ("music", "music_rev", "drums", "sfx", "sfx_rev")}


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(bus, sig, t, pan=0.0, gain=1.0):
    i = int(round(t * SR))
    if i >= N:
        return
    if sig.ndim == 1:
        th = (pan + 1) * np.pi / 4
        sig = np.stack([sig * np.cos(th), sig * np.sin(th)])
    n = min(sig.shape[1], N - i)
    if i < 0:
        sig = sig[:, -i:]
        n = min(sig.shape[1], N)
        i = 0
    BUS[bus][:, i:i + n] += sig[:, :n] * gain


def tt(sec):
    return np.arange(int(sec * SR)) / SR


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "lp", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "hp", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bp", fs=SR, output="sos"), x)


# ------------------------------------------------------------------ instruments
def pad_note(f, dur, att=0.9, rel=1.8, bright=0.62, det=0.005):
    t = tt(dur + rel)
    out = np.zeros_like(t)
    for d in (-det, 0.0, det):
        ff = f * (1 + d)
        ph = rng.random() * 6.283
        H = int(min(10, 4500 / ff))
        for h in range(1, H + 1):
            out += np.sin(6.283 * ff * h * t + ph * h) * (bright ** (h - 1)) / h
    env = np.clip(t / att, 0, 1) ** 2
    env *= np.where(t < dur, 1.0, np.exp(-(t - dur) * 3.2 / rel))
    lfo = 1 + 0.08 * np.sin(6.283 * (0.15 + rng.random() * 0.2) * t + rng.random() * 6)
    return out * env * lfo / 3


def kick(length=0.5, f0=46, f1=150, dec=6.5, sweep=32):
    t = tt(length)
    f = f0 + (f1 - f0) * np.exp(-t * sweep)
    s = np.sin(6.283 * np.cumsum(f) / SR) * np.exp(-t * dec) * (1 - np.exp(-t * 900))
    c = hp(rng.standard_normal(int(0.006 * SR)), 2500) * np.linspace(1, 0, int(0.006 * SR)) * 0.25
    s[:len(c)] += c
    return s


def hat(length=0.07, dec=60, f=7500):
    t = tt(length)
    return hp(rng.standard_normal(len(t)), f, 4) * np.exp(-t * dec)


def pluck(f, length=0.4, bright=1.0, dec=8.0):
    t = tt(length)
    mod = np.sin(6.283 * f * 2 * t) * np.exp(-t * 14) * 1.3 * bright
    return np.sin(6.283 * f * t + mod) * np.exp(-t * dec) * (1 - np.exp(-t * 600))


def bell(f, length=2.2):
    t = tt(length)
    s = np.zeros_like(t)
    for r, a, d in ((1, 1, 2.2), (2.76, .45, 4.5), (5.4, .22, 7), (8.93, .1, 11)):
        s += np.sin(6.283 * f * r * t) * a * np.exp(-t * d)
    return s * (1 - np.exp(-t * 500)) * 0.5


def blip(f0, f1, length=0.07):
    t = tt(length)
    f = f0 * (f1 / f0) ** (t / length)
    return np.sin(6.283 * np.cumsum(f) / SR) * np.sin(np.pi * t / length) ** 0.6 * np.exp(-t * 18)


def click(gain=1.0, f=2200):
    n = int(0.012 * SR)
    t = np.arange(n) / SR
    s = bp(rng.standard_normal(n), 1500, 7000) * np.exp(-t * 500) * 0.8
    s += np.sin(6.283 * f * t) * np.exp(-t * 700) * 0.5
    return s * gain


def whoosh(length, f_from, f_to, shape=0.6, chunks=96):
    """Band-passed noise with a swept centre. shape = where the swell peaks (0..1)."""
    n = int(length * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    step = n // chunks
    for k in range(chunks):
        a, b = k * step, (k + 1) * step if k < chunks - 1 else n
        u = (k + .5) / chunks
        fc = f_from * (f_to / f_from) ** u
        out[a:b] = sosfilt(butter(2, [max(30, fc / 1.7), min(SR / 2 - 100, fc * 1.7)], "bp", fs=SR, output="sos"), x[max(0, a - 2048):b])[-(b - a):]
    t = np.arange(n) / n
    env = np.where(t < shape, (t / shape) ** 2, ((1 - t) / (1 - shape)) ** 1.6)
    return out * env


def riser(length, f0=200, f1=2400):
    t = tt(length)
    u = t / length
    f = f0 * (f1 / f0) ** u
    tone = np.sin(6.283 * np.cumsum(f) / SR) * u ** 2 * 0.25
    return tone + whoosh(length, 400, 6000, shape=0.97) * 0.9


def impact(big=1.0):
    k = kick(1.4, 40, 190, dec=3.2 / big, sweep=22) * 1.1
    t = tt(1.4)
    k += np.sin(6.283 * 36 * t) * np.exp(-t * 2.2) * (1 - np.exp(-t * 200)) * 0.55 * big
    nz = lp(rng.standard_normal(len(t)), 2400) * np.exp(-t * 9) * 0.35
    return k + nz


def pixel_rain(t0, length=0.84, count=34, gain=0.5, notes=(84, 86, 88, 91, 93, 96)):
    for k in range(count):
        u = (k + rng.random() * .6) / count
        m = notes[rng.integers(len(notes))]
        add("sfx", blip(mtof(m), mtof(m) * 1.06, 0.045) * (0.6 + rng.random() * .4), t0 + u * length, pan=-0.85 + 1.7 * u, gain=gain * 0.22)
        if rng.random() < .3:
            add("sfx_rev", blip(mtof(m), mtof(m), 0.04), t0 + u * length, pan=-0.85 + 1.7 * u, gain=gain * 0.12)


# ------------------------------------------------------------------ harmony
CH = {
    "Dm9": [50, 53, 57, 60, 64], "Bb9": [46, 50, 53, 57, 60], "C11": [48, 55, 58, 62, 65],
    "F9": [53, 57, 60, 64, 67], "Gm9": [43, 50, 53, 57, 58], "Bbhi": [58, 62, 65, 69, 72],
}
ROOT = {"Dm9": 38, "Bb9": 34, "C11": 36, "F9": 41, "Gm9": 31, "Bbhi": 34}
SCHED = [  # start, end, chord, pad gain, brightness
    (0.0, 6.2, "Dm9", .55, .5), (6.2, 10.2, "Dm9", .7, .6), (10.2, 14.2, "Bb9", .7, .6), (14.2, 16.7, "C11", .7, .62),
    (16.7, 20.25, "Dm9", .6, .55), (20.25, 23.0, "F9", .8, .7),
    (26.2, 31.0, "Bbhi", .5, .45), (31.0, 34.6, "Gm9", .6, .5),
    (34.6, 39.2, "Dm9", .6, .55), (39.2, 44.6, "F9", .75, .66),
    (44.6, 48.6, "Bb9", .65, .6), (48.6, 52.6, "C11", .65, .62), (52.6, 55.6, "Dm9", .65, .6),
    (55.6, 59.15, "Bb9", .7, .55), (59.15, 63.2, "F9", .9, .62),
]


def chord_at(t):
    for a, b, c, *_ in SCHED:
        if a <= t < b:
            return c
    return "Dm9"


for a, b, c, g, br in SCHED:
    for i, m in enumerate(CH[c]):
        s = pad_note(mtof(m), b - a, bright=br)
        pan = (i / (len(CH[c]) - 1) - .5) * 0.9
        add("music", s, a, pan=pan, gain=g * 0.10)
        add("music_rev", s, a, pan=-pan, gain=g * 0.06)

# low drone under "speed only gets you so far"
dr = pad_note(mtof(38), 3.4, att=0.6, rel=1.2, bright=.35) + pad_note(mtof(45), 3.4, att=0.6, rel=1.2, bright=.3) * .6
add("music", dr, 23.2, gain=0.16)
add("music_rev", dr, 23.2, gain=0.08)

# ------------------------------------------------------------------ rhythm sections
SECTIONS = [  # start, end, step, kick, hats, bass, arp(step divisor), level
    (6.2, 16.7, .5, True, True, True, 2, 1.0),
    (16.7, 20.25, 1.0, True, True, True, 0, .8),
    (20.25, 23.0, .5, True, True, True, 2, 1.0),
    (35.0, 39.2, 1.0, True, False, True, 0, .75),
    (39.2, 44.6, .5, True, True, True, 2, 1.0),
    (44.6, 55.6, .5, True, True, True, 4, 1.05),
]
kick_times = []
for a, b, step, k, h, bass, arp, lvl in SECTIONS:
    n = int(round((b - a) / step))
    for j in range(n):
        tb = a + j * step
        if k:
            add("drums", kick(), tb, gain=0.62 * lvl)
            kick_times.append(tb)
        if h:
            add("drums", hat(), tb + step / 2, pan=.25, gain=0.07 * lvl)
            if step <= .5:
                add("drums", hat(0.03, 120, 9000), tb + step / 4, pan=-.3, gain=0.03 * lvl)
                add("drums", hat(0.03, 120, 9000), tb + 3 * step / 4, pan=-.3, gain=0.03 * lvl)
        if bass:
            c = chord_at(tb + .01)
            for q in (0, step / 2):
                t = tt(.24)
                f = mtof(ROOT[c])
                s = (np.sin(6.283 * f * t) + .25 * np.sin(6.283 * 2 * f * t)) * np.exp(-t * 7) * (1 - np.exp(-t * 300))
                add("music", lp(s, 600), tb + q, gain=0.22 * lvl)
        if arp:
            c = chord_at(tb + .01)
            tones = CH[c] + [m + 12 for m in CH[c]]
            for q in range(arp):
                idx = (j * arp + q) % len(tones)
                m = tones[[0, 2, 4, 1, 3, 5, 7, 9, 6, 8][idx % 10] % len(tones)] + 12
                tq = tb + q * step / arp
                s = pluck(mtof(m), .3, bright=.8)
                add("music", s, tq, pan=(.4 if (j * arp + q) % 2 else -.4), gain=0.045 * lvl)
                add("music_rev", s, tq, gain=0.03 * lvl)

# heartbeat under the orchestration
for k in range(4):
    add("drums", kick(.6, 42, 110, dec=6), 31.0 + k * .9, gain=.35)

# sidechain the music bus to the kicks
g = np.ones(N)
for tk in kick_times:
    i = int(tk * SR)
    t = np.arange(min(int(.45 * SR), N - i)) / SR
    g[i:i + len(t)] = np.minimum(g[i:i + len(t)], 1 - .5 * np.exp(-t * 9))
BUS["music"] *= g

# tape stop on the music + drums at 23.0
def tape_stop(bus, t0, length=.5):
    i0, n = int(t0 * SR), int(length * SR)
    seg = BUS[bus][:, i0:i0 + n].copy()
    u = np.arange(n) / n
    pos = np.cumsum((1 - u) ** 1.5) * 1.0
    pos = np.clip(pos, 0, n - 1)
    out = np.stack([np.interp(pos, np.arange(n), seg[c]) for c in range(2)]) * (1 - u) ** .5
    BUS[bus][:, i0:i0 + n] = out
    BUS[bus][:, i0 + n:int(26.2 * SR)] *= 0.0


for b in ("drums",):
    tape_stop(b, 23.0)
i0 = int(23.0 * SR)
fade = np.ones(N)
fade[i0:int(23.5 * SR)] = np.linspace(1, 0, int(23.5 * SR) - i0)
fade[int(23.5 * SR):int(26.2 * SR)] = 0
BUS["music"][:, :] *= fade
add("music", dr, 23.2, gain=0.16)  # drone survives the stop

# ------------------------------------------------------------------ SFX: hook
for k, ch in enumerate("Build with AI"):
    if ch != " ":
        add("sfx", click(.7 + .3 * rng.random(), 2000 + rng.random() * 600), 1.05 + k * .85 / 13, pan=-.2 + k * .03, gain=.45)
add("sfx", click(1.3, 1500), 2.0, gain=.7)
add("sfx", blip(600, 1800, .18), 2.05, gain=.18)
add("sfx", riser(.6, 300, 2600), 2.38, gain=.28)
add("sfx", impact(1.0), 2.98, gain=.75)
add("sfx_rev", impact(.6), 2.98, gain=.25)
add("sfx", whoosh(.55, 5000, 600, shape=.3), 2.82, pan=.4, gain=.35)
add("sfx", whoosh(.5, 3000, 9000, shape=.5), 5.56, pan=-.5, gain=.35)

# ------------------------------------------------------------------ SFX: products & models
KT = [6.3, 6.95, 7.6, 8.25]
for i, k in enumerate(KT):
    m = [74, 77, 81, 84][i]
    add("sfx", pluck(mtof(m), .4, 1.2), k + .04, pan=.3, gain=.16)
    add("sfx_rev", pluck(mtof(m), .4), k + .04, gain=.08)
    add("sfx", blip(mtof(m + 12), mtof(m + 19), .06), k + .9, pan=.45, gain=.10)
for k in range(10):
    add("sfx", click(.35, 2600), 8.45 + k * .145, pan=.5, gain=.25)
add("sfx", click(.8, 1800), 10.35, pan=.5, gain=.3)
add("sfx", whoosh(.6, 600, 3500, shape=.6), 10.85, gain=.22)
add("sfx", whoosh(.6, 400, 2400, shape=.5), 11.7, pan=.4, gain=.18)
add("sfx", click(1.0, 1700), 12.55, pan=.4, gain=.45)
add("sfx", click(.8, 2400), 12.72, pan=.4, gain=.35)
add("sfx", blip(900, 1500, .08), 12.62, pan=.4, gain=.15)
for k in (13.75, 14.75):
    add("sfx", click(.7, 2100), k, pan=.4, gain=.35)
    add("sfx", blip(mtof(88), mtof(91), .05), k + .02, pan=-.3, gain=.08)

# ------------------------------------------------------------------ SFX: wipes
for at in (16.7, 44.6, 55.6):
    pixel_rain(at - .42, .84, 40, gain=.9)
    add("sfx", whoosh(.84, 700, 7000, shape=.5), at - .42, gain=.25)
add("sfx", impact(.8), 16.72, gain=.35)
add("sfx", impact(.9), 44.62, gain=.4)

# ------------------------------------------------------------------ SFX: price
for i, d in enumerate([5, 0, 0]):
    st, du = 17.45 + i * .1, 1.15 + i * .1
    last = 0
    for f in range(int(du * 600)):
        t = st + f / 600
        u = f / (du * 600)
        k = int((1 - 2 ** (-10 * u)) * (20 + d))
        if k != last:
            add("sfx", click(.5, 2800 + i * 300), t, pan=-.2 + i * .2, gain=.22)
            last = k
add("sfx", pluck(mtof(81), .5), 18.62, gain=.12)
add("sfx", whoosh(.3, 6000, 800, shape=.25) * 1.2, 19.32, pan=-.4, gain=.45)
add("sfx", kick(.3, 60, 200, dec=12), 19.34, gain=.3)
add("sfx", whoosh(.6, 300, 2000, shape=.6), 19.7, pan=-.5, gain=.2)
add("sfx", riser(.35, 400, 3000), 19.9, gain=.22)
add("sfx", impact(1.1), 20.25, gain=.8)
add("sfx_rev", impact(.7), 20.25, gain=.3)
for i, m in enumerate(CH["F9"]):
    s = pluck(mtof(m + 12), .9, 1.0, dec=3.5)
    add("sfx", s, 20.25 + i * .012, pan=(i - 2) * .25, gain=.10)
    add("sfx_rev", s, 20.25, gain=.10)
for k in range(14):
    m = [84, 88, 91, 93, 96][k % 5]
    add("sfx", blip(mtof(m), mtof(m) * 1.1, .05), 20.3 + rng.random() * .35, pan=rng.uniform(-.7, .7), gain=.06)

# ------------------------------------------------------------------ SFX: "so far"
t = tt(2.6)
hpos = 1 - 2 ** (-10 * t / 2.5)
f = 220 + 1400 * (1 - hpos)
tone = np.sin(6.283 * np.cumsum(f) / SR) * (1 - hpos) ** .8 * (1 - np.exp(-t * 40))
add("sfx", lp(tone, 3000), 23.62, pan=0, gain=.12)
add("sfx", whoosh(1.6, 5000, 300, shape=.08), 23.62, gain=.4)
for k in range(3):
    add("sfx", click(.5, 1500), 25.4 + k * .385, gain=.18)

# ------------------------------------------------------------------ SFX: constellation
add("sfx_rev", bell(mtof(81)), 26.45, gain=.10)
star_notes = [69, 72, 76, 77, 81]
for i in range(5):
    add("sfx", bell(mtof(star_notes[i] + 12), 2.0), 27.95 + i * .17, pan=-.6 + i * .3, gain=.12)
    add("sfx_rev", bell(mtof(star_notes[i] + 12), 2.0), 27.95 + i * .17, gain=.12)
add("sfx", whoosh(1.0, 300, 4000, shape=.85), 28.5, gain=.2)
for i, m in enumerate(CH["Bbhi"]):
    s = bell(mtof(m), 3.0)
    add("sfx_rev", s, 29.4 + i * .1, pan=(i - 2) * .3, gain=.10)
add("sfx", whoosh(.9, 2500, 400, shape=.45), 30.95, gain=.22)
for i in range(5):
    add("sfx", blip(mtof(76 + i * 3), mtof(81 + i * 3), .07), 31.95 + i * .04, pan=-.6 + .3 * i, gain=.12)
for k in range(10):
    add("sfx", click(.25, 3000 + 300 * (k % 3)), 32.4 + k * .06, pan=rng.uniform(-.6, .6), gain=.18)
for i in range(5):
    add("sfx", blip(mtof(93 - i * 3), mtof(86 - i * 3), .07), 32.85 + (i % 3) * .08 + i * .01, pan=.6 - .3 * i, gain=.10)
RESULT = [0, 0, 1, 1, 1]
for i, ok in enumerate(RESULT):
    tm = 33.15 + [0, 1, 3, 4, 5][i] * .06
    if ok:
        add("sfx", pluck(mtof(88), .3, .6), tm, pan=-.5 + .25 * i, gain=.09)
    else:
        add("sfx", kick(.2, 90, 160, dec=20), tm, pan=-.5 + .25 * i, gain=.15)
for i, m in enumerate([77, 81, 84, 88]):
    add("sfx", bell(mtof(m + 12), 2.4), 33.55 + i * .05, pan=.4, gain=.10)
    add("sfx_rev", bell(mtof(m + 12), 2.4), 33.55 + i * .05, gain=.10)
add("sfx", whoosh(.7, 500, 3000, shape=.7), 34.0, gain=.22)

# ------------------------------------------------------------------ SFX: chart
add("sfx", whoosh(.8, 2000, 8000, shape=.5), 34.55, gain=.12)


def glide(t0, dur, m0, m1, gain, pan=0):
    t = tt(dur)
    f = mtof(m0) * (mtof(m1) / mtof(m0)) ** (t / dur)
    s = np.sin(6.283 * np.cumsum(f) / SR) * np.sin(np.pi * t / dur) ** .7
    add("sfx", s, t0, pan=pan, gain=gain)
    add("sfx_rev", s, t0, gain=gain * .7)


glide(35.25, .9, 64, 67, .035, -.4)
glide(35.75, 1.2, 62, 69, .05, -.2)
glide(37.15, 1.1, 69, 81, .07, .2)
for i, tm in enumerate([35.3, 35.45, 35.6, 35.65, 35.95]):
    add("sfx", blip(mtof(76), mtof(79), .04), tm, pan=-.4, gain=.05)
for i, tm in enumerate([35.8, 36.0, 36.2, 36.55, 36.85]):
    add("sfx", pluck(mtof([74, 76, 77, 79, 81][i]), .3), tm, pan=-.2, gain=.07)
for i, tm in enumerate([37.15, 37.4, 38.2]):
    add("sfx", bell(mtof([84, 88, 93][i]), 1.6), tm, pan=.2, gain=.10)
    add("sfx_rev", bell(mtof([84, 88, 93][i]), 1.6), tm, gain=.08)
add("sfx", riser(.65, 300, 3000), 38.55, gain=.22)
last = 0
for f in range(int(1.1 * 600)):
    tm = 38.65 + f / 600
    u = f / (1.1 * 600)
    k = round((1 - (1 - u) ** 3) * 15)
    if k != last:
        add("sfx", click(.6, 2600), tm, pan=.5, gain=.28)
        last = k
add("sfx", impact(.9), 39.2, gain=.6)
add("sfx_rev", impact(.5), 39.2, gain=.2)

# ------------------------------------------------------------------ SFX: router
LANE = [0, 1, 2, 4, 0, 3, 1]
for i, ln in enumerate(LANE):
    rl = 45.75 + i * .5
    add("sfx", whoosh(.42, 600, 2500, shape=.8), rl, pan=-.4, gain=.10)
    add("sfx", pluck(mtof(76), .25, .5), rl + .42, gain=.09)
    m = [72, 74, 77, 79, 84][ln] + 12
    add("sfx", blip(mtof(m), mtof(m) * 1.12, .06), rl + .5, pan=.3, gain=.10)
    add("sfx", click(.7, 2000), rl + .88, pan=.55, gain=.3)
    add("sfx", pluck(mtof(m), .35, .8), rl + .88, pan=.55, gain=.07)
add("sfx", whoosh(.6, 300, 1800, shape=.5), 49.95, gain=.16)

# ------------------------------------------------------------------ SFX: results
for i, tm in enumerate([51.0, 51.2, 51.4]):
    add("sfx", pluck(mtof([76, 79, 83][i]), .4), tm + .05, pan=-.5 + .5 * i, gain=.1)
for st, pan in ((51.9, 0), (52.05, .5)):
    last = 0
    for f in range(int(1.3 * 600)):
        tm = st + f / 600
        u = f / (1.3 * 600)
        e = 4 * u ** 3 if u < .5 else 1 - (-2 * u + 2) ** 3 / 2
        k = round(e * 40)
        if k != last and k % 2 == 0:
            add("sfx", click(.4, 2700), tm, pan=pan, gain=.18)
            last = k
add("sfx", bell(mtof(88), 1.5), 53.3, pan=.2, gain=.08)
add("sfx_rev", bell(mtof(88), 1.5), 53.3, gain=.08)

# ------------------------------------------------------------------ SFX: CTA
add("sfx", impact(.8), 55.62, gain=.35)
for k in range(9):
    if "go rogue."[k] != " ":
        add("sfx", click(1.0, 1600 + rng.random() * 500), 56.95 + k * .7 / 9, pan=-.1 + .03 * k, gain=.55)
        add("sfx", kick(.08, 120, 300, dec=50), 56.95 + k * .7 / 9, gain=.12)
add("sfx", whoosh(1.0, 400, 5000, shape=.95), 58.2, gain=.25)
add("sfx", impact(1.2), 59.15, gain=.85)
add("sfx_rev", impact(.8), 59.15, gain=.4)
for i, m in enumerate(CH["F9"] + [72, 76]):
    s = bell(mtof(m + 12), 3.5)
    add("sfx_rev", s, 59.15 + i * .03, pan=(i - 3) * .2, gain=.08)
for k in range(10):
    add("sfx", click(.8, 1800 + rng.random() * 500), 59.95 + k * .06, pan=.05 * k - .2, gain=.4)
add("sfx", pluck(mtof(84), .5, .7), 60.7, gain=.12)
add("sfx", bell(mtof(91), 2.0), 60.72, gain=.06)
add("sfx_rev", bell(mtof(91), 2.0), 60.72, gain=.08)

# ------------------------------------------------------------------ mix
def make_ir(sec=3.2, decay=2.4, seed=1):
    r = np.random.default_rng(seed)
    t = tt(sec)
    ir = np.stack([lp(r.standard_normal(len(t)), 5500) * np.exp(-t * decay) for _ in range(2)])
    pre = int(.025 * SR)
    ir = np.concatenate([np.zeros((2, pre)), ir], axis=1)
    return ir / np.sqrt((ir ** 2).sum() / 2)


IR = make_ir()
def reverb(x):
    return np.stack([fftconvolve(x[c], IR[c])[:N] for c in range(2)])


mix = (BUS["music"] * 1.0 + reverb(BUS["music_rev"]) * .55 + BUS["drums"] * .9
       + BUS["sfx"] * 1.0 + reverb(BUS["sfx_rev"]) * .6)
mix = np.stack([hp(mix[c], 28) for c in range(2)])
# loudness: normalise the busy middle, then soft-limit
ref = np.percentile(np.abs(mix), 99.5)
mix *= 0.55 / ref
mix = np.tanh(mix * 1.15) / 1.15
n0 = int(.02 * SR)
mix[:, :n0] *= np.linspace(0, 1, n0)
fs, fe = int(61.6 * SR), int(DUR * SR)
mix[:, fs:fe] *= np.linspace(1, 0, fe - fs) ** 1.5
mix[:, fe:] = 0
mix = mix[:, :int(DUR * SR)]
mix *= 0.89 / np.abs(mix).max()
wavfile.write("soundtrack.wav", SR, (mix.T * 32767).astype(np.int16))
rms = np.sqrt((mix ** 2).mean())
print("written soundtrack.wav", mix.shape, "rms dBFS %.1f" % (20 * np.log10(rms)))
