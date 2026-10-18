"""Sound synthesis with NumPy only (no pygame, no audio files): square / triangle / noise voices, short retro effects
and two looping chiptunes. Every buffer is float64 in [-1, 1], mono, starts and ends at exactly 0 and peaks below 1.

build_effects(sr, noise_seed) -> {name: float array}; build_music(name, sr, noise_seed) -> float array that loops
sample-exactly (whole number of steps, each note gated to zero at its end).
"""
import numpy as np

EFFECT_SECONDS = {          # exact length of every effect, in seconds (buffer length = round(seconds * sr))
    "menu_move": 0.06, "menu_confirm": 0.16, "attack": 0.14, "block": 0.14, "parry": 0.30, "dodge": 0.20,
    "hit": 0.16, "ko": 0.70, "round_start": 0.25, "count": 0.12, "fight": 0.35, "win": 1.00, "lose": 1.00,
}
EFFECT_PEAK = 0.8           # effects are normalised to this peak (music is quieter: see ui.yaml audio.music_ratio)
MUSIC_PEAK = 0.7
FADE = 0.004                # seconds: linear fade-in / fade-out at the ends of every note and buffer


def _n(seconds, sr):
    return int(round(seconds * sr))


def midi_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def osc(kind, freq, n, sr, duty=0.5, rng=None, hold=1):
    """One voice. `freq` is a scalar or a per-sample array (Hz). kind: square | triangle | noise (sample-and-hold)."""
    if kind == "noise":
        k = -(-n // hold)
        return np.repeat(rng.uniform(-1.0, 1.0, k), hold)[:n]
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,))
    ph = (np.cumsum(f) / sr) % 1.0
    if kind == "square":
        return np.where(ph < duty, 1.0, -1.0)
    return 4.0 * np.abs(ph - 0.5) - 1.0          # triangle


def sweep(f0, f1, n):
    """Exponential frequency glide from f0 to f1 over n samples."""
    return f0 * (f1 / f0) ** np.linspace(0.0, 1.0, n)


def env(n, sr, decay, attack=0.003):
    """Linear attack, exponential decay (time constant `decay` s), linear fade to 0 over the last FADE seconds."""
    t = np.arange(n) / sr
    e = np.exp(-t / decay) * np.minimum(t / attack, 1.0)
    k = min(n, max(_n(FADE, sr), 1))
    e[n - k:] *= np.linspace(1.0, 0.0, k)
    return e


def bump(n):
    """Smooth 0 -> 1 -> 0 swell (for whooshes)."""
    return np.sin(np.pi * np.arange(n) / max(n - 1, 1)) ** 2


def _fit(x, n):
    out = np.zeros(n)
    m = min(n, len(x))
    out[:m] = x[:m]
    return out


def _norm(x, peak):
    x = np.array(x, dtype=float)
    x[0] = x[-1] = 0.0
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def seq(notes, sr, kind="square", duty=0.5, decay=0.1):
    """Concatenate (hz | None, seconds) notes, each with its own envelope."""
    parts = []
    for hz, sec in notes:
        n = _n(sec, sr)
        parts.append(np.zeros(n) if hz is None else osc(kind, hz, n, sr, duty) * env(n, sr, decay))
    return np.concatenate(parts)


def _effect(name, sr, rng):
    n = _n(EFFECT_SECONDS[name], sr)
    sq = lambda f, d=0.5: osc("square", f, n, sr, d)
    tri = lambda f: osc("triangle", f, n, sr)
    nz = lambda hold=1: osc("noise", 0, n, sr, rng=rng, hold=hold)
    if name == "menu_move":
        return sq(660.0) * env(n, sr, 0.03)
    if name == "menu_confirm":
        return _fit(seq([(660.0, 0.07), (990.0, 0.09)], sr, "square", 0.25, 0.06), n)
    if name == "attack":
        return 0.7 * nz(2) * env(n, sr, 0.05, 0.02) + 0.4 * sq(sweep(900.0, 250.0, n), 0.25) * env(n, sr, 0.06)
    if name == "block":
        return (0.5 * sq(1180.0) + 0.4 * sq(1810.0, 0.25) + 0.5 * nz(1) * env(n, sr, 0.01)) * env(n, sr, 0.05, 0.001)
    if name == "parry":
        return (tri(1760.0) + 0.5 * tri(2640.0) + 0.3 * sq(3520.0, 0.125)) * env(n, sr, 0.11, 0.001)
    if name == "dodge":
        return (0.6 * nz(4) + 0.5 * tri(sweep(300.0, 1000.0, n))) * bump(n)
    if name == "hit":
        return tri(sweep(170.0, 45.0, n)) * env(n, sr, 0.07, 0.001) + 0.6 * nz(6) * env(n, sr, 0.02, 0.001)
    if name == "ko":
        return (sq(sweep(520.0, 55.0, n), 0.25) * env(n, sr, 0.35) + 0.4 * nz(8) * env(n, sr, 0.2, 0.01))
    if name == "round_start":
        return (tri(sweep(180.0, 540.0, n)) + 0.3 * nz(10)) * bump(n)
    if name == "count":
        return sq(440.0) * env(n, sr, 0.08)
    if name == "fight":
        return (sq(880.0) + 0.5 * tri(1760.0)) * env(n, sr, 0.25)
    if name == "win":
        c = [523.25, 659.25, 783.99, 1046.5]
        lead = seq([(c[0], 0.12), (c[1], 0.12), (c[2], 0.12), (c[3], 0.64)], sr, "square", 0.25, 0.3)
        low = seq([(c[0] / 2, 0.12), (c[1] / 2, 0.12), (c[2] / 2, 0.12), (c[3] / 2, 0.64)], sr, "triangle", decay=0.4)
        return _fit(lead + 0.8 * low, n)
    if name == "lose":
        c = [392.0, 329.63, 261.63]
        lead = seq([(c[0], 0.2), (c[1], 0.2), (c[2], 0.6)], sr, "square", 0.5, 0.35)
        low = seq([(c[0] / 2, 0.2), (c[1] / 2, 0.2), (c[2] / 2, 0.6)], sr, "triangle", decay=0.4)
        return _fit(0.7 * lead + low, n)
    raise KeyError(name)


def build_effects(sr, noise_seed=0):
    rng = np.random.default_rng(noise_seed)
    return {name: _norm(_effect(name, sr, rng), EFFECT_PEAK) for name in EFFECT_SECONDS}


# ---------------------------------------------------------------------------------------------- music
# One step = an eighth note; a voice is (kind, duty, gain, per-step MIDI notes); None = rest, "k" / "h" = kick / hat.
def _bars(*bars):
    return [s for b in bars for s in b]


_N = None
MUSIC = {
    "menu": {   # C major, calm. 4 bars of 8 steps
        "bpm": 96,
        "lead": ("square", 0.25, 0.28, _bars(
            [76, _N, 72, _N, 79, _N, 76, 72], [76, _N, 69, _N, 72, _N, 76, _N],
            [77, _N, 72, _N, 77, _N, 81, 77], [79, _N, 74, 76, 79, _N, _N, _N])),
        "bass": ("triangle", 0.5, 0.5, _bars(
            [48, _N, 48, _N, 55, _N, 48, _N], [45, _N, 45, _N, 52, _N, 45, _N],
            [41, _N, 41, _N, 48, _N, 41, _N], [43, _N, 43, _N, 50, _N, 43, _N])),
    },
    "fight": {  # A minor, driving. 4 bars of 8 steps
        "bpm": 144,
        "lead": ("square", 0.125, 0.2, _bars(
            [69, _N, 72, 76, 72, _N, 69, 72], [65, _N, 69, 72, 69, _N, 65, 69],
            [67, _N, 71, 74, 71, _N, 67, 71], [64, 68, 71, 76, 71, 68, 71, 68])),
        "bass": ("triangle", 0.5, 0.7, _bars(
            [45, 45, 57, 45, 45, 45, 57, 45], [41, 41, 53, 41, 41, 41, 53, 41],
            [43, 43, 55, 43, 43, 43, 55, 43], [40, 40, 52, 40, 40, 40, 52, 40])),
        "drums": (None, None, 0.25, _bars(
            ["k", _N, "h", _N, "k", _N, "h", "h"], ["k", _N, "h", _N, "k", _N, "h", "h"],
            ["k", _N, "h", _N, "k", _N, "h", "h"], ["k", _N, "h", "k", "k", _N, "h", "h"])),
    },
}


def _voice(spec, step_n, sr, rng):
    kind, duty, gain, steps = spec
    out = np.zeros(step_n * len(steps))
    for i, m in enumerate(steps):
        if m is None:
            continue
        seg = slice(i * step_n, (i + 1) * step_n)
        if kind is None:                                   # drums
            if m == "k":
                out[seg] = osc("triangle", sweep(160.0, 45.0, step_n), step_n, sr) * env(step_n, sr, 0.06, 0.001)
            else:
                out[seg] = 0.5 * osc("noise", 0, step_n, sr, rng=rng, hold=2) * env(step_n, sr, 0.02, 0.001)
        else:
            out[seg] = osc(kind, midi_hz(m), step_n, sr, duty) * env(step_n, sr, step_n / sr * 1.2)
    return out * gain


def build_music(name, sr, noise_seed=0):
    spec = MUSIC[name]
    step_n = _n(60.0 / spec["bpm"] / 2.0, sr)
    rng = np.random.default_rng(noise_seed)
    mix = sum(_voice(v, step_n, sr, rng) for k, v in spec.items() if k != "bpm")
    return _norm(mix, MUSIC_PEAK)


def to_int16(x):
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
