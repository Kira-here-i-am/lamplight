"""Easing and blending. Everything takes scalars or arrays."""
import numpy as np


def smooth(x):
    """Smoothstep on [0, 1], clamped."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease(x):
    """Cubic ease-in-out on [0, 1], clamped."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)


def ramp(t, a, b):
    """0 before a, 1 after b, smoothstep between: the workhorse of a timeline."""
    return smooth((t - a) / (b - a))


def mix(a, b, f):
    """Blend colours a -> b by f; f may be an array (broadcast over the colour axis)."""
    f = np.asarray(f, dtype=np.float64)[..., None]
    return a * (1 - f) + b * f


def level(dark, dim, full, l, low=0.3):
    """Three stops of light: 0 -> dark, `low` -> dim, 1 -> full. For things that are unlit,
    lit low, or lit, where 'low' is its own colour and not just a fade of 'full'."""
    l = np.asarray(l, dtype=np.float64)
    lo = np.clip(l / low, 0, 1)[..., None]
    hi = np.clip((l - low) / (1 - low), 0, 1)[..., None]
    a = dark * (1 - lo) + dim * lo
    return a * (1 - hi) + full * hi
