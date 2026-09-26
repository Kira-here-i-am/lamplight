"""Whole-frame passes on a float RGB buffer (H, W, 3), values nominally in [0, 1]."""
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])


def blur(img, r):
    """Three box passes, about a gaussian of sigma ~ r. Separable, via cumulative sums, so the
    cost doesn't grow with r."""
    out = img
    for _ in range(3):
        for ax in (0, 1):
            n = out.shape[ax]
            cs = np.cumsum(out, axis=ax, dtype=np.float32)
            pad = [(0, 0)] * out.ndim
            pad[ax] = (1, 0)
            cs = np.pad(cs, pad)
            idx = np.arange(n)
            lo = np.clip(idx - r, 0, n)
            hi = np.clip(idx + r + 1, 0, n)
            a = np.take(cs, hi, axis=ax) - np.take(cs, lo, axis=ax)
            shape = [1] * out.ndim
            shape[ax] = n
            out = a / (hi - lo).reshape(shape)
    return out


def bloom(buf, threshold=0.34, strength=0.85, small=3, large=6):
    """Light that spills. Pixels above `threshold` luminance are blurred at two scales (on half-
    and quarter-resolution copies, for speed) and added back. One pass for the whole frame,
    rather than halos on every mark: halos on thousands of short segments stack and blow out."""
    lum = buf @ LUMA
    k = np.clip((lum - threshold) / (1 - threshold), 0, 1)[..., None]
    bright = (buf * k).astype(np.float32)
    s = blur(bright[::2, ::2], small)
    l = blur(bright[::4, ::4], large)
    s = np.repeat(np.repeat(s, 2, 0), 2, 1)[:buf.shape[0], :buf.shape[1]]
    l = np.repeat(np.repeat(l, 4, 0), 4, 1)[:buf.shape[0], :buf.shape[1]]
    # upsampled blocks are smoothed once more at full size so nothing steps
    glow = blur(0.55 * s + 0.45 * l, 2)
    return buf + strength * glow


def to_srgb8(buf):
    """Float buffer -> uint8 image array (values are taken as already display-encoded)."""
    return (np.clip(buf, 0, 1) * 255 + 0.5).astype(np.uint8)
