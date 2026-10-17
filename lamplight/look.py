"""Instruments for judging motion, built for how I see: as pictures of what moved, not as playback.

A strip of frames shows where things are; these show how they move.

    plot(series, t0, t1)    values over time as lines: a stop is a flat step, a reversal a zero
                            crossing, a jerk a spike in speed. Constant speed reads mechanical;
                            ease reads as intent; a body never quite stops.
    onion(masks)            many instants overlaid, early in cold blue and late in warm amber, each
                            pixel keeping its brightest: what never moved comes out WHITE (every
                            colour at once), what moved comes out as a coloured smear along its
                            path. A limb in wind smears along its own arc; a zoom smears outward
                            in concentric copies.
    energy(masks)           how much each pixel changed, summed over consecutive frames: the places
                            that moved, lit; the places that held, black.

Each returns an RGB uint8 array; save it and look.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size=14):
    for name in ('consola.ttf', 'DejaVuSansMono.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def plot(series, t0, t1, size=(1400, 380), marks=(), title='', zero=True, ylim=None):
    """series: [(t, y, (r, g, b), label), ...] with t, y arrays. marks: [(t, label), ...] drawn as
    thin vertical lines. Seconds are gridded; the y range fits the data unless ylim is given."""
    W, H = size
    L, R, T, B = 70, 20, 50, 40
    img = Image.new('RGB', size, (14, 16, 22))
    dr = ImageDraw.Draw(img)
    f = _font(14)
    ys = np.concatenate([np.asarray(s[1], float) for s in series])
    lo, hi = (float(ys.min()), float(ys.max())) if ylim is None else ylim
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.06 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad

    def X(t):
        return L + (np.asarray(t, float) - t0) / (t1 - t0) * (W - L - R)

    def Y(y):
        return T + (hi - np.asarray(y, float)) / (hi - lo) * (H - T - B)

    for s in range(int(np.ceil(t0)), int(np.floor(t1)) + 1):
        x = float(X(s))
        dr.line([x, T, x, H - B], fill=(34, 38, 48) if s % 5 else (52, 58, 72))
        if s % 5 == 0:
            dr.text((x - 6, H - B + 6), f'{s}', fill=(150, 156, 170), font=f)
    if zero and lo < 0 < hi:
        dr.line([L, float(Y(0)), W - R, float(Y(0))], fill=(90, 96, 110))
    for v in np.linspace(lo + pad, hi - pad, 5):
        dr.text((6, float(Y(v)) - 8), f'{v:7.1f}', fill=(120, 126, 140), font=f)
    for t, label in marks:
        x = float(X(t))
        dr.line([x, T, x, H - B], fill=(120, 70, 70))
        dr.text((x + 3, T + 2), label, fill=(200, 140, 140), font=f)
    for k, (t, y, col, label) in enumerate(series):
        pts = list(zip(X(t).tolist(), Y(y).tolist()))
        dr.line(pts, fill=col, width=2)
        dr.text((L + 10 + (W - L - R) * k / max(1, len(series)), 6), label, fill=col, font=f)
    if title:                      # its own line under the legend, so the two never collide
        dr.text((W - R - 8 * len(title), 26), title, fill=(200, 200, 210), font=f)
    return np.asarray(img)


def ramp(n):
    """n colours from cold blue (first) to warm amber (last), each bright enough to add up toward white."""
    a = np.linspace(0, 1, n)[:, None]
    cold, warm = np.array([0.20, 0.45, 1.00]), np.array([1.00, 0.62, 0.15])
    mid = np.array([0.55, 1.00, 0.55])
    c = np.where(a < 0.5, cold + (mid - cold) * (a / 0.5), mid + (warm - mid) * ((a - 0.5) / 0.5))
    return c


def onion(masks):
    """Masks (each H x W in [0, 1]) overlaid, coloured by time; per channel the max, so a pixel covered
    at every instant is white and a pixel covered only sometimes keeps the colour of when."""
    cols = ramp(len(masks))
    out = np.zeros(masks[0].shape + (3,), np.float32)
    for m, c in zip(masks, cols):
        out = np.maximum(out, m[..., None] * c)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def energy(masks, gain=4.0):
    """Sum of |change| between consecutive masks, as a heat picture (black: never changed)."""
    e = np.zeros(masks[0].shape, np.float32)
    for a, b in zip(masks[:-1], masks[1:]):
        e += np.abs(b - a)
    e = np.clip(e * gain / max(1, len(masks) - 1), 0, 1)
    heat = np.stack([np.clip(3 * e, 0, 1), np.clip(3 * e - 1, 0, 1), np.clip(3 * e - 2, 0, 1)], -1)
    return (heat * 255).astype(np.uint8), e


def save(arr, path, crop=None, scale=1):
    im = Image.fromarray(arr)
    if crop is not None:
        im = im.crop(crop)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
    return path
