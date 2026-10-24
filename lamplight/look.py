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
    kymo(frames, path)      brightness along a path against time, one row a frame: whatever travels
                            along the path draws slanted lines, so direction and speed show at a
                            glance (down and to the RIGHT: moving toward the path's end).
    steps(frames)           the change into each frame, largest first, with WHERE it sits: a spike
                            is judged by where it is, not only by how big.

Each picture returns an RGB uint8 array; save it and look. steps returns a list.
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


def _lum(f):
    a = np.asarray(f, np.float32)
    if a.dtype != np.uint8 and a.max() <= 1.5:
        a = a * 255
    return a[..., :3] @ np.array([0.3, 0.59, 0.11], np.float32) if a.ndim == 3 else a


def kymo(frames, path, width=1):
    """frames: images in order (uint8 RGB, or float 0..1). path: [(x, y), ...] pixel points in order
    along something that may carry motion (a stream's centreline, a road, a branch). Each row is one
    frame's brightness along the path, averaged over `width` pixels across x. Built to answer "which
    way does this water flow" when frame differences and global optical flow were confounded by
    reflections sliding across it; the slant answered at once."""
    pts = np.asarray(path, int)
    h = width // 2
    rows = []
    for f in frames:
        L = _lum(f)
        rows.append([L[y, max(0, x - h):x + h + 1].mean() for x, y in pts])
    K = np.asarray(rows, np.float32)
    K = (K - K.min()) / (K.max() - K.min() + 1e-9)
    g = (K * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def steps(frames, thresh=20, top=8):
    """The mean |change| (0..255) into each frame, with how many pixels changed by more than `thresh`
    and the centre of where they are. Returns the `top` largest as [(i, mean, n_big, (y, x) or None)],
    plus the median of all steps: [(...), ...], median. Run it on the decoded video AND on the rendered
    frames: a step in the file but not the frames is the encoder's; in both, it's the scene's."""
    out, prev = [], None
    for i, f in enumerate(frames):
        L = _lum(f)
        if prev is not None:
            d = np.abs(L - prev)
            big = d > thresh
            ys, xs = np.nonzero(big)
            out.append((i, float(d.mean()), int(big.sum()), (int(ys.mean()), int(xs.mean())) if ys.size else None))
        prev = L
    med = float(np.median([o[1] for o in out])) if out else 0.0
    return sorted(out, key=lambda o: -o[1])[:top], med


def save(arr, path, crop=None, scale=1):
    im = Image.fromarray(arr)
    if crop is not None:
        im = im.crop(crop)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
    return path
