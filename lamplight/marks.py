"""Marks: handwriting, strokes, soft-edged shapes, text. Geometry in, primitives out."""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cursive(seed, words, xh=0.0125, letter=0.0105, slant=0.24, per=14):
    """Asemic handwriting: every word a looping pen line that looks written and can't be read.

    words: iterable of (u0, u1, v): a word's left and right ends and its baseline, in any units.
    xh is the x-height and `letter` a letter's width, in the same units.
    Returns segments (ua, va, ub, vb, word_index), ready to transform and draw as capsules.

    Each letter is one turn of a trochoid; a letter's loop depth and height are drawn at random,
    so some letters loop and some don't, with occasional ascenders and descenders. Seeded, so a
    page always writes the same hand."""
    rng = np.random.default_rng(seed)
    segs = []
    for wi, (u0, u1, v) in enumerate(words):
        width = u1 - u0
        n = max(2, int(round(width / letter)))
        R = width / (2 * math.pi * n)
        th = np.linspace(0, 2 * math.pi * n, n * per + 1)
        k = np.minimum((th / (2 * math.pi)).astype(int), n - 1)
        d = R * rng.uniform(0.35, 1.75, n)
        kind = rng.random(n)
        h = np.where(kind < 0.13, 2.4, np.where(kind < 0.20, -1.7, rng.uniform(0.8, 1.1, n))) * xh
        h[0] = xh
        x = u0 + R * th - d[k] * np.sin(th)
        y = v - h[k] * (1 - np.cos(th)) / 2
        x = x + slant * (v - y)            # the slant of a hand
        y = y + 0.0012 * np.sin(th * 0.37 + rng.uniform(0, 6))
        segs.append(np.stack([x[:-1], y[:-1], x[1:], y[1:], np.full(len(x) - 1, wi)], axis=1))
    if not segs:
        return np.zeros((0, 5))
    return np.concatenate(segs)


def polyline(B, x, y, th, rgb, a):
    """A stroke through screen points (x, y) with half-width th, drawn as flat-ended pieces so a
    translucent line doesn't bead: round caps would double the alpha at every joint. Only the two
    ends are rounded."""
    n = len(x) - 1
    mx, my = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    L = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])
    ang = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
    ends = np.zeros(n)
    ends[0] = ends[-1] = 1
    B.box(mx, my, L / 2 + th * ends, th, ang, th * ends, rgb, a)


def soft_box(B, cx, cy, hw, hh, rot, corner, rgb, a, feather):
    """A box whose edge fades over `feather` px, built from nested boxes whose alphas compose to
    `a` at the centre. An opaque target is capped just under 1, or every layer would be opaque
    and the edge wouldn't fade at all."""
    if feather < 1.0:
        B.box(cx, cy, hw, hh, rot, corner, rgb, a)
        return
    n = int(min(10, max(4, feather / 2)))
    al = 1 - (1 - min(a, 0.985)) ** (1.0 / n)
    for k in range(n):
        d = feather * (k + 0.5) / n
        B.box(cx, cy, hw - d, hh - d, rot, max(corner, feather - d), rgb, al)


class Type:
    """Text composited straight onto a float buffer. Fonts are cached per size."""

    def __init__(self, font_dir='C:/Windows/Fonts'):
        self.font_dir = font_dir
        self._fonts = {}

    def font(self, size, file):
        key = (int(round(size)), file)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(os.path.join(self.font_dir, file), key[0])
        return self._fonts[key]

    def draw(self, buf, x, y, s, size, rgb, a, anchor='mm', file='georgiai.ttf'):
        if a <= 0.004 or size < 4:
            return
        H, W = buf.shape[:2]
        f = self.font(size, file)
        l, tp, r, b = f.getbbox(s, anchor=anchor)
        w, h = r - l + 4, b - tp + 4
        im = Image.new('L', (w, h), 0)
        ImageDraw.Draw(im).text((-l + 2, -tp + 2), s, font=f, fill=255, anchor=anchor)
        m = np.asarray(im, dtype=np.float64) / 255.0 * a
        x0, y0 = int(round(x + l - 2)), int(round(y + tp - 2))
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + w, W), min(y0 + h, H)
        if X1 <= X0 or Y1 <= Y0:
            return
        mm = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
        buf[Y0:Y1, X0:X1] = buf[Y0:Y1, X0:X1] * (1 - mm) + np.asarray(rgb) * mm
