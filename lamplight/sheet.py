"""Looking at a loop without playing it.

    python -m lamplight.sheet SCENE.py sheet T0 T1 N COLS [--scale 0.25] [--name NAME]
        N frames across [T0, T1), tiled, each labelled with its time. The whole loop at a glance.
    python -m lamplight.sheet SCENE.py strip T0 N [--fps 15] [--crop X0 Y0 X1 Y1] [--cols 3]
        N CONSECUTIVE frames at the delivery fps, optionally cropped. A contact sheet can't show a
        step that exists only between frames; a strip of neighbours can.
    python -m lamplight.sheet SCENE.py still T [T ...]
        Full-size stills.

Output goes to out/ beside the scene file. Open the images and look: judging by eye is the method.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import scene
from .post import to_srgb8


def _setup(path):
    mod = scene.load(path)
    out = os.path.join(os.path.dirname(os.path.abspath(path)), 'out')
    os.makedirs(out, exist_ok=True)
    return mod, mod.make(), out


def _label_font():
    try:
        return ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 13)
    except OSError:
        return ImageFont.load_default()


def sheet(path, t0, t1, n, cols, scale=0.25, name=None):
    mod, obj, out = _setup(path)
    ts = np.linspace(t0, t1, n, endpoint=False)
    w, h = int(mod.W * scale), int(mod.H * scale)
    rows = (n + cols - 1) // cols
    im_out = Image.new('RGB', (cols * (w + 6) + 6, rows * (h + 22) + 6), (30, 30, 34))
    d = ImageDraw.Draw(im_out)
    f = _label_font()
    for i, t in enumerate(ts):
        im = Image.fromarray(to_srgb8(obj.frame(t))).resize((w, h), Image.LANCZOS)
        x, y = 6 + (i % cols) * (w + 6), 6 + (i // cols) * (h + 22)
        im_out.paste(im, (x, y + 16))
        d.text((x, y), f't={t:6.2f}s', fill=(200, 200, 200), font=f)
    p = os.path.join(out, name or f'sheet_{t0:05.2f}_{t1:05.2f}.png')
    im_out.save(p)
    print(p)


def strip(path, t0, n, fps=15, crop=None, cols=3, name=None):
    mod, obj, out = _setup(path)
    f = _label_font()
    tiles = []
    for k in range(n):
        t = t0 + k / fps
        im = Image.fromarray(to_srgb8(obj.frame(t)))
        if crop:
            im = im.crop(tuple(crop))
        ImageDraw.Draw(im).text((6, 4), f'{t:.3f}', fill=(255, 120, 120), font=f)
        tiles.append(im)
    w, h = tiles[0].size
    rows = (n + cols - 1) // cols
    im_out = Image.new('RGB', (w * cols, h * rows))
    for i, im in enumerate(tiles):
        im_out.paste(im, ((i % cols) * w, (i // cols) * h))
    p = os.path.join(out, name or f'strip_{t0:05.2f}.png')
    im_out.save(p)
    print(p)


def stills(path, ts):
    mod, obj, out = _setup(path)
    for t in ts:
        p = os.path.join(out, f'still_{t:05.2f}.png')
        Image.fromarray(to_srgb8(obj.frame(t))).save(p)
        print(p)


def main():
    ap = argparse.ArgumentParser(prog='python -m lamplight.sheet')
    ap.add_argument('scene')
    sub = ap.add_subparsers(dest='what', required=True)
    s = sub.add_parser('sheet')
    s.add_argument('t0', type=float); s.add_argument('t1', type=float)
    s.add_argument('n', type=int); s.add_argument('cols', type=int)
    s.add_argument('--scale', type=float, default=0.25); s.add_argument('--name')
    s = sub.add_parser('strip')
    s.add_argument('t0', type=float); s.add_argument('n', type=int)
    s.add_argument('--fps', type=float, default=15); s.add_argument('--crop', type=int, nargs=4)
    s.add_argument('--cols', type=int, default=3); s.add_argument('--name')
    s = sub.add_parser('still')
    s.add_argument('t', type=float, nargs='+')
    a = ap.parse_args()
    if a.what == 'sheet':
        sheet(a.scene, a.t0, a.t1, a.n, a.cols, a.scale, a.name)
    elif a.what == 'strip':
        strip(a.scene, a.t0, a.n, a.fps, a.crop, a.cols, a.name)
    else:
        stills(a.scene, a.t)


if __name__ == '__main__':
    main()
