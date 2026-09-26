"""Render a scene's loop once, then encode it.

    python -m lamplight.encode SCENE.py frames [--master-fps 30]
    python -m lamplight.encode SCENE.py mp4  [--fps 30] [--crf 18]
    python -m lamplight.encode SCENE.py gif  [--fps 15] --width W [--colors 256] [--dither 6]
    python -m lamplight.encode SCENE.py webp [--fps 15] --width W [--quality 80]
    python -m lamplight.encode SCENE.py clip --from T0 --to T1 [--hold 1.0] [--fps 30] [--crf 18]

Frames go to frames/ and encodes to out/, beside the scene file. Frames are rendered once at the
master size and fps; a lower fps takes every k-th frame, so every format shows the same instants.
GIF and WebP frame durations are quantised to their formats' units (GIF counts centiseconds) so
the loop still sums to exactly T: 15 fps is 70/60/70 ms, not 67 ms truncated to 60.

`clip` is for a video that plays once. A loop's seam is invisible only while it loops; played once,
the cut point starts mid-thought and ends mid-motion. A clip renders [T0, T1) of the scene straight
through (ignoring START), holds its last frame for --hold seconds so it comes to rest rather than
cutting, and writes out/NAME_clip.mp4 from frames_clip/.
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from . import palette, scene
from .post import to_srgb8

_worker = {}


def _init(path, master_fps, sub='frames', times=None):
    mod = scene.load(path)
    _worker.update(mod=mod, obj=mod.make(), fps=master_fps, dir=os.path.join(os.path.dirname(path), sub),
                   times=times)


def _render(i):
    mod, obj = _worker['mod'], _worker['obj']
    if _worker['times'] is not None:
        t = _worker['times'][i]
    else:
        n = round(mod.T * _worker['fps'])
        t = (scene.start_of(mod) + i * mod.T / n) % mod.T
    Image.fromarray(to_srgb8(obj.frame(t))).save(os.path.join(_worker['dir'], f'{i:04d}.png'), compress_level=1)
    return i


def render_frames(path, master_fps=30):
    path = os.path.abspath(path)
    mod = scene.load(path)
    fr = os.path.join(os.path.dirname(path), 'frames')
    os.makedirs(fr, exist_ok=True)
    for f in glob.glob(os.path.join(fr, '*.png')):
        os.remove(f)
    n = round(mod.T * master_fps)
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2), initializer=_init,
                             initargs=(path, master_fps)) as ex:
        for k, _ in enumerate(ex.map(_render, range(n), chunksize=8)):
            if k % 120 == 0:
                print('frame', k, '/', n, flush=True)
    print('rendered', n)


def clip(path, t0, t1, hold=1.0, fps=30, crf=18):
    """A one-shot video of [t0, t1), coming to rest on its last frame."""
    import av
    path = os.path.abspath(path)
    fr = os.path.join(os.path.dirname(path), 'frames_clip')
    os.makedirs(fr, exist_ok=True)
    for f in glob.glob(os.path.join(fr, '*.png')):
        os.remove(f)
    n = int(round((t1 - t0) * fps))
    times = [t0 + i / fps for i in range(n)]
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2), initializer=_init,
                             initargs=(path, fps, 'frames_clip', times)) as ex:
        for k, _ in enumerate(ex.map(_render, range(n), chunksize=8)):
            if k % 120 == 0:
                print('frame', k, '/', n, flush=True)
    files = sorted(glob.glob(os.path.join(fr, '*.png')))
    files += [files[-1]] * int(round(hold * fps))
    out = _out(path, '_clip.mp4')
    with av.open(out, 'w') as c:
        st = c.add_stream('libx264', rate=fps)
        im0 = Image.open(files[0])
        st.width, st.height = im0.size
        st.pix_fmt = 'yuv420p'
        st.options = {'crf': str(crf), 'preset': 'slow', 'tune': 'animation'}
        for f in files:
            for p in st.encode(av.VideoFrame.from_image(Image.open(f).convert('RGB'))):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    print(out, os.path.getsize(out) // 1024, 'KB', len(files), 'frames', f'{len(files) / fps:.2f} s')


def frames_at(path, fps, master_fps=30):
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(path)), 'frames', '*.png')))
    step = master_fps / fps
    assert abs(step - round(step)) < 1e-9, 'fps must divide the master fps'
    return files[::int(round(step))]


def durations(n, fps, unit):
    """Per-frame durations in ms, quantised to the format's unit, summing to exactly n / fps s."""
    ends = np.round(np.arange(1, n + 1) * 1000.0 / fps / unit) * unit
    return np.diff(np.concatenate([[0], ends])).astype(int).tolist()


def load_scaled(files, width):
    ims = []
    for f in files:
        im = Image.open(f).convert('RGB')
        if width != im.width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        ims.append(im)
    return ims


def _out(path, suffix):
    mod = scene.load(path)
    d = os.path.join(os.path.dirname(os.path.abspath(path)), 'out')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, scene.name_of(mod) + suffix)


def mp4(path, fps=30, crf=18, master_fps=30):
    import av
    files = frames_at(path, fps, master_fps)
    out = _out(path, '.mp4')
    with av.open(out, 'w') as c:
        st = c.add_stream('libx264', rate=fps)
        im0 = Image.open(files[0])
        st.width, st.height = im0.size
        st.pix_fmt = 'yuv420p'
        st.options = {'crf': str(crf), 'preset': 'slow', 'tune': 'animation'}
        for f in files:
            for p in st.encode(av.VideoFrame.from_image(Image.open(f).convert('RGB'))):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    print(out, os.path.getsize(out) // 1024, 'KB', len(files), 'frames')


def gif(path, fps, width, colors=256, dither=6.0, master_fps=30):
    mod = scene.load(path)
    ims = load_scaled(frames_at(path, fps, master_fps), width)
    # one palette for the whole loop, from a spread of frames, so nothing shimmers between frames
    sample = [np.asarray(ims[i]) for i in np.linspace(0, len(ims) - 1, 24).astype(int)]
    keys = mod.key_colors() if hasattr(mod, 'key_colors') else None
    pal = palette.build(Image.fromarray(np.concatenate(sample, axis=0)), colors, keys)
    q = [palette.bayer(im, dither).quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    out = _out(path, f'_{width}.gif')
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations(len(q), fps, 10), loop=0,
              optimize=False, disposal=1)
    print(out, os.path.getsize(out) // 1024, 'KB', len(q), 'frames')


def webp(path, fps, width, quality=80, master_fps=30):
    ims = load_scaled(frames_at(path, fps, master_fps), width)
    out = _out(path, f'_{width}.webp')
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=durations(len(ims), fps, 1), loop=0,
                quality=quality, method=6)
    print(out, os.path.getsize(out) // 1024, 'KB', len(ims), 'frames')


def main():
    ap = argparse.ArgumentParser(prog='python -m lamplight.encode')
    ap.add_argument('scene')
    ap.add_argument('what', choices=['frames', 'mp4', 'gif', 'webp', 'clip'])
    ap.add_argument('--master-fps', type=int, default=30)
    ap.add_argument('--fps', type=int)
    ap.add_argument('--width', type=int)
    ap.add_argument('--crf', type=int, default=18)
    ap.add_argument('--colors', type=int, default=256)
    ap.add_argument('--dither', type=float, default=6.0)
    ap.add_argument('--quality', type=int, default=80)
    ap.add_argument('--from', dest='t0', type=float)
    ap.add_argument('--to', dest='t1', type=float)
    ap.add_argument('--hold', type=float, default=1.0)
    a = ap.parse_args()
    if a.what == 'frames':
        render_frames(a.scene, a.master_fps)
    elif a.what == 'mp4':
        mp4(a.scene, a.fps or a.master_fps, a.crf, a.master_fps)
    elif a.what == 'gif':
        gif(a.scene, a.fps or 15, a.width, a.colors, a.dither, a.master_fps)
    elif a.what == 'webp':
        webp(a.scene, a.fps or 15, a.width, a.quality, a.master_fps)
    elif a.what == 'clip':
        clip(a.scene, a.t0, a.t1, a.hold, a.fps or 30, a.crf)


if __name__ == '__main__':
    main()
