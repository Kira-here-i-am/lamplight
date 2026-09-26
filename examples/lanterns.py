"""A small example scene: five lines of asemic handwriting come alight one after another on a
dark page, hold, and go dark again, in a 6-second seamless loop.

    python -m lamplight.sheet examples/lanterns.py sheet 0 6 12 4
    python -m lamplight.encode examples/lanterns.py frames
    python -m lamplight.encode examples/lanterns.py mp4
    python -m lamplight.encode examples/lanterns.py gif --width 480
"""
import numpy as np

from lamplight import Batch, bloom, cursive, level, polyline, ramp, soft_box

W, H = 480, 270
T = 6.0
START = 3.0          # begin on the frame with every line lit, the best still
NAME = 'lanterns'

GROUND = np.array([0.043, 0.050, 0.072])
SLAB = np.array([0.092, 0.103, 0.140])
INK_DARK = np.array([0.160, 0.177, 0.230])
EMBER = np.array([0.470, 0.405, 0.345])
GLOW = np.array([0.985, 0.925, 0.810])
THREAD = np.array([0.290, 0.680, 0.505])


class Lanterns:
    def __init__(self):
        rng = np.random.default_rng(5)
        self.lines = []
        for i in range(5):
            v = 70 + i * 32
            u, words = 130, []
            end = 130 + rng.uniform(180, 230)
            while u < end:
                w = rng.uniform(14, 42)
                words.append((u, min(u + w, end), v))
                u += w + 9
            # cursive() works in any units; here it is pixels, so scale the hand up
            self.lines.append(cursive(100 + i, words, xh=7.0, letter=6.0))

    def frame(self, t):
        t = t % T
        buf = np.empty((H, W, 3))
        buf[:] = GROUND
        B = Batch()
        soft_box(B, W / 2, H / 2, 170, 115, 0.0, 6, SLAB, 1.0, 14)
        for i, seg in enumerate(self.lines):
            # each line lights in turn and all go dark together: every ramp returns to 0 by T
            on = ramp(t, 0.4 + 0.45 * i, 0.9 + 0.45 * i) * (1 - ramp(t, 4.8, 5.8))
            col = level(INK_DARK, EMBER, GLOW, on)
            B.seg(seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3], 1.1, col, 1.0)
            y = 70 + i * 32 - 4
            x = np.array([112.0, 112.0])
            polyline(B, x, np.array([y - 8, y + 4.0]), 1.4, level(SLAB, THREAD * 0.6, THREAD, on), 1.0)
        B.draw(buf)
        return bloom(buf)


def make():
    return Lanterns()
