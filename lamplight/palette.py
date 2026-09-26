"""GIF palettes that keep what matters.

GIF allows 256 colours per frame. Median cut spends them where the pixels are, which in a dark
piece means on the dark: a small accent that carries meaning (who said something, say) gets
starved and drifts to a neighbouring colour. So the colours that carry meaning get reserved
slots, each with anti-aliasing ramps toward every ground it sits on, and median cut fills the rest.
Keep the reserve near half the palette: soft light needs adaptive colours too, or it bands.

Soft light (bloom, glow, gradients) bands in 256 colours no matter what. The fix is an ordered
dither, masked to where there is light: a fixed 8x8 pattern doesn't crawl between frames the way
error diffusion does, and leaving the dark ground exact keeps the file small, because flat
regions are what GIF compresses well.
"""
import numpy as np
from PIL import Image


def ramps(accents, grounds, fractions=(1.0, 0.55)):
    """Reserved colours: every accent at each fraction of the way over every ground."""
    keys = []
    for a in accents:
        for g in grounds:
            for f in fractions:
                keys.append(np.asarray(a) * f + np.asarray(g) * (1 - f))
    return np.unique((np.clip(np.array(keys), 0, 1) * 255 + 0.5).astype(np.uint8), axis=0)


def build(mosaic, colors=256, keys=None):
    """A palette image: median-cut colours from `mosaic` (a PIL image sampling the whole loop),
    plus the reserved keys."""
    keys = np.zeros((0, 3), np.uint8) if keys is None else keys
    n_adapt = max(16, colors - len(keys))
    adapt = mosaic.quantize(colors=n_adapt, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    ap = np.array(adapt.getpalette()[:3 * n_adapt], dtype=np.uint8).reshape(-1, 3)
    full = np.concatenate([ap, keys])[:256]
    pim = Image.new('P', (1, 1))
    pim.putpalette(full.ravel().tolist() + [0] * (768 - full.size))
    print('palette', len(ap), 'adaptive +', len(keys), 'key')
    return pim


_B8 = None


def bayer(im, amp, floor=30.0, span=14.0):
    """Ordered dither of amplitude `amp` (in 0-255 units), applied only where luminance rises
    above `floor` (fading in over `span`), so dark flat ground stays exact."""
    global _B8
    if amp <= 0:
        return im
    if _B8 is None:
        b = np.array([[0, 2], [3, 1]])
        for _ in range(2):
            b = np.block([[4 * b, 4 * b + 2], [4 * b + 3, 4 * b + 1]])
        _B8 = (b + 0.5) / 64.0 - 0.5
    a = np.asarray(im, dtype=np.float32)
    h, w = a.shape[:2]
    t = np.tile(_B8, (h // 8 + 1, w // 8 + 1))[:h, :w, None]
    lum = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    m = np.clip((lum - floor) / span, 0, 1)[..., None]
    return Image.fromarray(np.clip(a + t * amp * m, 0, 255).astype(np.uint8))
