"""Tiny anti-aliased rasterizer: oriented rounded boxes (capsules are a special case)
composited over a float RGB buffer, with an optional additive halo.

A primitive row is:
  cx, cy, hw, hh, cos, sin, corner, r, g, b, a, glow_w, glow_a
All in screen pixels. hw/hh are half extents *including* the corner radius.
"""
import numpy as np
from numba import njit

NCOL = 13


@njit(cache=True, fastmath=True)
def _draw(buf, prims):
    H, W, _ = buf.shape
    for k in range(prims.shape[0]):
        cx = prims[k, 0]; cy = prims[k, 1]
        hw = prims[k, 2]; hh = prims[k, 3]
        c = prims[k, 4]; s = prims[k, 5]
        rad = prims[k, 6]
        cr = prims[k, 7]; cg = prims[k, 8]; cb = prims[k, 9]; ca = prims[k, 10]
        gw = prims[k, 11]; ga = prims[k, 12]
        if ca <= 0.0 and ga <= 0.0:
            continue
        ext = hw + hh + 1.5
        if ga > 0.0:
            ext += 2.5 * gw
        x0 = int(cx - ext); x1 = int(cx + ext) + 1
        y0 = int(cy - ext); y1 = int(cy + ext) + 1
        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if x1 > W: x1 = W
        if y1 > H: y1 = H
        bx = hw - rad; by = hh - rad
        for y in range(y0, y1):
            py = y + 0.5 - cy
            for x in range(x0, x1):
                px = x + 0.5 - cx
                lx = abs(c * px + s * py)
                ly = abs(-s * px + c * py)
                qx = lx - bx; qy = ly - by
                ox = qx if qx > 0.0 else 0.0
                oy = qy if qy > 0.0 else 0.0
                inside = qx if qx > qy else qy
                if inside > 0.0:
                    inside = 0.0
                d = (ox * ox + oy * oy) ** 0.5 + inside - rad
                if ga > 0.0 and d > 0.0:
                    g = ga * np.exp(-(d * d) / (gw * gw))
                    buf[y, x, 0] += cr * g
                    buf[y, x, 1] += cg * g
                    buf[y, x, 2] += cb * g
                cov = 0.5 - d
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                al = cov * ca
                buf[y, x, 0] = buf[y, x, 0] * (1.0 - al) + cr * al
                buf[y, x, 1] = buf[y, x, 1] * (1.0 - al) + cg * al
                buf[y, x, 2] = buf[y, x, 2] * (1.0 - al) + cb * al


class Batch:
    """Collects primitives in painter's order."""

    def __init__(self):
        self.rows = []

    def box(self, cx, cy, hw, hh, ang, corner, rgb, a, glow_w=0.0, glow_a=0.0):
        """Vectorised: every arg may be an array (broadcast). rgb is (...,3)."""
        cx, cy, hw, hh, ang, corner, a, glow_w, glow_a = np.broadcast_arrays(
            *[np.asarray(v, dtype=np.float64) for v in (cx, cy, hw, hh, ang, corner, a, glow_w, glow_a)])
        n = cx.size
        if n == 0:
            return
        rgb = np.broadcast_to(np.asarray(rgb, dtype=np.float64), cx.shape + (3,)).reshape(n, 3)
        hw = hw.ravel(); hh = hh.ravel(); corner = corner.ravel(); a = a.ravel()
        # sub-pixel thin shapes: fatten to 0.5px and pay for it in alpha
        thin = hh < 0.5
        a = np.where(thin, a * np.maximum(hh, 0.0) / 0.5, a)
        hh = np.where(thin, 0.5, hh)
        hw = np.maximum(hw, 0.5)
        corner = np.minimum(corner, np.minimum(hw, hh))
        th = np.asarray(ang, dtype=np.float64).ravel()
        row = np.empty((n, NCOL))
        row[:, 0] = cx.ravel(); row[:, 1] = cy.ravel()
        row[:, 2] = hw; row[:, 3] = hh
        row[:, 4] = np.cos(th); row[:, 5] = np.sin(th)
        row[:, 6] = corner
        row[:, 7:10] = rgb
        row[:, 10] = np.clip(a, 0, 1)
        row[:, 11] = np.maximum(glow_w.ravel(), 1e-3)
        row[:, 12] = glow_a.ravel()
        self.rows.append(row)

    def seg(self, x0, y0, x1, y1, r, rgb, a, glow_w=0.0, glow_a=0.0):
        """Capsule from (x0,y0) to (x1,y1) with radius r (vectorised)."""
        x0, y0, x1, y1, r = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in (x0, y0, x1, y1, r)])
        dx = x1 - x0; dy = y1 - y0
        L = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        self.box((x0 + x1) / 2, (y0 + y1) / 2, L / 2 + r, r, ang, r, rgb, a, glow_w, glow_a)

    def draw(self, buf):
        if not self.rows:
            return
        prims = np.ascontiguousarray(np.concatenate(self.rows, axis=0), dtype=np.float64)
        _draw(buf, prims)
        self.rows = []
