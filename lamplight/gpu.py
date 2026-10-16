"""The GPU seam: rasterize triangles into an antialiased coverage mask on the graphics card.

The CPU path (PIL polygons drawn at 3x and filtered down) costs about half a second for a frame
full of fine branches; this draws the same shapes multisampled on the GPU and reads the mask back
in a few milliseconds, so a scene can rebuild moving geometry every frame and a whole clip can be
looked at again within a tick.

Fine strokes in motion are where rasterizers fail: a twig a quarter of a pixel wide, sampled four
times a pixel, lands on 0 or 1 sample and twinkles as it moves. So the mask is supersampled
(`scale`, default 2: sixteen samples a pixel with 4x MSAA), and `strip` gives hairlines a minimum
drawn width with their coverage carried as alpha (a 0.25 px twig is drawn 0.5 px wide at half
strength), which keeps the ink the same and the motion steady. Overlapping strokes take the MAX of
their coverage, not the sum, so a fork is never darker than the limbs that meet in it.

Being a good GPU neighbour, learned in anastomosis (its nice.py, issue #40): Windows preempts GPU
work between submissions, and a loop that never waits piles submissions into the queue until the
desktop compositor's small jobs wait behind it. So every draw here is DRAINED before it returns
(the readback already waits; the poll makes sure) and then YIELDS a few milliseconds, a window in
which this process holds nothing in the queue. The yield rides an injectable `sleep`. A process
that opens a Raster should end when its work is done.

wgpu-py 0.32: the public `on_submitted_work_done_sync` is broken (a callback-signature mismatch
against its own wgpu-native), so the drain uses the private `_poll_wait`, with the public call as
the fallback for a later wgpu-py.
"""
import time

import numpy as np

YIELD_SECONDS = 0.003

_SHADER = """
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) a: f32,
};
@vertex
fn vs(@location(0) p: vec2<f32>, @location(1) a: f32) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(p, 0.0, 1.0);
    o.a = a;
    return o;
}
@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(i.a, i.a, i.a, 1.0);
}
"""


def drain(device):
    poll_wait = getattr(device, "_poll_wait", None)
    if poll_wait is not None:
        poll_wait()
    else:  # a future wgpu-py without the private poll
        device.queue.on_submitted_work_done_sync()


class Raster:
    """Coverage masks of triangle soups, W x H, multisampled (4x) and supersampled (`scale`).

        r = Raster(W, H)
        tris, alpha = strip(pts, widths, min_width=0.5)
        mask = r.mask(tris, alpha)   # tris (N, 3, 2) in output pixels, y down; alpha (N, 3) or None
    """

    def __init__(self, width, height, samples=4, scale=2, sleep=time.sleep, yield_seconds=YIELD_SECONDS):
        import wgpu
        self.wgpu = wgpu
        self.W, self.H, self.samples, self.scale = int(width), int(height), samples, int(scale)
        self.sleep, self.yield_seconds = sleep, yield_seconds
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.adapter_info = dict(adapter.info)
        self.device = adapter.request_device_sync()
        fmt = wgpu.TextureFormat.rgba8unorm
        module = self.device.create_shader_module(code=_SHADER)
        mx = {"operation": wgpu.BlendOperation.max, "src_factor": wgpu.BlendFactor.one,
              "dst_factor": wgpu.BlendFactor.one}
        self.pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs", "buffers": [{
                "array_stride": 12, "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [{"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 0},
                               {"format": wgpu.VertexFormat.float32, "offset": 8, "shader_location": 1}]}]},
            fragment={"module": module, "entry_point": "fs",
                      "targets": [{"format": fmt, "blend": {"color": mx, "alpha": mx}}]},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            multisample={"count": samples},
        )
        self.w, self.h = self.W * self.scale, self.H * self.scale
        size = (self.w, self.h, 1)
        self.msaa = self.device.create_texture(size=size, format=fmt, sample_count=samples,
                                               usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        self.resolved = self.device.create_texture(size=size, format=fmt,
                                                   usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self.msaa_view, self.resolved_view = self.msaa.create_view(), self.resolved.create_view()

    def mask(self, tris, alpha=None):
        """Coverage in [0, 1], float32 (H, W), of the given triangles (output pixels, y down)."""
        wgpu = self.wgpu
        tris = np.asarray(tris, np.float32).reshape(-1, 2)
        a = np.ones(len(tris), np.float32) if alpha is None else np.asarray(alpha, np.float32).reshape(-1)
        v = np.empty((len(tris), 3), np.float32)
        v[:, 0] = tris[:, 0] / self.W * 2.0 - 1.0
        v[:, 1] = 1.0 - tris[:, 1] / self.H * 2.0
        v[:, 2] = a
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{
            "view": self.msaa_view, "resolve_target": self.resolved_view,
            "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.discard,
            "clear_value": (0.0, 0.0, 0.0, 0.0)}])
        if len(v):
            vb = self.device.create_buffer_with_data(data=v.tobytes(), usage=wgpu.BufferUsage.VERTEX)
            rp.set_pipeline(self.pipeline)
            rp.set_vertex_buffer(0, vb)
            rp.draw(len(v))
        rp.end()
        self.device.queue.submit([enc.finish()])
        data = self.device.queue.read_texture({"texture": self.resolved, "origin": (0, 0, 0)},
                                              {"bytes_per_row": self.w * 4, "rows_per_image": self.h},
                                              (self.w, self.h, 1))
        drain(self.device)
        if self.yield_seconds > 0:
            self.sleep(self.yield_seconds)
        m = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 4)[..., 0].astype(np.float32) / 255.0
        s = self.scale
        return m if s == 1 else m.reshape(self.H, s, self.W, s).mean(axis=(1, 3))


def strip(pts, widths, min_width=0.0):
    """Triangles, and their vertices' coverage, for a tapered stroke along a polyline.

    pts (n, 2), widths (n,) full widths, in output pixels. A width under `min_width` is drawn at
    min_width with its coverage scaled down to match, so hairlines keep their ink without twinkling.
    Returns (tris (2(n-1), 3, 2), alpha (2(n-1), 3))."""
    pts = np.asarray(pts, np.float32)
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    w = np.asarray(widths, np.float32)
    drawn = np.maximum(w, min_width) if min_width > 0 else w
    cov = np.minimum(1.0, w / np.maximum(drawn, 1e-9))
    hw = (drawn / 2.0)[:, None]
    L, R = pts + n * hw, pts - n * hw
    a, b, c, e = L[:-1], R[:-1], L[1:], R[1:]
    ca, cc = cov[:-1], cov[1:]
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([b, e, c], 1)], 0)
    alpha = np.concatenate([np.stack([ca, ca, cc], 1), np.stack([ca, cc, cc], 1)], 0)
    return tris, alpha


def strips(P, widths, counts, min_width=0.0):
    """Many tapered strokes at once. P (M, 2) holds the polylines end to end, widths (M,) a full
    width per point, counts the number of points in each polyline. Returns (tris, alpha) like
    strip(), for all of them, built without a Python loop over the strokes."""
    P = np.asarray(P, np.float32)
    w = np.asarray(widths, np.float32)
    counts = np.asarray(counts, np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    g = np.arange(len(P))
    k = g - np.repeat(starts, counts)
    n = np.repeat(counts, counts)
    d = P[np.where(k < n - 1, g + 1, g)] - P[np.where(k > 0, g - 1, g)]
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    drawn = np.maximum(w, min_width) if min_width > 0 else w
    cov = np.minimum(1.0, w / np.maximum(drawn, 1e-9))
    hw = (drawn / 2.0)[:, None]
    L, R = P + nrm * hw, P - nrm * hw
    s = g[k < n - 1]
    a, b, c, e = L[s], R[s], L[s + 1], R[s + 1]
    ca, cc = cov[s], cov[s + 1]
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([b, e, c], 1)], 0)
    alpha = np.concatenate([np.stack([ca, ca, cc], 1), np.stack([ca, cc, cc], 1)], 0)
    return tris, alpha


def strip_triangles(pts, widths):
    """Triangles for a tapered stroke along a polyline: pts (n, 2), widths (n,) full widths."""
    return strip(pts, widths)[0]
