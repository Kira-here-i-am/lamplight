"""lamplight: small tools for looping animations made as pure functions of time, with light as
the medium. See README.md."""
from .raster import Batch
from .ease import smooth, ease, ramp, mix, level
from .post import blur, bloom, to_srgb8
from .marks import cursive, polyline, soft_box, Type

__version__ = '0.1.0'
