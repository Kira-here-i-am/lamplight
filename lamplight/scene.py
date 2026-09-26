"""The scene contract, and loading a scene from a file.

A scene is a Python file that defines:

    W, H        frame size in pixels
    T           loop length in seconds; frame(t) must equal frame(t + T), so the loop is seamless
    make()      returns an object with frame(t) -> float RGB array (H, W, 3), values in [0, 1]

and optionally:

    START       where the rendered loop begins (default 0). A seamless loop can start anywhere,
                so choose the instant that works best as a still: some mail clients and previews
                show only the first frame.
    NAME        prefix for output files (default: the file's name)
    key_colors()  uint8 (N, 3) colours the GIF palette must keep exactly (see palette.ramps)

Every frame must be a pure function of t, with seeded randomness only, so frames can be
rendered in any order, in parallel, and any instant can be re-rendered to look at.
"""
import importlib.util
import os
import sys

_cache = {}


def load(path):
    path = os.path.abspath(path)
    if path not in _cache:
        d = os.path.dirname(path)
        if d not in sys.path:
            sys.path.insert(0, d)
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _cache[path] = mod
    return _cache[path]


def name_of(mod):
    return getattr(mod, 'NAME', mod.__name__)


def start_of(mod):
    return float(getattr(mod, 'START', 0.0))
