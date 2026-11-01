"""Child process for peak-memory measurement.

Usage: mem_probe.py VARIANT PATH Z Y X CONN
Prints READY after imports and JIT warm-up, waits for a GO line on stdin, runs
VARIANT on the raw uint8 volume at PATH and prints one JSON line.  The parent
(run_bench.py) samples this process's working set while it runs.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc3d  # noqa: E402
import scipy.ndimage as ndi  # noqa: E402

from ufccl import label, label_tiled, stream_components  # noqa: E402
from ufccl.volume import RawVolume  # noqa: E402

STRUCT = {6: 1, 18: 2, 26: 3}


def run(variant, path, shape, conn):
    Z, Y, X = shape
    if variant in ("ufccl", "ufccl+stats", "scipy", "cc3d"):
        m = np.fromfile(path, np.uint8).reshape(shape).view(np.bool_)
        if variant == "ufccl":
            return label(m, conn).n
        if variant == "ufccl+stats":
            return label(m, conn, stats=True).n
        if variant == "scipy":
            return ndi.label(m, ndi.generate_binary_structure(3, STRUCT[conn]))[1]
        return cc3d.connected_components(m, connectivity=conn, return_N=True)[1]
    if variant == "stream":
        vol = RawVolume(path, shape)
        n = len(stream_components(vol.slices(), conn))
        vol.close()
        return n
    if variant == "tiled":
        src = RawVolume(path, shape)
        with tempfile.TemporaryDirectory() as d:
            out = RawVolume.create(os.path.join(d, "out.i32"), shape, np.int32)
            _, n, _ = label_tiled(src, (128, 128, 128), conn, out=out)
            out.close()
        src.close()
        return n
    raise ValueError(variant)


def main():
    variant, path, Z, Y, X, conn = sys.argv[1:7]
    shape, conn = (int(Z), int(Y), int(X)), int(conn)
    # warm-up on a tiny volume so JIT compilation and imports are excluded
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.u8")
        (np.random.default_rng(0).random((8, 8, 8)) < 0.4).astype(np.uint8).tofile(p)
        run(variant, p, (8, 8, 8), conn)
    print("READY", flush=True)
    sys.stdin.readline()
    t = time.perf_counter()
    n = run(variant, path, shape, conn)
    dt = time.perf_counter() - t
    print(json.dumps({"n": int(n), "seconds": dt}), flush=True)


if __name__ == "__main__":
    main()
