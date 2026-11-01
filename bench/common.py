"""Shared helpers for the benchmark scripts."""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ufccl.synth import blob_slices, lesion_phantom, noise_slices  # noqa: E402
from ufccl.volume import RawVolume  # noqa: E402

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
for d in (DATA, RESULTS, FIGS):
    os.makedirs(d, exist_ok=True)

BLOB = dict(cell=16, thr=0.60)
NOISE_P = 0.31          # close to the 6-connected site-percolation threshold (~0.3116)
LESION_DENSITY = 30     # lesions per 128^3 voxels


def slices_for(kind, shape, seed=0):
    if kind == "blobs":
        return blob_slices(shape, BLOB["cell"], BLOB["thr"], seed)
    if kind == "noise":
        return noise_slices(shape, NOISE_P, seed)
    if kind == "lesions":
        n = max(1, int(round(LESION_DENSITY * np.prod(shape) / 128 ** 3)))
        vol = lesion_phantom(shape, n_lesions=n, r_range=(3, 12), n_vessels=max(1, n // 10),
                             seed=seed)
        return iter(vol)
    raise ValueError(kind)


def volume_array(kind, shape, seed=0):
    return np.stack(list(slices_for(kind, shape, seed)))


def volume_file(kind, shape, seed=0):
    """Raw uint8 file for (kind, shape); created once and reused."""
    path = os.path.join(DATA, f"{kind}_{shape[0]}x{shape[1]}x{shape[2]}_s{seed}.u8")
    if not os.path.exists(path):
        vol = RawVolume.from_slices(path, slices_for(kind, shape, seed), shape[1:])
        vol.close()
    return path


def best_of(fn, repeats):
    """(best seconds, result of the last call)."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def hardware():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import numba
        import scipy
        import cc3d
        info.update(numpy=np.__version__, numba=numba.__version__, scipy=scipy.__version__,
                    cc3d=cc3d.__version__ if hasattr(cc3d, "__version__") else "4.x")
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name; "
             "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1); "
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfCores"],
            capture_output=True, text=True, timeout=30).stdout.split("\n")
        info.update(cpu=out[0].strip(), ram_gb=float(out[1]), cores=int(out[2]))
    except Exception:
        info["cpu"] = platform.processor()
    return info


def save_json(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
