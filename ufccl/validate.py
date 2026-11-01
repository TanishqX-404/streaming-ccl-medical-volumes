"""Partition-equivalence helpers for comparing labelings."""
import numpy as np
from numba import njit


@njit(cache=True)
def _canon_flat(flat, maxlab):
    mp = np.zeros(maxlab + 1, np.int64)
    out = np.zeros(flat.shape[0], np.int64)
    n = 0
    for i in range(flat.shape[0]):
        v = flat[i]
        if v != 0:
            if mp[v] == 0:
                n += 1
                mp[v] = n
            out[i] = mp[v]
    return out, n


def canonicalize(lab):
    """Relabel by first appearance in raster order (background stays 0)."""
    lab = np.ascontiguousarray(lab)
    flat = lab.reshape(-1)
    if flat.size == 0:
        return lab.astype(np.int64), 0
    if flat.min() < 0:
        raise ValueError("negative label")
    out, n = _canon_flat(flat, int(flat.max()))
    return out.reshape(lab.shape), int(n)


def same_partition(a, b):
    """True iff both labelings induce the same partition of the foreground."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if not np.array_equal(a != 0, b != 0):
        return False
    ca, na = canonicalize(a)
    cb, nb = canonicalize(b)
    return na == nb and bool(np.array_equal(ca, cb))


def check_invariants(mask, lab, n):
    """Section 6.5 invariants; raises AssertionError on violation."""
    mask = np.asarray(mask) != 0
    lab = np.asarray(lab)
    assert np.all(lab[mask] != 0), "foreground voxel without label"
    assert np.all(lab[~mask] == 0), "background voxel with label"
    ids = np.unique(lab[mask])
    assert len(ids) == n, f"{len(ids)} distinct labels but n={n}"
    if n:
        assert ids[0] == 1 and ids[-1] == n, "labels are not consecutive 1..n"
