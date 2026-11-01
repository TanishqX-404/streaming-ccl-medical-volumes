"""In-memory two-pass 3-D connected component labeling with Union-Find.

Pass 1 assigns provisional labels in raster order (z, then y, then x) and
records equivalences with Union-Find.  Pass 2 resolves every provisional label
to its root, numbers roots consecutively in order of first appearance, and
relabels the volume in place.  Optional per-component statistics are fused
into the pass-2 sweep so no extra read of the volume is needed.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from .stats import ComponentTable
from .unionfind import find, stat_add, stats_new, union


def causal_offsets(connectivity):
    """Neighbour offsets (dz, dy, dx) already visited in raster order."""
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18 or 26")
    max_nonzero = {6: 1, 18: 2, 26: 3}[connectivity]
    offs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nz = (dz != 0) + (dy != 0) + (dx != 0)
                if 0 < nz <= max_nonzero and (dz, dy, dx) < (0, 0, 0):
                    offs.append((dz, dy, dx))
    return np.array(offs, dtype=np.int64)


@njit(cache=True)
def count_run_starts(m2):
    """Foreground runs along the last axis of a 2-D view.

    Every provisional label begins at a run start, so this is a tight upper
    bound on the number of labels (about N/2 only for a checkerboard).
    """
    c = 0
    for r in range(m2.shape[0]):
        prev = False
        for x in range(m2.shape[1]):
            v = m2[r, x] != 0
            if v and not prev:
                c += 1
            prev = v
    return c


@njit(cache=True)
def _pass1_6(mask, lab, parent, size):
    Z, Y, X = mask.shape
    nid = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if mask[z, y, x] == 0:
                    continue
                a = lab[z - 1, y, x] if z > 0 else 0
                b = lab[z, y - 1, x] if y > 0 else 0
                c = lab[z, y, x - 1] if x > 0 else 0
                if a == 0 and b == 0 and c == 0:
                    nid += 1
                    lab[z, y, x] = nid
                    continue
                m = a
                if b != 0 and (m == 0 or b < m):
                    m = b
                if c != 0 and (m == 0 or c < m):
                    m = c
                lab[z, y, x] = m
                if a != 0 and a != m:
                    union(parent, size, m, a)
                if b != 0 and b != m:
                    union(parent, size, m, b)
                if c != 0 and c != m:
                    union(parent, size, m, c)
    return nid


@njit(cache=True)
def _pass1_generic(mask, offs, lab, parent, size):
    Z, Y, X = mask.shape
    K = offs.shape[0]
    nid = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if mask[z, y, x] == 0:
                    continue
                m = 0
                for k in range(K):
                    zz = z + offs[k, 0]
                    yy = y + offs[k, 1]
                    xx = x + offs[k, 2]
                    if zz < 0 or yy < 0 or yy >= Y or xx < 0 or xx >= X:
                        continue
                    v = lab[zz, yy, xx]
                    if v != 0 and (m == 0 or v < m):
                        m = v
                if m == 0:
                    nid += 1
                    lab[z, y, x] = nid
                    continue
                lab[z, y, x] = m
                for k in range(K):
                    zz = z + offs[k, 0]
                    yy = y + offs[k, 1]
                    xx = x + offs[k, 2]
                    if zz < 0 or yy < 0 or yy >= Y or xx < 0 or xx >= X:
                        continue
                    v = lab[zz, yy, xx]
                    if v != 0 and v != m:
                        union(parent, size, m, v)
    return nid


@njit(cache=True)
def resolve(parent, n_prov):
    """Map provisional ids 1..n_prov to consecutive final ids.

    Final ids follow the order of first appearance in raster order because
    provisional ids are created in that order.  ``root_final`` and ``final``
    are separate arrays so a root's slot is never overwritten while other
    provisional ids still refer to it.
    """
    final = np.zeros(n_prov + 1, dtype=parent.dtype)
    root_final = np.zeros(n_prov + 1, dtype=parent.dtype)
    n = 0
    for i in range(1, n_prov + 1):
        r = find(parent, i)
        if root_final[r] == 0:
            n += 1
            root_final[r] = n
        final[i] = root_final[r]
    return final, n


@njit(cache=True)
def relabel(lab, final):
    Z, Y, X = lab.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                v = lab[z, y, x]
                if v != 0:
                    lab[z, y, x] = final[v]


@njit(cache=True)
def relabel_stats(lab, final, S, oz, oy, ox, fY, fX):
    """Relabel in place and accumulate statistics in the same sweep.

    (oz, oy, ox) is the position of this block inside a larger volume of
    in-plane size (fY, fX); it makes bounding boxes and raster indices global.
    """
    Z, Y, X = lab.shape
    for z in range(Z):
        gz = z + oz
        for y in range(Y):
            gy = y + oy
            for x in range(X):
                v = lab[z, y, x]
                if v != 0:
                    f = final[v]
                    lab[z, y, x] = f
                    gx = x + ox
                    stat_add(S, f, gz, gy, gx, (gz * fY + gy) * fX + gx)


@dataclass
class LabelResult:
    labels: np.ndarray
    n: int
    table: ComponentTable = None


def as_mask(mask):
    """C-contiguous bool/uint8 3-D array (other dtypes are binarised)."""
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError("mask must be 3-D (Z, Y, X)")
    if mask.dtype != np.bool_ and mask.dtype != np.uint8:
        mask = mask != 0
    return np.ascontiguousarray(mask)


def label(mask, connectivity=6, stats=False, spacing=(1.0, 1.0, 1.0),
          origin=(0, 0, 0), full_shape=None):
    """Label a 3-D binary mask.

    Parameters
    ----------
    connectivity : 6, 18 or 26.
    stats : also return a :class:`ComponentTable` (voxel count, centroid,
        bounding box, physical volume), computed in the relabel sweep.
    origin, full_shape : position of ``mask`` inside a larger volume; only
        affects the statistics (used by the tiled variant).
    """
    mask = as_mask(mask)
    Z, Y, X = mask.shape
    if mask.size == 0:
        return LabelResult(np.zeros(mask.shape, np.int32), 0,
                           ComponentTable(np.zeros((0, 11), np.int64), spacing) if stats else None)
    runs = int(count_run_starts(mask.reshape(-1, X)))
    cap = runs + 2
    dt = np.int32 if cap < 2**31 - 1 else np.int64
    lab = np.zeros(mask.shape, dtype=dt)
    parent = np.arange(cap, dtype=dt)
    size = np.ones(cap, dtype=dt)
    if connectivity == 6:
        n_prov = _pass1_6(mask, lab, parent, size)
    else:
        n_prov = _pass1_generic(mask, causal_offsets(connectivity), lab, parent, size)
    final, n = resolve(parent, n_prov)
    del parent, size
    table = None
    if stats:
        fz, fy, fx = full_shape if full_shape is not None else mask.shape
        S = stats_new(n + 1)
        relabel_stats(lab, final, S, origin[0], origin[1], origin[2], fy, fx)
        table = ComponentTable(S[1:], spacing)
    else:
        relabel(lab, final)
    return LabelResult(lab, int(n), table)
