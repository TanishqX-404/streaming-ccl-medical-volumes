"""Tiled out-of-core labeling with seam merge.

1. Each tile is read, labeled independently with the in-memory algorithm and
   written back with *global provisional ids* (tile base offset + local id).
   Per-tile statistics rows are kept, indexed by global provisional id.
2. Seam merge: for each tile, read the tile plus a one-voxel halo.  Every
   boundary voxel of the tile is checked against its causal neighbours that
   lie outside the tile (faces, edges and corners, according to the chosen
   connectivity); label pairs are merged in a global Union-Find.  Each adjacent
   voxel pair is examined exactly once, from its later voxel's tile.
3. Roots are numbered by first raster index (the same order as the in-memory
   labeler), statistics are folded into roots, and the tiles are rewritten
   with final ids.
"""
import numpy as np
from numba import njit

from .label import label as label_block
from .label import causal_offsets
from .stats import ComponentTable
from .unionfind import C_FIRST, NSTAT, find, fold_into_roots, union
from .volume import ArrayVolume


@njit(cache=True)
def _offset_labels(lab, base):
    Z, Y, X = lab.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                v = lab[z, y, x]
                if v != 0:
                    lab[z, y, x] = v + base


@njit(cache=True)
def _seam_tile(H, tz0, tz1, ty0, ty1, tx0, tx1, offs, parent, size):
    """Merge labels across the boundary of the tile occupying
    [tz0:tz1, ty0:ty1, tx0:tx1] of the halo block ``H``."""
    Hz, Hy, Hx = H.shape
    K = offs.shape[0]
    n = 0
    for z in range(tz0, tz1):
        zface = z == tz0 or z == tz1 - 1
        for y in range(ty0, ty1):
            yface = zface or y == ty0 or y == ty1 - 1
            x = tx0
            while x < tx1:
                v = H[z, y, x]
                if v != 0:
                    for k in range(K):
                        zz = z + offs[k, 0]
                        yy = y + offs[k, 1]
                        xx = x + offs[k, 2]
                        if zz < 0 or yy < 0 or xx < 0 or yy >= Hy or xx >= Hx:
                            continue
                        if zz >= tz0 and yy >= ty0 and yy < ty1 and xx >= tx0 and xx < tx1:
                            continue
                        w = H[zz, yy, xx]
                        if w != 0:
                            union(parent, size, v, w)
                            n += 1
                if yface or x == tx1 - 1:
                    x += 1
                else:
                    x = tx1 - 1  # interior row: only the two end voxels
    return n


@njit(cache=True)
def _final_map(parent, n_total, roots_sorted, nroots):
    """final[i] for every provisional id, numbering roots in the given order."""
    root_final = np.zeros(n_total + 1, np.int32)
    for j in range(nroots):
        root_final[roots_sorted[j]] = j + 1
    final = np.zeros(n_total + 1, np.int32)
    for i in range(1, n_total + 1):
        final[i] = root_final[find(parent, i)]
    return final


def _tile_ranges(n, t):
    return [(a, min(a + t, n)) for a in range(0, n, t)]


def label_tiled(src, tile=(128, 128, 128), connectivity=6, out=None,
                stats=True, spacing=(1.0, 1.0, 1.0), return_info=False):
    """Label a volume tile by tile.

    ``src`` is an ndarray or an object with ``shape`` and ``read_block``;
    ``out`` is an ndarray or a volume (e.g. :class:`RawVolume`, int32) the
    labels are written to; by default a RAM array is allocated and returned.
    Returns ``(out, n_components, table)``; ``table`` is ``None`` when
    ``stats=False``.
    """
    if isinstance(src, np.ndarray):
        src = ArrayVolume(src)
    Z, Y, X = src.shape
    if isinstance(out, np.ndarray):
        out = ArrayVolume(out)
    owns_out = out is None
    if out is None:
        out = ArrayVolume(np.zeros((Z, Y, X), np.int32))
    tz, ty, tx = (min(int(t), s) if s else 1 for t, s in zip(tile, (Z, Y, X)))
    zr, yr, xr = _tile_ranges(Z, tz), _tile_ranges(Y, ty), _tile_ranges(X, tx)
    offs = causal_offsets(connectivity)
    limit = np.iinfo(out.dtype).max

    # 1. per-tile labeling
    base = 0
    stat_blocks = []
    bases = {}
    for (z0, z1) in zr:
        for (y0, y1) in yr:
            for (x0, x1) in xr:
                blk = src.read_block(z0, z1, y0, y1, x0, x1)
                res = label_block(blk, connectivity, stats=stats, spacing=spacing,
                                  origin=(z0, y0, x0), full_shape=(Z, Y, X))
                if base + res.n > limit:
                    raise OverflowError("label dtype too small for provisional ids")
                lab = res.labels.astype(out.dtype, copy=False)
                if base:
                    _offset_labels(lab, base)
                out.write_block(z0, y0, x0, lab)
                if stats:
                    stat_blocks.append(res.table.data)
                bases[(z0, y0, x0)] = base
                base += res.n
    n_total = base
    pdt = np.int32 if n_total < 2**31 - 1 else np.int64
    parent = np.arange(n_total + 1, dtype=pdt)
    size = np.ones(n_total + 1, dtype=pdt)

    # 2. seam merge on tile + halo blocks
    n_pairs = 0
    for (z0, z1) in zr:
        for (y0, y1) in yr:
            for (x0, x1) in xr:
                hz0, hy0, hx0 = max(z0 - 1, 0), max(y0 - 1, 0), max(x0 - 1, 0)
                hy1, hx1 = min(y1 + 1, Y), min(x1 + 1, X)
                H = out.read_block(hz0, z1, hy0, hy1, hx0, hx1)
                n_pairs += _seam_tile(H, z0 - hz0, z1 - hz0, y0 - hy0, y1 - hy0,
                                      x0 - hx0, x1 - hx0, offs, parent, size)

    # 3. resolve roots, fold stats, number by first raster index
    table = None
    if stats:
        S = np.zeros((n_total + 1, NSTAT), np.int64)
        if stat_blocks:
            S[1:] = np.vstack(stat_blocks)
        fold_into_roots(parent, S, n_total)
        is_root = parent[1:] == np.arange(1, n_total + 1, dtype=pdt)
        roots = np.nonzero(is_root)[0] + 1
        order = np.argsort(S[roots, C_FIRST], kind="stable")
        roots_sorted = roots[order].astype(np.int64)
        table = ComponentTable(S[roots_sorted], spacing)
    else:
        # without stats, number roots by smallest provisional id, which is
        # the order of first appearance within the global tile order only;
        # canonicalise afterwards if exact raster order is required
        is_root = parent[1:] == np.arange(1, n_total + 1, dtype=pdt)
        roots_sorted = (np.nonzero(is_root)[0] + 1).astype(np.int64)
    n = len(roots_sorted)
    final = _final_map(parent, n_total, roots_sorted, n) if n_total else np.zeros(1, np.int32)
    del parent, size
    for (z0, z1) in zr:
        for (y0, y1) in yr:
            for (x0, x1) in xr:
                blk = out.read_block(z0, z1, y0, y1, x0, x1)
                out.write_block(z0, y0, x0, final[blk].astype(out.dtype, copy=False))
    result = out.arr if (owns_out and hasattr(out, "arr")) else out
    if return_info:
        return result, n, table, {"tiles": len(zr) * len(yr) * len(xr),
                                  "provisional": n_total, "seam_unions": n_pairs}
    return result, n, table
