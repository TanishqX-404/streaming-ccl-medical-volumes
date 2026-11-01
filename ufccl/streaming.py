"""Slice-streaming labeling with early finalisation of components.

Only the previous and current slice of provisional labels are kept.  Labels
1..K are the components that touch the previous slice (the *frontier*); labels
K+1.. are created for the current slice.  Union-Find runs over this small id
space and each id owns a statistics row.  After a slice is scanned:

* every non-root row is folded into its root's row (additive / min / max),
* a root that no voxel of the current slice carries has reference count zero:
  the component is closed, its row is emitted and its slot is freed,
* surviving roots are renumbered 1..K' (an order-preserving in-place
  compaction) and the current labels are remapped, becoming the next frontier.

Peak memory is O(Y*X + open components), independent of Z.  The per-slice
recount of frontier voxels per root replaces an incrementally maintained
reference count; it costs one O(Y*X) sweep that the remap needs anyway.
"""
import itertools

import numpy as np
from numba import njit

from .label import causal_offsets, count_run_starts
from .stats import ComponentTable
from .unionfind import (NSTAT, find, fold_into_roots, stat_add, stat_reset,
                        union)


@njit(cache=True)
def _scan_slice(sl, z, prev, cur, K, parent, size, S, hasv, offs):
    Y, X = sl.shape
    nk = offs.shape[0]
    for i in range(1, K + 1):
        parent[i] = i
        size[i] = 1
        hasv[i] = False
    M = K
    zbase = np.int64(z) * Y * X
    for y in range(Y):
        for x in range(X):
            if sl[y, x] == 0:
                cur[y, x] = 0
                continue
            m = 0
            for k in range(nk):
                yy = y + offs[k, 1]
                xx = x + offs[k, 2]
                if yy < 0 or yy >= Y or xx < 0 or xx >= X:
                    continue
                if offs[k, 0] < 0:
                    v = prev[yy, xx]
                else:
                    v = cur[yy, xx]
                if v != 0 and (m == 0 or v < m):
                    m = v
            if m == 0:
                M += 1
                parent[M] = M
                size[M] = 1
                hasv[M] = False
                stat_reset(S, M)
                m = M
            else:
                for k in range(nk):
                    yy = y + offs[k, 1]
                    xx = x + offs[k, 2]
                    if yy < 0 or yy >= Y or xx < 0 or xx >= X:
                        continue
                    if offs[k, 0] < 0:
                        v = prev[yy, xx]
                    else:
                        v = cur[yy, xx]
                    if v != 0 and v != m:
                        union(parent, size, m, v)
            cur[y, x] = m
            hasv[m] = True
            stat_add(S, m, z, y, x, zbase + y * X + x)
    return M


@njit(cache=True)
def _classify(parent, S, hasv, present, newid, closed, M):
    """Fold stats into roots, then split roots into open and closed.

    Returns (n_open, n_closed).  ``closed[:n_closed]`` lists the ids of roots
    with zero frontier voxels; ``newid`` maps open roots to 1..n_open.
    """
    fold_into_roots(parent, S, M)
    for i in range(1, M + 1):
        present[i] = False
    for i in range(1, M + 1):
        if hasv[i]:
            present[find(parent, i)] = True
    nn = 0
    nc = 0
    for i in range(1, M + 1):
        if parent[i] == i:
            if present[i]:
                nn += 1
                newid[i] = nn
            else:
                closed[nc] = i
                nc += 1
    return nn, nc


@njit(cache=True)
def _compact(cur, parent, newid, S, M):
    """Move open roots' rows to 1..K' and remap ``cur`` to the new ids."""
    Y, X = cur.shape
    # order-preserving: newid[i] <= i, so each destination row has already
    # been consumed (or is the source row itself)
    for i in range(1, M + 1):
        if parent[i] == i:
            j = newid[i]
            if j != 0 and j != i:
                for c in range(S.shape[1]):
                    S[j, c] = S[i, c]
    for y in range(Y):
        for x in range(X):
            v = cur[y, x]
            if v != 0:
                cur[y, x] = newid[find(parent, v)]


@njit(cache=True)
def _clear_newid(newid, M):
    for i in range(M + 1):
        newid[i] = 0


def stream_components(slices, connectivity=6, spacing=(1.0, 1.0, 1.0),
                      return_info=False):
    """Component table from an iterable of 2-D mask slices (z increasing).

    The label volume is never materialised; the result is the per-component
    table (count, centroid, bounding box, first raster index), ordered like the
    in-memory labeler's ids.
    """
    offs = causal_offsets(connectivity)
    it = iter(slices)
    try:
        first = next(it)
    except StopIteration:
        t = ComponentTable(np.zeros((0, NSTAT), np.int64), spacing)
        return (t, {"slices": 0, "peak_open": 0, "peak_capacity": 0}) if return_info else t
    first = np.asarray(first)
    Y, X = first.shape
    prev = np.zeros((Y, X), np.int32)
    cur = np.zeros((Y, X), np.int32)
    cap = 0
    parent = size = hasv = present = newid = closed = S = None
    K = 0
    z = 0
    peak_open = 0
    open_trace = []
    closed_trace = []
    emitted = []
    for sl in itertools.chain([first], it):
        sl = np.asarray(sl)
        if sl.shape != (Y, X):
            raise ValueError("all slices must have the same shape")
        if sl.dtype != np.bool_ and sl.dtype != np.uint8:
            sl = sl != 0
        sl = np.ascontiguousarray(sl)
        runs = int(count_run_starts(sl))
        need = K + min(runs, (Y * X + 1) // 2) + 2
        if need > cap:
            newcap = max(need, 2 * cap, 256)
            nS = np.zeros((newcap, NSTAT), np.int64)
            if S is not None:
                nS[:K + 1] = S[:K + 1]
            S = nS
            parent = np.zeros(newcap, np.int32)
            size = np.zeros(newcap, np.int32)
            hasv = np.zeros(newcap, np.bool_)
            present = np.zeros(newcap, np.bool_)
            newid = np.zeros(newcap, np.int32)
            closed = np.zeros(newcap, np.int32)
            cap = newcap
        M = _scan_slice(sl, z, prev, cur, K, parent, size, S, hasv, offs)
        _clear_newid(newid, M)
        nn, nc = _classify(parent, S, hasv, present, newid, closed, M)
        open_trace.append(nn)
        closed_trace.append(nc)
        if nc:
            emitted.append(S[closed[:nc]].copy())
        _compact(cur, parent, newid, S, M)
        K = nn
        peak_open = max(peak_open, K)
        prev, cur = cur, prev
        z += 1
    if K:
        emitted.append(S[1:K + 1].copy())
    data = np.vstack(emitted) if emitted else np.zeros((0, NSTAT), np.int64)
    table = ComponentTable(data, spacing).sorted_by_first_index()
    if return_info:
        return table, {"slices": z, "peak_open": peak_open, "peak_capacity": cap,
                       "open_per_slice": open_trace, "closed_per_slice": closed_trace}
    return table
