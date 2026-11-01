"""Operation-counting labeler for complexity validation and ablations.

Same raster scan as :mod:`ufccl.label` but with selectable Union-Find variants
and counters.  Used only for measurement; the production kernels live in
``label.py``.

compress: 0 = none, 1 = path halving, 2 = full compression (two passes)
link:     0 = union by size, 1 = union by index (smaller id is root),
          2 = naive (root of the first argument is attached under the second)

Counter vector (int64): [0] find calls (pass 1), [1] parent-pointer steps
(pass 1), [2] longest single find (pass 1), [3] union calls, [4] unions that
merged two different sets, [5] find calls (pass 2), [6] steps (pass 2),
[7] longest find (pass 2).
"""
import numpy as np
from numba import njit

from .label import causal_offsets, count_run_starts

COUNTER_NAMES = ["finds1", "steps1", "maxpath1", "union_calls", "merges",
                 "finds2", "steps2", "maxpath2"]


@njit(cache=True)
def _find_c(parent, x, compress, cnt, k):
    cnt[k] += 1
    h = 0
    if compress == 1:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
            h += 1
        r = x
    else:
        r = x
        while parent[r] != r:
            r = parent[r]
            h += 1
        if compress == 2:
            while parent[x] != r and x != r:
                nxt = parent[x]
                parent[x] = r
                x = nxt
                h += 1
    cnt[k + 1] += h
    if h > cnt[k + 2]:
        cnt[k + 2] = h
    return r


@njit(cache=True)
def _union_c(parent, size, a, b, compress, link, cnt):
    ra = _find_c(parent, a, compress, cnt, 0)
    rb = _find_c(parent, b, compress, cnt, 0)
    cnt[3] += 1
    if ra == rb:
        return
    cnt[4] += 1
    if link == 0:
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
    elif link == 1:
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
    else:
        parent[ra] = rb


@njit(cache=True)
def _run(mask, offs, lab, parent, size, compress, link, cnt):
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
                        _union_c(parent, size, m, v, compress, link, cnt)
    # resolution: one find per provisional label
    cnt2 = np.zeros(3, np.int64)
    roots = 0
    for i in range(1, nid + 1):
        r = _find_c(parent, i, compress, cnt2, 0)
        if r == i:
            roots += 1
    cnt[5] = cnt2[0]
    cnt[6] = cnt2[1]
    cnt[7] = cnt2[2]
    return nid, roots


def label_counted(mask, connectivity=6, compress=1, link=0):
    """Returns (n_components, n_provisional, counters dict)."""
    mask = np.ascontiguousarray(mask)
    runs = int(count_run_starts(mask.reshape(-1, mask.shape[2])))
    lab = np.zeros(mask.shape, np.int32)
    parent = np.arange(runs + 2, dtype=np.int32)
    size = np.ones(runs + 2, dtype=np.int32)
    cnt = np.zeros(8, np.int64)
    nid, roots = _run(mask, causal_offsets(connectivity), lab, parent, size,
                      compress, link, cnt)
    return int(roots), int(nid), dict(zip(COUNTER_NAMES, (int(c) for c in cnt)))
