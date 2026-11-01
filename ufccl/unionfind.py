"""Union-Find primitives and per-component statistics rows (Numba).

Union-Find uses path halving in ``find`` and union by size in ``union``.

Statistics rows are int64 vectors of length ``NSTAT`` stored in a 2-D array
``S`` with one row per component (or provisional label).  Merging two rows is
additive for count/sums and min/max for the bounding box and first index, so a
row can be folded into the row of its Union-Find root at any time.
"""
import numpy as np
from numba import njit

# Column layout of a statistics row.
C_COUNT = 0
C_SZ, C_SY, C_SX = 1, 2, 3          # coordinate sums (for centroid)
C_MINZ, C_MINY, C_MINX = 4, 5, 6    # bounding box, inclusive
C_MAXZ, C_MAXY, C_MAXX = 7, 8, 9
C_FIRST = 10                         # smallest raster index (z*Y*X + y*X + x)
NSTAT = 11

_BIG = 1 << 62


@njit(cache=True)
def find(parent, x):
    """Representative of x's set, with path halving."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def union(parent, size, a, b):
    """Merge the sets of a and b by size; returns the surviving root."""
    ra = find(parent, a)
    rb = find(parent, b)
    if ra == rb:
        return ra
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]
    return ra


@njit(cache=True)
def stat_reset(S, i):
    for c in range(NSTAT):
        S[i, c] = 0
    S[i, C_MINZ] = _BIG
    S[i, C_MINY] = _BIG
    S[i, C_MINX] = _BIG
    S[i, C_MAXZ] = -1
    S[i, C_MAXY] = -1
    S[i, C_MAXX] = -1
    S[i, C_FIRST] = _BIG


@njit(cache=True)
def stats_new(n):
    """``n`` empty statistics rows."""
    S = np.zeros((n, NSTAT), np.int64)
    for i in range(n):
        stat_reset(S, i)
    return S


@njit(cache=True)
def stat_add(S, i, z, y, x, lin):
    """Add voxel (z, y, x) with raster index ``lin`` to row i."""
    S[i, C_COUNT] += 1
    S[i, C_SZ] += z
    S[i, C_SY] += y
    S[i, C_SX] += x
    if z < S[i, C_MINZ]:
        S[i, C_MINZ] = z
    if y < S[i, C_MINY]:
        S[i, C_MINY] = y
    if x < S[i, C_MINX]:
        S[i, C_MINX] = x
    if z > S[i, C_MAXZ]:
        S[i, C_MAXZ] = z
    if y > S[i, C_MAXY]:
        S[i, C_MAXY] = y
    if x > S[i, C_MAXX]:
        S[i, C_MAXX] = x
    if lin < S[i, C_FIRST]:
        S[i, C_FIRST] = lin


@njit(cache=True)
def stat_merge(S, dst, src):
    """Fold row ``src`` into row ``dst``."""
    S[dst, C_COUNT] += S[src, C_COUNT]
    S[dst, C_SZ] += S[src, C_SZ]
    S[dst, C_SY] += S[src, C_SY]
    S[dst, C_SX] += S[src, C_SX]
    for c in range(C_MINZ, C_MINX + 1):
        if S[src, c] < S[dst, c]:
            S[dst, c] = S[src, c]
    for c in range(C_MAXZ, C_MAXX + 1):
        if S[src, c] > S[dst, c]:
            S[dst, c] = S[src, c]
    if S[src, C_FIRST] < S[dst, C_FIRST]:
        S[dst, C_FIRST] = S[src, C_FIRST]


@njit(cache=True)
def fold_into_roots(parent, S, n):
    """Fold the stats row of every non-root id in 1..n into its root's row."""
    for i in range(1, n + 1):
        r = find(parent, i)
        if r != i:
            stat_merge(S, r, i)


class UnionFind:
    """Plain Python wrapper around the jitted primitives (used by tests)."""

    def __init__(self, n, dtype=np.int32):
        self.parent = np.arange(n, dtype=dtype)
        self.size = np.ones(n, dtype=dtype)

    def find(self, x):
        return int(find(self.parent, x))

    def union(self, a, b):
        return int(union(self.parent, self.size, a, b))

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    @property
    def n_sets(self):
        return int(sum(1 for i in range(len(self.parent)) if self.parent[i] == i))
