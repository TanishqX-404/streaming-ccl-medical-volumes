import numpy as np

from ufccl.unionfind import (C_COUNT, C_FIRST, C_MAXZ, C_MINZ, NSTAT, UnionFind,
                             find, fold_into_roots, stat_add, stats_new, union)


class NaiveSets:
    def __init__(self, n):
        self.group = list(range(n))

    def union(self, a, b):
        ga, gb = self.group[a], self.group[b]
        if ga != gb:
            self.group = [ga if g == gb else g for g in self.group]

    def connected(self, a, b):
        return self.group[a] == self.group[b]

    def n_sets(self):
        return len(set(self.group))


def test_random_operation_sequences_match_naive_model():
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = int(rng.integers(1, 80))
        uf, ref = UnionFind(n), NaiveSets(n)
        for _ in range(300):
            a, b = int(rng.integers(n)), int(rng.integers(n))
            if rng.random() < 0.5:
                uf.union(a, b)
                ref.union(a, b)
            else:
                assert uf.connected(a, b) == ref.connected(a, b)
        assert uf.n_sets == ref.n_sets()


def test_union_by_size_keeps_larger_root():
    uf = UnionFind(6)
    for i in (1, 2, 3):
        uf.union(0, i)           # set of 4 rooted at 0
    root = uf.union(5, 0)        # singleton joins big set
    assert root == uf.find(0) == uf.find(5)
    assert uf.find(0) == 0
    assert uf.size[0] == 5


def test_path_halving_shortens_chain():
    n = 64
    parent = np.arange(n, dtype=np.int32)
    for i in range(1, n):        # build a worst-case chain n-1 -> n-2 -> ... -> 0
        parent[i] = i - 1
    assert find(parent, n - 1) == 0
    depth = 0
    x = n - 1
    while parent[x] != x:
        x = parent[x]
        depth += 1
    assert depth <= n // 2 + 1


def test_find_is_idempotent_and_union_is_symmetric():
    a, b = UnionFind(10), UnionFind(10)
    pairs = [(1, 2), (3, 4), (2, 3), (7, 8), (0, 9)]
    for x, y in pairs:
        a.union(x, y)
        b.union(y, x)
    for i in range(10):
        assert a.find(a.find(i)) == a.find(i)
        for j in range(10):
            assert a.connected(i, j) == b.connected(i, j)


def test_stat_rows_merge_additively():
    S = stats_new(4)
    stat_add(S, 1, 0, 0, 0, 0)
    stat_add(S, 1, 0, 0, 1, 1)
    stat_add(S, 2, 3, 0, 0, 30)
    parent = np.arange(4, dtype=np.int32)
    size = np.ones(4, dtype=np.int32)
    union(parent, size, 1, 2)
    fold_into_roots(parent, S, 3)
    r = find(parent, 1)
    assert S[r, C_COUNT] == 3
    assert S[r, C_MINZ] == 0 and S[r, C_MAXZ] == 3
    assert S[r, C_FIRST] == 0
    assert S.shape[1] == NSTAT
