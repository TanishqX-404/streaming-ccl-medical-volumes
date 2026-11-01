"""Property-based tests: random volumes, every variant against scipy."""
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ufccl import (check_invariants, label, label_tiled, same_partition,
                   stream_components)

from conftest import scipy_label

SETTINGS = settings(max_examples=150, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

dim = st.integers(1, 9)
shapes = st.tuples(dim, dim, dim)


@st.composite
def masks(draw):
    shape = draw(shapes)
    p = draw(st.sampled_from([0.1, 0.3, 0.5, 0.7, 0.95]))
    seed = draw(st.integers(0, 2**32 - 1))
    return np.random.default_rng(seed).random(shape) < p


@st.composite
def raw_masks(draw):
    return draw(arrays(np.bool_, draw(shapes)))


@SETTINGS
@given(m=st.one_of(masks(), raw_masks()), c=st.sampled_from([6, 18, 26]))
def test_in_memory_matches_scipy_and_invariants(m, c):
    ref, n = scipy_label(m, c)
    r = label(m, c)
    assert r.n == n
    assert same_partition(r.labels, ref)
    check_invariants(m, r.labels, r.n)


@SETTINGS
@given(m=st.one_of(masks(), raw_masks()), c=st.sampled_from([6, 18, 26]))
def test_streaming_table_equals_in_memory_table(m, c):
    r = label(m, c, stats=True)
    t = stream_components(iter(m), c)
    assert t.same_as(r.table)


@SETTINGS
@given(m=st.one_of(masks(), raw_masks()), c=st.sampled_from([6, 18, 26]),
       tile=st.tuples(st.integers(1, 5), st.integers(1, 5), st.integers(1, 5)))
def test_tiled_equals_in_memory_exactly(m, c, tile):
    r = label(m, c, stats=True)
    lab, n, table = label_tiled(m, tile, c)
    assert n == r.n
    assert np.array_equal(lab, r.labels)
    assert table.same_as(r.table)


@SETTINGS
@given(m=masks(), tile=st.tuples(st.integers(1, 5), st.integers(1, 5), st.integers(1, 5)))
def test_tiled_without_stats_has_same_partition(m, tile):
    ref, n = scipy_label(m, 6)
    lab, n_t, table = label_tiled(m, tile, 6, stats=False)
    assert table is None and n_t == n
    assert same_partition(lab, ref)


@SETTINGS
@given(m=masks())
def test_connectivity_is_monotone(m):
    n6, n18, n26 = (label(m, c).n for c in (6, 18, 26))
    assert n6 >= n18 >= n26
