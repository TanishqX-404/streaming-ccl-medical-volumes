import cc3d
import numpy as np
import pytest
import scipy.ndimage as ndi

from ufccl import canonicalize, check_invariants, label, same_partition
from ufccl.synth import checkerboard, random_mask, snake2d, snake3d

from conftest import scipy_label


@pytest.mark.parametrize("p", [0.05, 0.2, 0.31, 0.45, 0.8])
@pytest.mark.parametrize("shape", [(12, 14, 16), (1, 30, 30), (30, 1, 30), (30, 30, 1), (5, 6, 40)])
def test_matches_scipy_and_cc3d(conn, p, shape):
    m = random_mask(shape, p, seed=hash((p, shape, conn)) % 1000)
    ref, n_ref = scipy_label(m, conn)
    r = label(m, conn)
    assert r.n == n_ref
    assert same_partition(r.labels, ref)
    check_invariants(m, r.labels, r.n)
    c3 = cc3d.connected_components(m, connectivity=conn)
    assert same_partition(r.labels, c3)


def test_labels_follow_first_appearance_order_like_scipy(conn):
    m = random_mask((15, 15, 15), 0.3, seed=3)
    ref, _ = scipy_label(m, conn)
    assert np.array_equal(label(m, conn).labels, ref)


def test_empty_full_single_voxel_line(conn):
    z = np.zeros((4, 5, 6), bool)
    assert label(z, conn).n == 0 and not label(z, conn).labels.any()
    f = np.ones((4, 5, 6), bool)
    r = label(f, conn)
    assert r.n == 1 and np.all(r.labels == 1)
    v = np.zeros((3, 3, 3), bool)
    v[1, 1, 1] = True
    assert label(v, conn).n == 1
    line = np.zeros((1, 1, 9), bool)
    line[0, 0, :] = True
    assert label(line, conn).n == 1
    gap = line.copy()
    gap[0, 0, 4] = False
    assert label(gap, conn).n == 2


def test_connectivity_definitions_on_diagonal_pairs():
    face = np.zeros((2, 2, 2), bool)
    face[0, 0, 0] = face[0, 0, 1] = True
    edge = np.zeros((2, 2, 2), bool)
    edge[0, 0, 0] = edge[0, 1, 1] = True
    corner = np.zeros((2, 2, 2), bool)
    corner[0, 0, 0] = corner[1, 1, 1] = True
    assert [label(face, c).n for c in (6, 18, 26)] == [1, 1, 1]
    assert [label(edge, c).n for c in (6, 18, 26)] == [2, 1, 1]
    assert [label(corner, c).n for c in (6, 18, 26)] == [2, 2, 1]


def test_checkerboard_worst_case_label_count():
    m = checkerboard((9, 10, 11))
    r6 = label(m, 6)
    assert r6.n == int(m.sum())          # every voxel isolated under 6-connectivity
    assert label(m, 26).n == 1           # fully merged under 26-connectivity
    assert same_partition(r6.labels, scipy_label(m, 6)[0])


def test_snake_is_single_component_and_matches_oracle():
    m = snake2d(41, 63)[None]
    assert label(m, 6).n == 1
    m3 = snake3d(9, 21, 31)
    r = label(m3, 6)
    assert r.n == 1 and r.table is None
    assert same_partition(r.labels, scipy_label(m3, 6)[0])


def test_non_bool_inputs_are_binarised():
    m = random_mask((8, 8, 8), 0.4, seed=5)
    for dt in (np.uint8, np.int16, np.float32):
        assert same_partition(label(m.astype(dt) * 3, 6).labels, label(m, 6).labels)


def test_stats_match_scipy_measurements(conn):
    spacing = (2.5, 0.7, 0.8)
    m = random_mask((20, 24, 22), 0.25, seed=11)
    ref, n = scipy_label(m, conn)
    r = label(m, conn, stats=True, spacing=spacing)
    t = r.table
    assert len(t) == n
    idx = np.arange(1, n + 1)
    assert np.array_equal(t.voxels, ndi.sum(m, ref, idx).astype(int))
    com = np.array(ndi.center_of_mass(m, ref, idx))
    assert np.allclose(t.centroid_vox, com)
    assert np.allclose(t.centroid_mm, com * spacing)
    for i, sl in enumerate(ndi.find_objects(ref)):
        lo = [s.start for s in sl]
        hi = [s.stop - 1 for s in sl]
        assert list(t.bbox_min[i]) == lo and list(t.bbox_max[i]) == hi
    assert np.allclose(t.volume_mm3, t.voxels * np.prod(spacing))
    flat = np.argsort(ref.ravel(), kind="stable")
    first = [int(np.flatnonzero(ref.ravel() == i)[0]) for i in range(1, n + 1)]
    assert list(t.first_index) == first


def test_table_filter_and_csv(tmp_path):
    m = random_mask((10, 10, 10), 0.3, seed=2)
    t = label(m, 6, stats=True, spacing=(2.0, 1.0, 1.0)).table
    big = t.filter(min_volume_mm3=20.0)
    assert np.all(big.volume_mm3 >= 20.0) and len(big) <= len(t)
    p = tmp_path / "t.csv"
    big.to_csv(p)
    assert p.read_text().count("\n") == len(big) + 1


def test_pass2_does_not_corrupt_shared_roots():
    # many provisional labels share a root that is itself not the smallest id
    m = np.zeros((1, 7, 7), bool)
    m[0, :, 0] = m[0, 0, :] = m[0, :, 6] = True     # U shape: needs merges
    m[0, 3, 3] = True                               # separate speck
    r = label(m, 6)
    assert r.n == 2
    assert same_partition(r.labels, scipy_label(m, 6)[0])
