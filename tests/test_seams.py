"""Tiled variant: components crossing tile seams in every way."""
import numpy as np
import pytest

from ufccl import label, label_tiled, same_partition
from ufccl.synth import lesion_phantom, random_mask
from ufccl.volume import RawVolume

from conftest import scipy_label


def _check(m, tile, conn):
    r = label(m, conn, stats=True)
    lab, n, table = label_tiled(m, tile, conn)
    assert n == r.n
    assert np.array_equal(lab, r.labels)
    assert table.same_as(r.table)
    return n


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_line_crossing_every_seam_along_each_axis(axis, conn):
    m = np.zeros((12, 12, 12), bool)
    idx = [6, 6, 6]
    idx[axis] = slice(None)
    m[tuple(idx)] = True
    assert _check(m, (4, 4, 4), conn) == 1


def test_diagonals_crossing_seams_need_matching_connectivity():
    # seam between voxel 3 and 4 on every axis
    edge = np.zeros((8, 8, 8), bool)
    edge[3, 3, 3] = edge[3, 4, 4] = True       # edge neighbour across y and x seams
    corner = np.zeros((8, 8, 8), bool)
    corner[3, 3, 3] = corner[4, 4, 4] = True   # corner neighbour across all three
    anti_corner = np.zeros((8, 8, 8), bool)
    anti_corner[3, 4, 3] = anti_corner[4, 3, 4] = True   # dz=+1, dy=-1, dx=+1
    anti_edge = np.zeros((8, 8, 8), bool)
    anti_edge[3, 4, 3] = anti_edge[4, 3, 3] = True       # dz=+1, dy=-1: later voxel sits on
    # the tile's high-y face and its causal neighbour is in the +y tile (high-side halo)
    for m, expect in ((edge, (2, 1, 1)), (corner, (2, 2, 1)),
                      (anti_corner, (2, 2, 1)), (anti_edge, (2, 1, 1))):
        for c, e in zip((6, 18, 26), expect):
            assert _check(m, (4, 4, 4), c) == e, (c,)


def test_dense_random_with_many_tile_shapes(conn):
    m = random_mask((17, 19, 23), 0.33, seed=7)
    ref = scipy_label(m, conn)[0]
    for tile in [(1, 1, 1), (2, 3, 5), (4, 4, 4), (8, 8, 8), (17, 19, 23), (100, 100, 100)]:
        lab, n, _ = label_tiled(m, tile, conn)
        assert np.array_equal(lab, ref), tile


def test_sparse_lesions_spanning_many_tiles(conn):
    m = lesion_phantom((40, 48, 48), n_lesions=25, r_range=(4, 10), n_vessels=3, seed=4)
    assert _check(m, (13, 16, 11), conn) >= 1


def test_file_backed_out_of_core_matches_in_memory(tmp_path, conn):
    m = random_mask((20, 21, 22), 0.3, seed=9)
    src_path, out_path = tmp_path / "m.u8", tmp_path / "out.i32"
    m.astype(np.uint8).tofile(src_path)
    src = RawVolume(src_path, m.shape, np.uint8)
    out = RawVolume.create(out_path, m.shape, np.int32)
    _, n, table = label_tiled(src, (7, 8, 9), conn, out=out)
    out.flush()
    got = np.fromfile(out_path, np.int32).reshape(m.shape)
    r = label(m, conn, stats=True)
    assert n == r.n and np.array_equal(got, r.labels) and table.same_as(r.table)
    src.close()
    out.close()


def test_tile_larger_than_volume_and_unit_dimensions():
    m = random_mask((1, 5, 40), 0.5, seed=1)
    assert same_partition(label_tiled(m, (64, 64, 16), 6)[0], scipy_label(m, 6)[0])
