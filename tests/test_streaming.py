import numpy as np

from ufccl import label, stream_components
from ufccl.synth import blob_slices, lesion_phantom, random_mask, snake3d, stack


def test_streaming_equals_in_memory_on_blobs(conn):
    m = stack(blob_slices((40, 48, 48), cell=8, thr=0.58, seed=2))
    r = label(m, conn, stats=True, spacing=(2.0, 0.8, 0.8))
    t = stream_components(iter(m), conn, spacing=(2.0, 0.8, 0.8))
    assert len(t) > 1 and t.same_as(r.table)
    assert np.allclose(t.volume_mm3, r.table.volume_mm3)


def test_components_close_early_and_memory_is_bounded():
    # lesions occupy short z-ranges: peak open components must stay far below total
    m = lesion_phantom((120, 40, 40), n_lesions=60, r_range=(2, 4), seed=3)
    t, info = stream_components(iter(m), 6, return_info=True)
    assert len(t) == label(m, 6).n
    assert info["peak_open"] < len(t)
    assert info["peak_capacity"] < 40 * 40


def test_long_snake_is_one_component():
    m = snake3d(9, 21, 31)
    t = stream_components(iter(m), 6)
    assert len(t) == 1 and t.voxels[0] == m.sum()


def test_empty_and_single_slice():
    assert len(stream_components(iter([]), 6)) == 0
    assert len(stream_components(iter(np.zeros((3, 4, 4), bool)), 6)) == 0
    m = random_mask((1, 20, 20), 0.4, seed=1)
    assert stream_components(iter(m), 6).same_as(label(m, 6, stats=True).table)


def test_component_merging_after_it_looked_closed():
    # two prongs that start separate and join several slices later
    m = np.zeros((6, 5, 5), bool)
    m[0:4, 0, 0] = True
    m[0:4, 0, 4] = True
    m[3, 0, :] = True
    t = stream_components(iter(m), 6)
    assert len(t) == 1 and t.voxels[0] == m.sum()
