"""Per-component table built from statistics rows."""
from dataclasses import dataclass

import numpy as np

from .unionfind import (C_COUNT, C_FIRST, C_MAXX, C_MAXY, C_MAXZ, C_MINX,
                        C_MINY, C_MINZ, C_SX, C_SY, C_SZ, NSTAT)

CSV_FIELDS = ["id", "voxels", "volume_mm3",
              "centroid_z_mm", "centroid_y_mm", "centroid_x_mm",
              "bbox_z0", "bbox_y0", "bbox_x0", "bbox_z1", "bbox_y1", "bbox_x1"]


@dataclass
class ComponentTable:
    """One row per component, ordered by first appearance in raster order.

    ``data`` is an (n, NSTAT) int64 array; ``ids`` are the label values the
    rows belong to (1..n unless the table was filtered).  Bounding boxes are
    inclusive voxel indices; centroids are in voxel units until multiplied by
    ``spacing`` (z, y, x).
    """
    data: np.ndarray
    spacing: tuple = (1.0, 1.0, 1.0)
    ids: np.ndarray = None

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.int64).reshape(-1, NSTAT)
        if self.ids is None:
            self.ids = np.arange(1, len(self.data) + 1)

    def __len__(self):
        return len(self.data)

    @property
    def voxels(self):
        return self.data[:, C_COUNT]

    @property
    def voxel_volume_mm3(self):
        return float(np.prod(self.spacing))

    @property
    def volume_mm3(self):
        return self.voxels * self.voxel_volume_mm3

    @property
    def centroid_vox(self):
        s = self.data[:, [C_SZ, C_SY, C_SX]].astype(np.float64)
        return s / np.maximum(self.voxels, 1)[:, None]

    @property
    def centroid_mm(self):
        return self.centroid_vox * np.asarray(self.spacing)

    @property
    def bbox_min(self):
        return self.data[:, [C_MINZ, C_MINY, C_MINX]]

    @property
    def bbox_max(self):
        return self.data[:, [C_MAXZ, C_MAXY, C_MAXX]]

    @property
    def first_index(self):
        return self.data[:, C_FIRST]

    def filter(self, min_voxels=0, min_volume_mm3=0.0):
        keep = (self.voxels >= min_voxels) & (self.volume_mm3 >= min_volume_mm3)
        return ComponentTable(self.data[keep], self.spacing, self.ids[keep])

    def sorted_by_first_index(self):
        o = np.argsort(self.first_index, kind="stable")
        return ComponentTable(self.data[o], self.spacing, self.ids[o])

    def same_as(self, other):
        return (self.data.shape == other.data.shape
                and bool(np.array_equal(self.data, other.data)))

    def rows(self):
        cm, b0, b1 = self.centroid_mm, self.bbox_min, self.bbox_max
        vol = self.volume_mm3
        for i in range(len(self)):
            yield (int(self.ids[i]), int(self.voxels[i]), float(vol[i]),
                   float(cm[i, 0]), float(cm[i, 1]), float(cm[i, 2]),
                   int(b0[i, 0]), int(b0[i, 1]), int(b0[i, 2]),
                   int(b1[i, 0]), int(b1[i, 1]), int(b1[i, 2]))

    def to_csv(self, path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_FIELDS)
            w.writerows(self.rows())
