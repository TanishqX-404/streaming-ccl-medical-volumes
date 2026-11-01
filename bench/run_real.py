"""Benchmark on a Medical Segmentation Decathlon task directory.

    python bench/run_real.py PATH/Task06_Lung [--limit N]

Expects imagesTr/*.nii.gz and labelsTr/*.nii.gz.  Two masks per case:
  tumour : label == 1 (the annotation itself)
  hu     : CT in the soft-tissue window [-200, 300] HU
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

from common import DATA, RESULTS, best_of, hardware, save_json
from run_bench import STRUCT, probe, sp_label

import cc3d
import scipy.ndimage as ndi

from ufccl import label, label_tiled, same_partition, stream_components
from ufccl.io import hu_window_mask, load_nifti


def check_all(mask, tag, row):
    """Exactness of every variant at 6/18/26 on one mask; returns n components per conn."""
    ns = {}
    for conn in (6, 18, 26):
        ref, n_ref = sp_label(mask, conn)
        r = label(mask, conn, stats=True)
        ok_sp = bool(r.n == n_ref and np.array_equal(r.labels, ref))
        ok_cc = same_partition(r.labels, cc3d.connected_components(mask, connectivity=conn))
        ok_st = stream_components(iter(mask), conn).same_as(r.table)
        lab_t, n_t, tab_t = label_tiled(mask, (128, 128, 128), conn)
        ok_ti = bool(n_t == r.n and np.array_equal(lab_t, r.labels) and tab_t.same_as(r.table))
        row[f"{tag}_c{conn}_n"] = int(r.n)
        row[f"{tag}_c{conn}_ok"] = bool(ok_sp and ok_cc and ok_st and ok_ti)
        ns[conn] = r.table
        del ref, r, lab_t
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-memory", action="store_true")
    a = ap.parse_args()
    imgs = sorted(glob.glob(os.path.join(a.task_dir, "imagesTr", "lung_*.nii.gz")))
    if a.limit:
        imgs = imgs[:a.limit]
    # warm-up JIT
    m0 = np.random.default_rng(0).random((20, 20, 20)) < 0.4
    for c in (6, 18, 26):
        label(m0, c, stats=True)
    stream_components(iter(m0), 6)
    label_tiled(m0, (8, 8, 8), 6)

    rows, lesions = [], []
    for i, ip in enumerate(imgs):
        name = os.path.basename(ip).replace(".nii.gz", "")
        lp = os.path.join(a.task_dir, "labelsTr", os.path.basename(ip))
        ct, spacing, _ = load_nifti(ip, dtype=np.float32)
        lab_vol, sp2, _ = load_nifti(lp)
        assert lab_vol.shape == ct.shape and np.allclose(spacing, sp2), name
        row = {"case": name, "shape": list(ct.shape), "spacing_mm": list(spacing),
               "voxels": int(ct.size)}
        tumour = lab_vol == 1
        hu = hu_window_mask(ct, -200.0, 300.0)
        del ct, lab_vol
        row["tumour_voxels"] = int(tumour.sum())
        row["hu_fg_fraction"] = float(hu.mean())

        # exactness of all variants, both masks, all connectivities
        tum_tabs = check_all(tumour, "tumour", row)
        check_all(hu, "hu", row)

        # per-lesion table for the tumour mask (6-connectivity)
        t6 = tum_tabs[6]
        for k in range(len(t6)):
            lesions.append((name, int(t6.ids[k]), int(t6.voxels[k]),
                            float(t6.volume_mm3[k])))

        # timing on the HU mask (the large one), 6-connectivity
        reps = 3
        row["hu_ufccl_s"] = best_of(lambda: label(hu, 6), reps)[0]
        row["hu_ufccl_stats_s"] = best_of(lambda: label(hu, 6, stats=True), reps)[0]
        row["hu_scipy_s"] = best_of(lambda: ndi.label(hu), reps)[0]
        row["hu_cc3d_s"] = best_of(lambda: cc3d.connected_components(hu, connectivity=6), reps)[0]
        row["hu_stream_s"] = best_of(lambda: stream_components(iter(hu), 6), reps)[0]
        row["hu_tiled_s"] = best_of(lambda: label_tiled(hu, (128, 128, 128), 6), 2)[0]
        row["tumour_ufccl_s"] = best_of(lambda: label(tumour, 6), reps)[0]
        row["tumour_scipy_s"] = best_of(lambda: ndi.label(tumour), reps)[0]
        row["tumour_cc3d_s"] = best_of(lambda: cc3d.connected_components(tumour, connectivity=6), reps)[0]
        rows.append(row)
        print(i + 1, len(imgs), name, row["shape"], "hu_n6", row["hu_c6_n"], "tum_n6", row["tumour_c6_n"],
              "ok", all(v for k, v in row.items() if k.endswith("_ok")),
              f"ufccl {row['hu_ufccl_s']:.2f}s scipy {row['hu_scipy_s']:.2f}s cc3d {row['hu_cc3d_s']:.2f}s",
              flush=True)
        # keep the three largest HU masks on disk for the memory probe
        if not a.no_memory:
            keep = sorted(rows, key=lambda r: -r["voxels"])[:3]
            if row in keep:
                os.makedirs(DATA, exist_ok=True)
                hu.astype(np.uint8).tofile(os.path.join(DATA, f"real_{name}.u8"))
        del tumour, hu

    mem = []
    if not a.no_memory:
        big = sorted(rows, key=lambda r: -r["voxels"])[:3]
        for r in big:
            path = os.path.join(DATA, f"real_{r['case']}.u8")
            for v in ("ufccl", "ufccl+stats", "scipy", "cc3d", "tiled", "stream"):
                m = probe(v, path, tuple(r["shape"]))
                m.update(case=r["case"], variant=v, shape=r["shape"], voxels=r["voxels"])
                mem.append(m)
                print("mem", r["case"], v, round(m["delta_mb"], 1), "MB", m["n"], flush=True)
            os.remove(path)
        for f in os.listdir(DATA):
            if f.startswith("real_"):
                os.remove(os.path.join(DATA, f))
    with open(os.path.join(RESULTS, "real_lesions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "lesion_id", "voxels", "volume_mm3"])
        w.writerows(lesions)
    save_json("real.json", {"hardware": hardware(), "task_dir": os.path.basename(a.task_dir.rstrip("/\\")),
                            "cases": rows, "memory": mem})


if __name__ == "__main__":
    main()
