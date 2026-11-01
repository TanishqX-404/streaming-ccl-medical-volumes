"""Lesion-counting pipeline: mask -> CCL -> size filter -> per-lesion table.

    python -m apps.lesion_count --input mask.nii.gz --min-volume-mm3 50 --out out/
    python -m apps.lesion_count --synthetic --out out/

``--input`` takes a NIfTI binary mask (e.g. an MSD / LiTS segmentation output;
pass ``--label-value`` to select one class of a multi-class mask).  Without
data, ``--synthetic`` runs the pipeline on a phantom.  This is a computational
method evaluated on public or synthetic data, not a diagnostic tool.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ufccl import label  # noqa: E402
from ufccl.synth import lesion_phantom  # noqa: E402


def count_lesions(mask, spacing, connectivity=6, min_volume_mm3=0.0):
    """Returns (label volume, table of lesions >= min_volume_mm3)."""
    r = label(mask, connectivity, stats=True, spacing=spacing)
    return r.labels, r.table.filter(min_volume_mm3=min_volume_mm3)


def lesion_wise_scores(pred_labels, pred_ids, gt_labels, n_gt):
    """Lesion-level sensitivity and false positives.

    A ground-truth lesion is detected if any kept predicted lesion overlaps it
    by at least one voxel; a kept predicted lesion that overlaps no
    ground-truth lesion is a false positive.
    """
    keep = np.zeros(int(pred_labels.max()) + 1, bool)
    keep[np.asarray(pred_ids, dtype=int)] = True
    kept = keep[pred_labels] & (pred_labels > 0)
    both = kept & (gt_labels > 0)
    pairs = np.unique(pred_labels[both].astype(np.int64) * (n_gt + 1) + gt_labels[both])
    detected = len(np.unique(pairs % (n_gt + 1)))
    matched_pred = len(np.unique(pairs // (n_gt + 1)))
    n_pred = len(pred_ids)
    return {"gt": n_gt, "detected": detected, "pred": n_pred,
            "false_positives": n_pred - matched_pred,
            "sensitivity": detected / n_gt if n_gt else float("nan")}


def noisy_prediction(gt_mask, spacing, rng, miss_rate=0.1, n_fp_blobs=25, speckle=2e-4):
    """Simulated segmentation output: ragged boundaries, missed lesions,
    spurious blobs and speckle (a stand-in for a real model's mask)."""
    import scipy.ndimage as ndi
    r = label(gt_mask, 6)
    gt_lab, n = r.labels, int(r.n)
    drop = rng.random(n + 1) < miss_rate
    drop[0] = False
    pred = gt_mask & ~drop[gt_lab]
    edge = ndi.gaussian_filter(rng.standard_normal(gt_mask.shape).astype(np.float32), 1.5)
    pred = (ndi.gaussian_filter(pred.astype(np.float32), 1.0) + 0.35 * edge) > 0.5
    fp = lesion_phantom(gt_mask.shape, n_lesions=n_fp_blobs, r_range=(1.0, 3.0),
                        seed=int(rng.integers(1 << 30)))
    pred |= fp & ~ndi.binary_dilation(gt_mask, iterations=2)
    pred |= rng.random(gt_mask.shape) < speckle
    return pred


def synthetic_study(n_scans=10, shape=(96, 192, 192), spacing=(2.5, 0.8, 0.8),
                    thresholds_mm3=(0, 5, 10, 20, 40, 80, 160)):
    rows = []
    for seed in range(n_scans):
        rng = np.random.default_rng(seed)
        gt = lesion_phantom(shape, n_lesions=18, r_range=(3, 9), seed=seed)
        g = label(gt, 6)
        gt_lab, n_gt = g.labels, int(g.n)
        pred = noisy_prediction(gt, spacing, rng)
        pred_lab, tab = count_lesions(pred, spacing, 6, 0.0)
        dice = 2.0 * (gt & pred).sum() / (gt.sum() + pred.sum())
        for thr in thresholds_mm3:
            kept = tab.filter(min_volume_mm3=thr)
            sc = lesion_wise_scores(pred_lab, kept.ids, gt_lab, n_gt)
            rows.append(dict(scan=seed, min_volume_mm3=thr, dice=float(dice), **sc))
    summary = {}
    for thr in thresholds_mm3:
        r = [x for x in rows if x["min_volume_mm3"] == thr]
        summary[str(thr)] = {
            "sensitivity_mean": float(np.mean([x["sensitivity"] for x in r])),
            "fp_per_scan_mean": float(np.mean([x["false_positives"] for x in r])),
            "pred_lesions_mean": float(np.mean([x["pred"] for x in r])),
            "gt_lesions_mean": float(np.mean([x["gt"] for x in r])),
        }
    return {"shape": list(shape), "spacing_mm": list(spacing), "scans": n_scans,
            "dice_mean": float(np.mean([x["dice"] for x in rows])),
            "summary": summary, "rows": rows}


def make_figure(vol, labels, table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    if len(table):
        big = int(np.argmax(table.voxels))
        zc = int(round(table.centroid_vox[big, 0]))
    else:
        zc = labels.shape[0] // 2
    sl = labels[zc]
    ax[0].imshow(vol[zc], cmap="gray")
    ax[0].set_title(f"mask, slice z={zc}")
    shown = np.ma.masked_where(sl == 0, sl % 20)
    ax[1].imshow(np.zeros_like(sl), cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(shown, cmap="tab20", interpolation="nearest")
    ax[1].set_title("labelled lesions")
    if len(table):
        ax[2].hist(table.volume_mm3, bins=min(30, max(5, len(table))), color="#4c78a8")
    ax[2].set_xlabel("lesion volume (mm$^3$)")
    ax[2].set_ylabel("count")
    ax[2].set_title(f"{len(table)} lesions")
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--label-value", type=int, default=None)
    ap.add_argument("--connectivity", type=int, default=6, choices=[6, 18, 26])
    ap.add_argument("--min-volume-mm3", type=float, default=0.0)
    ap.add_argument("--out", default="lesion_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.synthetic:
        spacing = (2.5, 0.8, 0.8)
        mask = lesion_phantom((96, 192, 192), 18, (3, 9), seed=0)
    elif a.input:
        from ufccl.io import load_nifti
        vol, spacing, _ = load_nifti(a.input)
        mask = (vol == a.label_value) if a.label_value is not None else (vol > 0)
    else:
        ap.error("give --input or --synthetic")
    labels, table = count_lesions(mask, spacing, a.connectivity, a.min_volume_mm3)
    table.to_csv(os.path.join(a.out, "lesions.csv"))
    make_figure(mask, labels, table, os.path.join(a.out, "lesions.png"))
    print(f"{len(table)} lesions (>= {a.min_volume_mm3} mm3, {a.connectivity}-connectivity), "
          f"total volume {table.volume_mm3.sum():.1f} mm3 -> {a.out}")


if __name__ == "__main__":
    main()
