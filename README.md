# ufccl3d — Union-Find connected component labeling for volumetric scans

3-D connected component labeling (6/18/26-connectivity) with Union-Find
(path halving, union by size), in three variants that give identical results:

| variant | entry point | memory | output |
|---|---|---|---|
| in-memory two-pass | `ufccl.label(mask, conn, stats=True)` | O(N) | label volume + per-lesion table |
| slice-streaming | `ufccl.stream_components(slices, conn)` | O(Y·X + open components) | per-lesion table (components are finalised as soon as they end) |
| tiled out-of-core | `ufccl.label_tiled(src, tile, conn, out=RawVolume(...))` | O(tile) + O(provisional labels) | label volume on disk + table |

The per-lesion table has voxel count, physical volume (uses voxel spacing),
centroid and bounding box.  Components are numbered by first appearance in
raster order, the same numbering `scipy.ndimage.label` uses, so in-memory and
tiled label volumes are **bit-identical** to scipy's, not merely
partition-equivalent.

The algorithm is classical (Rosenfeld–Pfaltz 1966, Tarjan 1975).  The
contribution is the engineering — streaming finalisation, incremental
statistics, seam-merged tiling — and the measurements below.

```
ufccl/        unionfind.py  label.py  streaming.py  tiled.py  volume.py
              stats.py  io.py (NIfTI/DICOM)  synth.py  validate.py  instrumented.py
tests/        130 tests: UF model tests, scipy/cc3d oracles, hypothesis, seams, streaming, I/O
bench/        run_bench.py (correctness|scaling|memory|complexity|app)  run_real.py  plot_results.py
apps/         lesion_count.py   (mask -> CCL -> size filter -> CSV + figure)
results/      RESULTS.md (all tables), *.json (raw numbers), figures/
report/       main.tex (LaTeX project report, SIT template), figures/, tools/make_figures.py
```

## Run

```bash
pip install numpy numba scipy nibabel pydicom matplotlib psutil pytest hypothesis connected-components-3d
python -m pytest                              # 130 tests, ~15 s after the first JIT compile
python bench/run_bench.py all                 # ~10 min; then:
python bench/plot_results.py                  # figures + results/RESULTS.md
python -m apps.lesion_count --synthetic --out out/
python -m apps.lesion_count --input liver_seg.nii.gz --label-value 2 --min-volume-mm3 50 --out out/
```

### Real data (not included in this repository)

The 63 CT scans of MSD Task06_Lung are public (CC-BY-SA 4.0) but about 9 GB, so
they are not stored here.  Download `Task06_Lung.tar` from the
[AWS Open Data registry](https://registry.opendata.aws/msd/) (bucket
`msd-for-monai`), extract it into its own folder, then:

```bash
python bench/run_real.py PATH/TO/Task06_Lung            # ~100 min for all 63 scans
python report/tools/make_figures.py PATH/TO/Task06_Lung # regenerates the report figures
```

`results/real.json` and `results/real_lesions.csv` hold the numbers (timings and
per-lesion volumes, no image data) from the run reported below.

### Report

`report/main.tex` is the project report (pdfLaTeX, run twice; needs only
`report/sit_logo.png` and `report/figures/`).

## Results

Machine: AMD Ryzen 5 5600H (6 cores, used single-threaded), 15.4 GB RAM,
Windows 11, Python 3.10.11, NumPy 2.2.6, Numba 0.67, SciPy 1.15.3, cc3d 4.1.1.
Best of 3–5 runs.  Full tables: [results/RESULTS.md](results/RESULTS.md).

**Two kinds of data.**  First, synthetic volumes for the scaling, ablation and
memory-vs-size experiments: `blobs`
(smooth thresholded value noise, ~21 % foreground), `lesions` (sparse
ellipsoids and random-walk vessels, ~2.5 %) and `noise` (i.i.d. p = 0.31, near
the 6-connected percolation threshold: 7.2 M components at 512³).  Second,
the 63 training CT scans of Medical Segmentation Decathlon Task06_Lung (see
"Real data" below).  LiTS, LIDC-IDRI and BraTS were not used.

### Correctness

130 volumes × {6, 18, 26} = 390 comparisons (76,770 components): 390/390 exact
label-array matches with `scipy.ndimage.label`, 390/390 partition matches with
cc3d, 390/390 streaming tables equal to the in-memory table, 390/390 tiled
outputs equal to the in-memory output (random tile shapes down to 1 voxel), and
390/390 voxel counts and centroids equal to scipy's.  All 21 benchmark volumes
(up to 512³) also match scipy exactly at 6/18/26.  Removing the one-voxel
high-side halo from the seam merge makes 6 tests fail, so the seam tests do
catch cross-seam errors.

### Speed (6-connectivity, seconds, 512³ = 134 M voxels)

| input | ufccl | ufccl + stats | scipy | cc3d | streaming | tiled 128³ (RAM) |
|---|---|---|---|---|---|---|
| blobs | 1.58 | 1.62 | 2.62 | **1.00** | 1.96 | 3.17 |
| lesions | 0.78 | 0.72 | 1.96 | **0.58** | 0.73 | 2.88 |
| noise (7.2 M comps) | 2.73 | 3.58 | 3.48 | **1.72** | 7.51 | 6.87 |

Time is linear in N from 64³ to 512³ for every method
(`results/figures/scaling_time.png`).  For volumes of 256³ and larger, ufccl is 1.3–2.5× faster than scipy
and 1.3–1.7× slower than cc3d; throughput is 85–225 Mvoxel/s depending on
input (49 on the percolation-threshold noise).  18- and 26-connectivity use a
generic offset loop and are slower than 6: 26-connectivity on blobs 512³ takes
4.3 s against cc3d's 0.94 s (4.6×).  I did not optimise that path (no
run-based scanning or decision trees).

### Memory (peak extra working set, 512³ blobs; each variant in its own process)

| ufccl in-memory | scipy | cc3d | **tiled** (128³, labels on disk) | **streaming** |
|---|---|---|---|---|
| 662 MB | 644 MB | 900 MB | 43 MB | 9.6 MB |

Streaming uses ~69× less memory than the in-memory labeler here and tiling
~15× less.  At fixed 256×256 slices the in-memory labeler grows linearly in Z
(40 → 161 → 662 MB for Z = 128 / 512 / 2048) while streaming stays at
1.2 → 1.8 → 4.1 MB.  At Z = 8192 streaming rises to 36 MB, but that is the
output table (133,220 components × 88 B, copied a few times), not
working state.  The streaming and tiled variants do not hold a full label
volume in RAM; streaming does not produce one at all.

### Union-Find behaviour (counted with an instrumented kernel)

* Path halving + union by size: 0.4–0.95 parent-pointer steps per find and a
  longest single find of 2 on blobs and noise up to 16.8 M voxels.
  Steps per voxel stay flat from 33 k to 16.8 M voxels, which is the measured
  version of the O(N·α(N)) claim.
* The serpentine "snake" from the project brief is **not** a stress case for
  this scan order (longest find: 1).  A comb (vertical teeth merged one at a
  time by a bottom bar) is: with no compression and naive linking the longest
  find is W/2 and work grows quadratically (1,333 steps/voxel at 48 k voxels);
  with halving + union by size it stays at 0.17 steps/voxel for every size
  tested.
* On realistic masks the variants differ little in time; only the combination
  of no compression and naive linking is bad (blobs: 32 M steps against
  122 k, 13× slower).

### Statistics

Computing statistics in the relabel sweep adds 0–30 % to labeling time
(blobs 512³: +0.04 s; noise 512³: +0.85 s), against 0.28 s and 1.49 s for a
separate post-hoc pass over the label volume.  In the streaming variant they
are the only output.  Streaming and tiled tables are exactly equal to the
in-memory table in every test and benchmark volume.

### Real data: MSD Task06_Lung, 63 training CTs (4.63 G voxels)

Two masks per scan: the tumour annotation (label 1) and the soft-tissue window
[-200, 300] HU.  Scans are 112–636 slices of 512×512, loaded with the NIfTI
header spacing after reorientation to RAS+.  Only the training scans were used
(Task06_Lung.tar from the AWS Open Data MSD bucket, CC-BY-SA 4.0).

* **Exactness:** for both masks and all of 6/18/26, 63/63 scans give an exact
  match with scipy, a partition match with cc3d, an identical streaming table
  and an identical tiled output.  The soft-tissue mask has 1.59 M components
  at 6-connectivity (0.45 M at 18, 0.33 M at 26) across the set.
* **Time, soft-tissue mask, 6-connectivity** (total over 63 scans / median per
  scan): ufccl 38.2 s / 0.56 s, scipy 65.8 s / 0.97 s, cc3d 18.2 s / 0.28 s,
  streaming 63.9 s / 0.94 s, tiled 128³ (RAM) 100.1 s / 1.47 s.  ufccl is 1.72×
  faster than scipy (1.39–2.15× per scan) and 2.08× slower than cc3d
  (1.70–2.53×).  Statistics in the same sweep raise ufccl to 48.2 s.
* **Peak memory, three largest scans (149–167 M voxels):** in-memory ufccl
  706–794 MB, scipy 709–797 MB, cc3d 847–973 MB, tiled 26–36 MB, streaming
  4–5 MB.
* **Tumour components:** the annotations contain 164 components at
  6-connectivity (144 at 18, 141 at 26), median 1 per scan, range 1–14.
  Volumes are very skewed: median 3 mm³, with 92 of the 164 components under
  10 mm³ (annotation fragments down to a single voxel), and the largest
  338 cm³.  A 1,000 mm³ minimum keeps 60 components and drops three scans
  that have only smaller components, so a fixed size filter changes which
  scans count as having a lesion.  Because these are the annotations
  themselves there is no detection or false-positive measurement; that needs a
  model's predictions.

### Lesion-counting demo (synthetic, illustrative only)

10 phantom scans (96×192×192, spacing 2.5×0.8×0.8 mm) with a simulated
segmentation output (Dice 0.85 against the phantom).  Without size filtering
the output has 742 components per scan, 726 of them false positives, mostly
speckle.  Filtering by lesion volume cuts false positives to 24.8 / 15.7 / 4.5
/ 0 per scan at ≥ 5 / 40 / 80 / 160 mm³, with lesion-wise sensitivity
0.927 → 0.893 once any filter is applied.  The false-positive sizes are set by my
simulation, so this shows the pipeline works, not how a real model behaves.

## Limitations

* Real-data evaluation covers one dataset (MSD Task06_Lung, CT, 63 scans).
  The liver, hepatic vessel, LiTS, LIDC-IDRI and BraTS data were not tried, and
  there are no model predictions, so lesion sensitivity and false positives are
  only measured on the synthetic demo.
* cc3d is faster than ufccl on all inputs (about 1.3–1.7× on the synthetic
  volumes, 2.1× on the CT soft-tissue masks at 6-connectivity).  The
  case for ufccl is memory-bounded streaming and tiling with fused statistics,
  not raw speed.
* Single-threaded; no GPU, no multi-process tiling.
* Streaming and tiled memory figures exclude the output table, which grows
  with the component count.
* Tiled labels with `stats=False` are partition-correct but numbered by tile
  order, not raster order; use `stats=True` for scipy-identical numbering.
* Peak memory is sampled from the child's working set every 3 ms, so
  sub-3 ms spikes could be missed and numbers below ~1 MB are noise.
* Python 3.10 was used rather than 3.11.

This is a computational method evaluated on synthetic and public data, not a
diagnostic tool.

## Data attribution

`report/figures/fig_real_ct.pdf` and `report/figures/banner.png` show CT slices
from the Medical Segmentation Decathlon Task06_Lung training set (scans
`lung_026` and others), which is released under
[CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); these images
are derived from that data and remain under the same licence.  Dataset:
M. Antonelli et al., "The Medical Segmentation Decathlon", Nature
Communications 13, 4128 (2022).  The code is not covered by that licence.

## Authors

Tanishq Singh Sisodiya and Subhanan Chatterjee, Symbiosis Institute of
Technology, Hyderabad (Data Structures in C).
