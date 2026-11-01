# Results

Hardware: AMD Ryzen 5 5600H with Radeon Graphics (6 cores), 15.4 GB RAM; Python 3.10.11, NumPy 2.2.6, Numba 0.67.0, SciPy 1.15.3, cc3d 4.x. Single thread, best of 3–5 runs.

## Correctness

130 volumes × 3 connectivities = 390 comparisons (76,770 components, 1,470,597 foreground voxels).

| check | passed |
|---|---|
| scipy_exact | 390/390 |
| cc3d_partition | 390/390 |
| stream_table | 390/390 |
| tiled_exact | 390/390 |
| stats_vs_scipy | 390/390 |

## Time (6-connectivity, seconds; throughput in Mvoxel/s)

| input | N | fg % | components | ufccl | ufccl+stats | scipy | cc3d | streaming | tiled 128³ | ufccl Mvox/s | exact vs scipy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blobs | 64³ | 20.6 | 88 | 0.002 | 0.003 | 0.005 | 0.001 | 0.004 | 0.005 | 116 | True |
| blobs | 96³ | 24.3 | 205 | 0.008 | 0.009 | 0.015 | 0.005 | 0.014 | 0.016 | 111 | True |
| blobs | 128³ | 26.6 | 492 | 0.020 | 0.023 | 0.037 | 0.012 | 0.031 | 0.039 | 104 | True |
| blobs | 192³ | 21.7 | 1,868 | 0.061 | 0.074 | 0.130 | 0.038 | 0.091 | 0.151 | 116 | True |
| blobs | 256³ | 21.1 | 4,264 | 0.164 | 0.195 | 0.282 | 0.098 | 0.238 | 0.359 | 102 | True |
| blobs | 384³ | 20.6 | 14,207 | 0.563 | 0.627 | 0.991 | 0.390 | 0.734 | 1.290 | 101 | True |
| blobs | 512³ | 20.4 | 32,818 | 1.578 | 1.617 | 2.617 | 1.000 | 1.957 | 3.172 | 85 | True |
| lesions | 64³ | 1.8 | 5 | 0.001 | 0.001 | 0.003 | 0.001 | 0.001 | 0.003 | 223 | True |
| lesions | 96³ | 2.2 | 13 | 0.004 | 0.005 | 0.010 | 0.003 | 0.004 | 0.012 | 225 | True |
| lesions | 128³ | 2.4 | 31 | 0.011 | 0.012 | 0.026 | 0.006 | 0.009 | 0.027 | 184 | True |
| lesions | 192³ | 2.5 | 96 | 0.034 | 0.042 | 0.092 | 0.024 | 0.029 | 0.118 | 209 | True |
| lesions | 256³ | 2.5 | 211 | 0.081 | 0.094 | 0.203 | 0.059 | 0.064 | 0.287 | 207 | True |
| lesions | 384³ | 2.6 | 705 | 0.286 | 0.316 | 0.665 | 0.185 | 0.227 | 0.952 | 198 | True |
| lesions | 512³ | 2.7 | 1,593 | 0.776 | 0.718 | 1.958 | 0.581 | 0.725 | 2.875 | 173 | True |
| noise | 64³ | 30.9 | 14,699 | 0.005 | 0.007 | 0.007 | 0.003 | 0.015 | 0.013 | 51 | True |
| noise | 96³ | 30.9 | 49,086 | 0.018 | 0.023 | 0.027 | 0.013 | 0.053 | 0.038 | 50 | True |
| noise | 128³ | 31.0 | 114,977 | 0.043 | 0.059 | 0.071 | 0.031 | 0.118 | 0.102 | 49 | True |
| noise | 192³ | 31.0 | 384,765 | 0.150 | 0.186 | 0.206 | 0.091 | 0.354 | 0.332 | 47 | True |
| noise | 256³ | 31.0 | 905,133 | 0.336 | 0.435 | 0.453 | 0.216 | 0.840 | 0.820 | 50 | True |
| noise | 384³ | 31.0 | 3,040,054 | 1.156 | 1.434 | 1.548 | 0.737 | 2.687 | 2.614 | 49 | True |
| noise | 512³ | 31.0 | 7,181,695 | 2.732 | 3.580 | 3.481 | 1.719 | 7.514 | 6.872 | 49 | True |

## Connectivity (blobs)

| N | comps 6 | comps 18 | comps 26 | ufccl 6 | ufccl 18 | ufccl 26 | cc3d 6 | cc3d 26 |
|---|---|---|---|---|---|---|---|---|
| 64³ | 88 | 85 | 84 | 0.002 | 0.006 | 0.009 | 0.001 | 0.001 |
| 96³ | 205 | 180 | 180 | 0.008 | 0.021 | 0.030 | 0.005 | 0.005 |
| 128³ | 492 | 448 | 441 | 0.020 | 0.053 | 0.077 | 0.012 | 0.012 |
| 192³ | 1,868 | 1,751 | 1,739 | 0.061 | 0.162 | 0.217 | 0.038 | 0.037 |
| 256³ | 4,264 | 3,966 | 3,936 | 0.164 | 0.377 | 0.561 | 0.098 | 0.103 |
| 384³ | 14,207 | 13,233 | 13,145 | 0.563 | 1.220 | 1.800 | 0.390 | 0.318 |
| 512³ | 32,818 | 30,608 | 30,360 | 1.578 | 3.213 | 4.329 | 1.000 | 0.941 |

## Statistics: fused vs post-hoc (6-connectivity, seconds)

| input | N | labeling only | labeling + fused stats | post-hoc stats pass alone |
|---|---|---|---|---|
| blobs | 64³ | 0.002 | 0.003 | 0.001 |
| blobs | 96³ | 0.008 | 0.009 | 0.002 |
| blobs | 128³ | 0.020 | 0.023 | 0.005 |
| blobs | 192³ | 0.061 | 0.074 | 0.015 |
| blobs | 256³ | 0.164 | 0.195 | 0.038 |
| blobs | 384³ | 0.563 | 0.627 | 0.122 |
| blobs | 512³ | 1.578 | 1.617 | 0.279 |
| lesions | 64³ | 0.001 | 0.001 | 0.000 |
| lesions | 96³ | 0.004 | 0.005 | 0.001 |
| lesions | 128³ | 0.011 | 0.012 | 0.002 |
| lesions | 192³ | 0.034 | 0.042 | 0.006 |
| lesions | 256³ | 0.081 | 0.094 | 0.014 |
| lesions | 384³ | 0.286 | 0.316 | 0.051 |
| lesions | 512³ | 0.776 | 0.718 | 0.125 |
| noise | 64³ | 0.005 | 0.007 | 0.003 |
| noise | 96³ | 0.018 | 0.023 | 0.009 |
| noise | 128³ | 0.043 | 0.059 | 0.024 |
| noise | 192³ | 0.150 | 0.186 | 0.073 |
| noise | 256³ | 0.336 | 0.435 | 0.177 |
| noise | 384³ | 1.156 | 1.434 | 0.580 |
| noise | 512³ | 2.732 | 3.580 | 1.486 |

All benchmark volumes: streaming table, tiled labels and in-memory labels agree with scipy exactly for 6/18/26: **True**.

## Peak memory (extra working set over the post-warm-up baseline, MB)

| N | ufccl | ufccl+stats | scipy | cc3d | tiled (disk out) | streaming |
|---|---|---|---|---|---|---|
| 128³ | 9 | 10 | 10 | 13 | 33 | 0 |
| 256³ | 80 | 83 | 81 | 112 | 34 | 1 |
| 384³ | 273 | 274 | 271 | 380 | 37 | 4 |
| 512³ | 662 | 663 | 644 | 900 | 43 | 10 |

Fixed 256×256 slices, growing depth:

| Z | voxels | ufccl in-memory | streaming |
|---|---|---|---|
| 128 | 8 M | 40 | 1 |
| 512 | 34 M | 161 | 2 |
| 2048 | 134 M | 662 | 4 |
| 8192 | 537 M | - | 36 |

## Union-Find ablation (6-connectivity, 96³-class inputs)

Steps = parent-pointer loop iterations (pass 1 / pass 2). Times use the instrumented kernel, so compare rows with each other only.

| input | compression | link | time (s) | find calls | steps/find | longest find | steps total (p1+p2) |
|---|---|---|---|---|---|---|---|
| blobs | none | size | 0.015 | 128,312 | 1.18 | 4 | 155,403 |
| blobs | none | index | 0.016 | 128,312 | 1.36 | 5 | 179,251 |
| blobs | none | naive | 0.183 | 128,312 | 229.68 | 1749 | 32,365,181 |
| blobs | halving | size | 0.014 | 128,312 | 0.92 | 2 | 121,616 |
| blobs | halving | index | 0.021 | 128,312 | 0.95 | 2 | 124,348 |
| blobs | halving | naive | 0.015 | 128,312 | 1.05 | 8 | 138,734 |
| blobs | full | size | 0.014 | 128,312 | 0.93 | 5 | 123,435 |
| blobs | full | index | 0.014 | 128,312 | 0.96 | 5 | 126,813 |
| blobs | full | naive | 0.014 | 128,312 | 1.41 | 25 | 189,968 |
| noise | none | size | 0.027 | 107,600 | 0.50 | 5 | 115,214 |
| noise | none | index | 0.023 | 107,600 | 0.51 | 6 | 120,020 |
| noise | none | naive | 0.033 | 107,600 | 2.31 | 195 | 2,103,487 |
| noise | halving | size | 0.023 | 107,600 | 0.41 | 2 | 85,926 |
| noise | halving | index | 0.022 | 107,600 | 0.40 | 2 | 85,716 |
| noise | halving | naive | 0.023 | 107,600 | 0.58 | 7 | 124,843 |
| noise | full | size | 0.022 | 107,600 | 0.47 | 7 | 117,974 |
| noise | full | index | 0.022 | 107,600 | 0.48 | 7 | 120,979 |
| noise | full | naive | 0.023 | 107,600 | 1.14 | 27 | 231,026 |
| snake | none | size | 0.001 | 624 | 0.50 | 1 | 767 |
| snake | none | index | 0.001 | 624 | 0.50 | 1 | 767 |
| snake | none | naive | 0.001 | 624 | 0.23 | 12 | 27,300 |
| snake | halving | size | 0.001 | 624 | 0.50 | 1 | 623 |
| snake | halving | index | 0.001 | 624 | 0.50 | 1 | 623 |
| snake | halving | naive | 0.001 | 624 | 0.12 | 6 | 723 |
| snake | full | size | 0.001 | 624 | 0.50 | 1 | 911 |
| snake | full | index | 0.001 | 624 | 0.50 | 1 | 911 |
| snake | full | naive | 0.001 | 624 | 0.44 | 23 | 1,210 |
| comb | none | size | 0.001 | 4,000 | 0.00 | 0 | 2,000 |
| comb | none | index | 0.001 | 4,000 | 0.00 | 0 | 2,000 |
| comb | none | naive | 0.019 | 4,000 | 499.75 | 1999 | 4,000,000 |
| comb | halving | size | 0.000 | 4,000 | 0.00 | 0 | 2,000 |
| comb | halving | index | 0.000 | 4,000 | 0.00 | 0 | 2,000 |
| comb | halving | naive | 0.000 | 4,000 | 0.50 | 1 | 6,925 |
| comb | full | size | 0.000 | 4,000 | 0.00 | 0 | 2,000 |
| comb | full | index | 0.000 | 4,000 | 0.00 | 0 | 2,000 |
| comb | full | naive | 0.000 | 4,000 | 1.50 | 3 | 11,993 |

## Operation counts vs N (path halving + union by size unless noted)

| input | N | provisional labels | find calls | steps/find | steps/N | longest find |
|---|---|---|---|---|---|---|
| blobs | 32,768 | 154 | 5,712 | 0.868 | 0.151 | 1 |
| blobs | 110,592 | 262 | 7,484 | 0.857 | 0.058 | 2 |
| blobs | 262,144 | 996 | 29,544 | 0.917 | 0.103 | 1 |
| blobs | 884,736 | 3,201 | 131,513 | 0.925 | 0.137 | 2 |
| blobs | 2,097,152 | 7,850 | 299,010 | 0.950 | 0.136 | 2 |
| blobs | 7,077,888 | 23,678 | 878,388 | 0.927 | 0.115 | 2 |
| blobs | 16,777,216 | 56,789 | 2,080,219 | 0.954 | 0.118 | 2 |
| noise | 32,768 | 3,527 | 7,223 | 0.395 | 0.087 | 2 |
| noise | 110,592 | 11,608 | 24,776 | 0.414 | 0.093 | 2 |
| noise | 262,144 | 27,146 | 58,600 | 0.425 | 0.095 | 2 |
| noise | 884,736 | 91,229 | 198,829 | 0.432 | 0.097 | 2 |
| noise | 2,097,152 | 215,358 | 472,904 | 0.436 | 0.098 | 2 |
| noise | 7,077,888 | 727,493 | 1,605,367 | 0.440 | 0.100 | 2 |
| noise | 16,777,216 | 1,717,570 | 3,799,356 | 0.440 | 0.100 | 2 |
| snake | 4,913 | 41 | 121 | 0.653 | 0.016 | 1 |
| snake | 35,937 | 145 | 433 | 0.663 | 0.008 | 1 |
| snake | 117,649 | 313 | 937 | 0.665 | 0.005 | 1 |
| snake | 274,625 | 545 | 1,633 | 0.666 | 0.004 | 1 |
| snake | 912,673 | 1,201 | 3,601 | 0.666 | 0.003 | 1 |
| comb (halving+size) | 1,503 | 251 | 751 | 0.333 | 0.166 | 0 |
| comb (none+naive) | 1,503 | 251 | 751 | 83.222 | 41.583 | 249 |
| comb (halving+size) | 3,003 | 501 | 1,501 | 0.333 | 0.167 | 0 |
| comb (none+naive) | 3,003 | 501 | 1,501 | 166.556 | 83.250 | 499 |
| comb (halving+size) | 6,003 | 1,001 | 3,001 | 0.333 | 0.167 | 0 |
| comb (none+naive) | 6,003 | 1,001 | 3,001 | 333.222 | 166.583 | 999 |
| comb (halving+size) | 12,003 | 2,001 | 6,001 | 0.333 | 0.167 | 0 |
| comb (none+naive) | 12,003 | 2,001 | 6,001 | 666.556 | 333.250 | 1999 |
| comb (halving+size) | 24,003 | 4,001 | 12,001 | 0.333 | 0.167 | 0 |
| comb (none+naive) | 24,003 | 4,001 | 12,001 | 1333.222 | 666.583 | 3999 |
| comb (halving+size) | 48,003 | 8,001 | 24,001 | 0.333 | 0.167 | 0 |
| comb (none+naive) | 48,003 | 8,001 | 24,001 | 2666.556 | 1333.250 | 7999 |

## Lesion-counting application (synthetic phantoms)

10 phantom scans of shape (96, 192, 192), spacing (2.5, 0.8, 0.8) mm; simulated segmentation output with mean voxel Dice 0.850.

| min volume (mm³) | predicted lesions / scan | false positives / scan | lesion sensitivity |
|---|---|---|---|
| 0 | 741.8 | 725.5 | 0.927 |
| 5 | 40.5 | 24.8 | 0.893 |
| 10 | 40.4 | 24.7 | 0.893 |
| 20 | 38.4 | 22.7 | 0.893 |
| 40 | 31.4 | 15.7 | 0.893 |
| 80 | 20.2 | 4.5 | 0.893 |
| 160 | 15.7 | 0.0 | 0.893 |

## Real data: Medical Segmentation Decathlon Task06_Lung (training set, 63 CT scans)

Public de-identified CT (CC-BY-SA 4.0). Two masks per scan: the tumour annotation (label 1) and the soft-tissue window [-200, 300] HU. Volumes are reoriented to RAS+ and stored (Z, Y, X); spacing is read from the NIfTI header.

Scans: 63, shapes from 112 to 636 slices of 512x512; 4.63 G voxels in total, largest 167 M voxels.

### Exactness

| mask | connectivity | scans where scipy (exact), cc3d (partition), streaming and tiled all agree | components (all scans) |
|---|---|---|---|
| tumour | 6 | 63/63 | 164 |
| tumour | 18 | 63/63 | 144 |
| tumour | 26 | 63/63 | 141 |
| hu | 6 | 63/63 | 1,594,786 |
| hu | 18 | 63/63 | 452,133 |
| hu | 26 | 63/63 | 331,557 |

### Time on the soft-tissue mask, 6-connectivity (seconds per scan)

| method | total | mean | median | max | Mvoxel/s (total) |
|---|---|---|---|---|---|
| ufccl | 38.2 | 0.61 | 0.56 | 1.32 | 121 |
| ufccl + stats | 48.2 | 0.77 | 0.71 | 1.74 | 96 |
| scipy | 65.8 | 1.05 | 0.97 | 2.33 | 70 |
| cc3d | 18.2 | 0.29 | 0.28 | 0.61 | 254 |
| ufccl streaming | 63.9 | 1.02 | 0.94 | 2.30 | 72 |
| ufccl tiled 128³ (RAM) | 100.1 | 1.59 | 1.47 | 3.57 | 46 |

Per scan, ufccl is 1.72x faster than scipy (range 1.39-2.15) and 2.08x slower than cc3d (range 1.70-2.53).

Tumour mask (very sparse), total seconds over all scans: ufccl 12.2, scipy 50.9, cc3d 7.6.

### Peak memory on the three largest scans (soft-tissue mask, MB over baseline)

| scan | voxels | ufccl | ufccl+stats | scipy | cc3d | tiled | stream |
|---|---|---|---|---|---|---|---|
| lung_047 | 167 M | 794 | 796 | 797 | 973 | 36 | 4 |
| lung_014 | 154 M | 728 | 735 | 738 | 847 | 31 | 5 |
| lung_006 | 149 M | 706 | 704 | 709 | 895 | 26 | 4 |

### Tumour components (6-connectivity, annotation itself)

164 components in 63 scans: per scan median 1, range 1-14. Volume (mm³): median 3, quartiles 1-4125, min 1, max 338472.

| connectivity | components | per-scan median |
|---|---|---|
| 6 | 164 | 1 |
| 18 | 144 | 1 |
| 26 | 141 | 1 |

| minimum volume (mm³) | components kept | scans with ≥1 kept |
|---|---|---|
| 0 | 164 | 63 |
| 10 | 72 | 63 |
| 50 | 68 | 63 |
| 100 | 65 | 63 |
| 500 | 63 | 63 |
| 1000 | 60 | 60 |
