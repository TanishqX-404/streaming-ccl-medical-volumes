"""Figures and results/RESULTS.md from the JSON files written by run_bench.py."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGS, RESULTS, load_json

COL = {"ufccl": "#1f77b4", "scipy": "#d62728", "cc3d": "#2ca02c",
       "stream": "#ff7f0e", "tiled": "#9467bd", "stats": "#8c564b"}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def have(name):
    return os.path.exists(os.path.join(RESULTS, name))


def fig_scaling(rows):
    kinds = sorted({r["kind"] for r in rows})
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        rr = [r for r in rows if r["kind"] == kind]
        N = np.array([r["voxels"] for r in rr])
        for key, label, c, ls in [("conn6.ufccl_s", "ufccl (6)", COL["ufccl"], "-"),
                                  ("conn6.scipy_s", "scipy (6)", COL["scipy"], "-"),
                                  ("conn6.cc3d_s", "cc3d (6)", COL["cc3d"], "-"),
                                  ("stream6_s", "ufccl streaming", COL["stream"], "--"),
                                  ("tiled6_s", "ufccl tiled 128³", COL["tiled"], "--")]:
            y = [r["conn6"][key.split(".")[1]] if "." in key else r[key] for r in rr]
            ax.loglog(N, y, ls, color=c, marker="o", ms=3.5, label=label)
        ax.loglog(N, N / N[-1] * [r["conn6"]["ufccl_s"] for r in rr][-1], ":", color="gray",
                  label="linear")
        ax.set_title(kind)
        ax.set_xlabel("voxels N")
    axes[0].set_ylabel("wall-clock time (s)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "scaling_time.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    big = [r for r in rows if r["n"] == max(x["n"] for x in rows)]
    names = []
    for r in big:
        names.append(r["kind"])
    x = np.arange(len(big))
    w = 0.16
    series = [("ufccl", lambda r: r["conn6"]["ufccl_s"]), ("scipy", lambda r: r["conn6"]["scipy_s"]),
              ("cc3d", lambda r: r["conn6"]["cc3d_s"]), ("stream", lambda r: r["stream6_s"]),
              ("tiled", lambda r: r["tiled6_s"])]
    for i, (k, f) in enumerate(series):
        thr = [r["voxels"] / f(r) / 1e6 for r in big]
        ax.bar(x + (i - 2) * w, thr, w, color=COL[k], label=k)
    ax.set_xticks(x, names)
    ax.set_ylabel("throughput (Mvoxel/s)")
    ax.set_title(f"6-connectivity, {big[0]['n']}³ volume")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "throughput.png"))
    plt.close(fig)


def fig_connectivity(rows):
    rr = [r for r in rows if r["kind"] == "blobs"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, ls in zip((6, 18, 26), ("-", "--", ":")):
        ax.loglog([r["voxels"] for r in rr], [r[f"conn{c}"]["ufccl_s"] for r in rr], ls,
                  color=COL["ufccl"], marker="o", ms=3, label=f"ufccl {c}")
        ax.loglog([r["voxels"] for r in rr], [r[f"conn{c}"]["cc3d_s"] for r in rr], ls,
                  color=COL["cc3d"], marker="s", ms=3, label=f"cc3d {c}")
    ax.set_xlabel("voxels N")
    ax.set_ylabel("time (s)")
    ax.set_title("connectivity ablation (blobs)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "connectivity.png"))
    plt.close(fig)


def fig_memory(rows):
    cube = [r for r in rows if r["experiment"] == "cube"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for v in ("ufccl", "ufccl+stats", "scipy", "cc3d", "tiled", "stream"):
        rr = [r for r in cube if r["variant"] == v]
        if not rr:
            continue
        c = COL.get(v.split("+")[0], "gray")
        ax.loglog([r["voxels"] for r in rr], [max(r["delta_mb"], 1) for r in rr],
                  "--" if v in ("stream", "tiled") else "-", marker="o", ms=4,
                  color=c if v != "ufccl+stats" else COL["stats"], label=v)
    ax.set_xlabel("voxels N")
    ax.set_ylabel("peak extra memory (MB)")
    ax.set_title("cube volumes (blobs)")
    ax.legend(fontsize=8)
    ax = axes[1]
    dep = [r for r in rows if r["experiment"] == "depth"]
    for v in ("ufccl", "stream"):
        rr = [r for r in dep if r["variant"] == v]
        ax.semilogx([r["shape"][0] for r in rr], [r["delta_mb"] for r in rr], marker="o",
                    color=COL[v], label=v)
    ax.set_xlabel("depth Z (slices of 256×256)")
    ax.set_ylabel("peak extra memory (MB)")
    ax.set_title("fixed slice size, growing depth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "memory.png"))
    plt.close(fig)


def fig_complexity(d):
    sc = d["scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series = [("blobs", None, "#1f77b4", "-"), ("noise", None, "#d62728", "-"),
              ("snake", None, "#2ca02c", "-"), ("comb", "halving+size", "#9467bd", "-"),
              ("comb", "none+naive", "#9467bd", "--")]
    for kind, cfg, c, ls in series:
        rr = [r for r in sc if r["kind"] == kind and r.get("config") == cfg]
        if not rr:
            continue
        N = [r["voxels"] for r in rr]
        name = kind if cfg is None else f"{kind} ({cfg})"
        axes[0].loglog(N, [(r["steps1"] + r["steps2"]) / r["voxels"] for r in rr],
                       ls, marker="o", ms=3, color=c, label=name)
        axes[1].loglog(N, [max(r["maxpath1"], 1) for r in rr], ls, marker="o", ms=3, color=c,
                       label=name)
    axes[0].set_xlabel("voxels N")
    axes[0].set_ylabel("parent-pointer steps per voxel")
    axes[0].set_title("find work per voxel (flat ⇒ O(N α(N)))")
    axes[1].set_xlabel("voxels N")
    axes[1].set_ylabel("longest single find (steps)")
    axes[1].set_title("worst path length")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "complexity.png"))
    plt.close(fig)


def fig_app(d):
    s = d["summary"]
    thr = sorted(s, key=float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fp = [s[t]["fp_per_scan_mean"] for t in thr]
    se = [s[t]["sensitivity_mean"] for t in thr]
    ax[0].plot([float(t) for t in thr], fp, marker="o", color="#d62728")
    ax[0].set_xlabel("minimum lesion volume (mm³)")
    ax[0].set_ylabel("false-positive lesions per scan")
    ax[0].set_title("size filter removes spurious components")
    ax[1].plot(fp, se, marker="o", color="#1f77b4")
    for t, x, y in zip(thr, fp, se):
        ax[1].annotate(t, (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax[1].set_xlabel("false positives per scan")
    ax[1].set_ylabel("lesion-wise sensitivity")
    ax[1].set_title("operating curve (labels: min volume, mm³)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "lesion_operating_curve.png"))
    plt.close(fig)


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def real_section(L):
    import csv
    d = load_json("real.json")
    cs = d["cases"]
    n = len(cs)
    vox = np.array([c["voxels"] for c in cs], float)
    L.append(f"## Real data: Medical Segmentation Decathlon {d['task_dir']} (training set, {n} CT scans)\n")
    L.append("Public de-identified CT (CC-BY-SA 4.0). Two masks per scan: the tumour annotation "
             "(label 1) and the soft-tissue window [-200, 300] HU. Volumes are reoriented to RAS+ "
             "and stored (Z, Y, X); spacing is read from the NIfTI header.\n")
    L.append(f"Scans: {n}, shapes from {min(c['shape'][0] for c in cs)} to {max(c['shape'][0] for c in cs)} "
             f"slices of {cs[0]['shape'][1]}x{cs[0]['shape'][2]}; {vox.sum() / 1e9:.2f} G voxels in total, "
             f"largest {vox.max() / 1e6:.0f} M voxels.\n")
    L.append("### Exactness\n")
    L.append("| mask | connectivity | scans where scipy (exact), cc3d (partition), streaming and tiled all agree "
             "| components (all scans) |\n|---|---|---|---|")
    for tag in ("tumour", "hu"):
        for c in (6, 18, 26):
            ok = sum(r[f"{tag}_c{c}_ok"] for r in cs)
            tot = sum(r[f"{tag}_c{c}_n"] for r in cs)
            L.append(f"| {tag} | {c} | {ok}/{n} | {tot:,} |")
    L.append("")
    L.append("### Time on the soft-tissue mask, 6-connectivity (seconds per scan)\n")
    L.append("| method | total | mean | median | max | Mvoxel/s (total) |\n|---|---|---|---|---|---|")
    for key, name in [("hu_ufccl_s", "ufccl"), ("hu_ufccl_stats_s", "ufccl + stats"),
                      ("hu_scipy_s", "scipy"), ("hu_cc3d_s", "cc3d"),
                      ("hu_stream_s", "ufccl streaming"), ("hu_tiled_s", "ufccl tiled 128³ (RAM)")]:
        t = np.array([c[key] for c in cs])
        L.append(f"| {name} | {t.sum():.1f} | {t.mean():.2f} | {np.median(t):.2f} | {t.max():.2f} "
                 f"| {vox.sum() / t.sum() / 1e6:.0f} |")
    r_sp = np.array([c["hu_scipy_s"] / c["hu_ufccl_s"] for c in cs])
    r_cc = np.array([c["hu_ufccl_s"] / c["hu_cc3d_s"] for c in cs])
    L.append(f"\nPer scan, ufccl is {np.median(r_sp):.2f}x faster than scipy (range {r_sp.min():.2f}-{r_sp.max():.2f}) "
             f"and {np.median(r_cc):.2f}x slower than cc3d (range {r_cc.min():.2f}-{r_cc.max():.2f}).\n")
    L.append("Tumour mask (very sparse), total seconds over all scans: "
             f"ufccl {sum(c['tumour_ufccl_s'] for c in cs):.1f}, scipy {sum(c['tumour_scipy_s'] for c in cs):.1f}, "
             f"cc3d {sum(c['tumour_cc3d_s'] for c in cs):.1f}.\n")
    if d["memory"]:
        L.append("### Peak memory on the three largest scans (soft-tissue mask, MB over baseline)\n")
        variants = ["ufccl", "ufccl+stats", "scipy", "cc3d", "tiled", "stream"]
        L.append("| scan | voxels | " + " | ".join(variants) + " |\n|---|---|" + "---|" * len(variants))
        for case in dict.fromkeys(m["case"] for m in d["memory"]):
            rr = {m["variant"]: m for m in d["memory"] if m["case"] == case}
            L.append(f"| {case} | {next(iter(rr.values()))['voxels'] / 1e6:.0f} M | "
                     + " | ".join(f"{rr[v]['delta_mb']:.0f}" for v in variants) + " |")
        L.append("")
    path = os.path.join(RESULTS, "real_lesions.csv")
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        vol = np.array([float(r["volume_mm3"]) for r in rows])
        per_case = np.array([sum(1 for r in rows if r["case"] == c["case"]) for c in cs])
        L.append("### Tumour components (6-connectivity, annotation itself)\n")
        L.append(f"{len(rows)} components in {n} scans: per scan median {np.median(per_case):.0f}, "
                 f"range {per_case.min()}-{per_case.max()}. Volume (mm³): median {np.median(vol):.0f}, "
                 f"quartiles {np.percentile(vol, 25):.0f}-{np.percentile(vol, 75):.0f}, "
                 f"min {vol.min():.0f}, max {vol.max():.0f}.\n")
        L.append("| connectivity | components | per-scan median |\n|---|---|---|")
        for c in (6, 18, 26):
            k = [r[f"tumour_c{c}_n"] for r in cs]
            L.append(f"| {c} | {sum(k):,} | {np.median(k):.0f} |")
        L.append("")
        L.append("| minimum volume (mm³) | components kept | scans with ≥1 kept |\n|---|---|---|")
        for t in (0, 10, 50, 100, 500, 1000):
            kept = [r for r in rows if float(r["volume_mm3"]) >= t]
            L.append(f"| {t} | {len(kept)} | {len({r['case'] for r in kept})} |")
        L.append("")
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, name, c in [("hu_ufccl_s", "ufccl", COL["ufccl"]), ("hu_scipy_s", "scipy", COL["scipy"]),
                         ("hu_cc3d_s", "cc3d", COL["cc3d"]), ("hu_stream_s", "streaming", COL["stream"]),
                         ("hu_tiled_s", "tiled", COL["tiled"])]:
        ax.scatter(vox / 1e6, [x[key] for x in cs], s=10, color=c, label=name)
    ax.set_xlabel("scan size (Mvoxel)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"MSD {d['task_dir']} CT, soft-tissue mask, 6-conn")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "real_time.png"))
    plt.close(fig)


def write_markdown():
    L = ["# Results\n"]
    if have("scaling.json"):
        hw = load_json("scaling.json")["hardware"]
        L.append(f"Hardware: {hw.get('cpu')} ({hw.get('cores')} cores), {hw.get('ram_gb')} GB RAM; "
                 f"Python {hw.get('python')}, NumPy {hw.get('numpy')}, Numba {hw.get('numba')}, "
                 f"SciPy {hw.get('scipy')}, cc3d {hw.get('cc3d')}. Single thread, best of 3–5 runs.\n")
    if have("correctness.json"):
        c = load_json("correctness.json")
        L.append("## Correctness\n")
        L.append(f"{c['volumes']} volumes × 3 connectivities = {c['comparisons']} comparisons "
                 f"({c['components_checked']:,} components, {c['foreground_voxels_checked']:,} "
                 f"foreground voxels).\n")
        L.append("| check | passed |\n|---|---|")
        for k, v in c["checks"].items():
            L.append(f"| {k} | {v['passed']}/{v['total']} |")
        L.append("")
    if have("scaling.json"):
        rows = load_json("scaling.json")["rows"]
        L.append("## Time (6-connectivity, seconds; throughput in Mvoxel/s)\n")
        L.append("| input | N | fg % | components | ufccl | ufccl+stats | scipy | cc3d | streaming | tiled 128³ "
                 "| ufccl Mvox/s | exact vs scipy |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            c6 = r["conn6"]
            L.append(f"| {r['kind']} | {r['n']}³ | {100 * r['fg_fraction']:.1f} | {c6['components']:,} "
                     f"| {fmt(c6['ufccl_s'])} | {fmt(r['ufccl6_stats_s'])} | {fmt(c6['scipy_s'])} "
                     f"| {fmt(c6['cc3d_s'])} | {fmt(r['stream6_s'])} | {fmt(r['tiled6_s'])} "
                     f"| {r['voxels'] / c6['ufccl_s'] / 1e6:.0f} | {c6['exact_vs_scipy']} |")
        L.append("")
        L.append("## Connectivity (blobs)\n")
        L.append("| N | comps 6 | comps 18 | comps 26 | ufccl 6 | ufccl 18 | ufccl 26 | cc3d 6 | cc3d 26 |"
                 "\n|---|---|---|---|---|---|---|---|---|")
        for r in [r for r in rows if r["kind"] == "blobs"]:
            L.append(f"| {r['n']}³ | {r['conn6']['components']:,} | {r['conn18']['components']:,} "
                     f"| {r['conn26']['components']:,} | {fmt(r['conn6']['ufccl_s'])} "
                     f"| {fmt(r['conn18']['ufccl_s'])} | {fmt(r['conn26']['ufccl_s'])} "
                     f"| {fmt(r['conn6']['cc3d_s'])} | {fmt(r['conn26']['cc3d_s'])} |")
        L.append("")
        L.append("## Statistics: fused vs post-hoc (6-connectivity, seconds)\n")
        L.append("| input | N | labeling only | labeling + fused stats | post-hoc stats pass alone |"
                 "\n|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['kind']} | {r['n']}³ | {fmt(r['conn6']['ufccl_s'])} | {fmt(r['ufccl6_stats_s'])} "
                     f"| {fmt(r['posthoc_stats_s'])} |")
        L.append("")
        ok = all(r["stream6_matches"] and r["tiled6_matches"] and r["conn6"]["exact_vs_scipy"]
                 and r["conn18"]["exact_vs_scipy"] and r["conn26"]["exact_vs_scipy"] for r in rows)
        L.append(f"All benchmark volumes: streaming table, tiled labels and in-memory labels agree "
                 f"with scipy exactly for 6/18/26: **{ok}**.\n")
        fig_scaling(rows)
        fig_connectivity(rows)
    if have("memory.json"):
        rows = load_json("memory.json")["rows"]
        L.append("## Peak memory (extra working set over the post-warm-up baseline, MB)\n")
        L.append("| N | " + " | ".join(["ufccl", "ufccl+stats", "scipy", "cc3d", "tiled (disk out)", "streaming"])
                 + " |\n|---|---|---|---|---|---|---|")
        for n in sorted({r["shape"][0] for r in rows if r["experiment"] == "cube"}):
            cells = []
            for v in ("ufccl", "ufccl+stats", "scipy", "cc3d", "tiled", "stream"):
                m = [r for r in rows if r["experiment"] == "cube" and r["variant"] == v
                     and r["shape"][0] == n]
                cells.append(f"{m[0]['delta_mb']:.0f}" if m else "-")
            L.append(f"| {n}³ | " + " | ".join(cells) + " |")
        L.append("\nFixed 256×256 slices, growing depth:\n")
        L.append("| Z | voxels | ufccl in-memory | streaming |\n|---|---|---|---|")
        for Z in sorted({r["shape"][0] for r in rows if r["experiment"] == "depth"}):
            def g(v):
                m = [r for r in rows if r["experiment"] == "depth" and r["variant"] == v
                     and r["shape"][0] == Z]
                return f"{m[0]['delta_mb']:.0f}" if m else "-"
            L.append(f"| {Z} | {Z * 256 * 256 / 1e6:.0f} M | {g('ufccl')} | {g('stream')} |")
        L.append("")
        fig_memory(rows)
    if have("complexity.json"):
        d = load_json("complexity.json")
        L.append("## Union-Find ablation (6-connectivity, 96³-class inputs)\n")
        L.append("Steps = parent-pointer loop iterations (pass 1 / pass 2). "
                 "Times use the instrumented kernel, so compare rows with each other only.\n")
        L.append("| input | compression | link | time (s) | find calls | steps/find | longest find "
                 "| steps total (p1+p2) |\n|---|---|---|---|---|---|---|---|")
        for r in d["ablation"]:
            L.append(f"| {r['input']} | {r['compress']} | {r['link']} | {fmt(r['seconds'])} "
                     f"| {r['finds1']:,} | {r['steps1'] / max(r['finds1'], 1):.2f} | {r['maxpath1']} "
                     f"| {r['steps1'] + r['steps2']:,} |")
        L.append("\n## Operation counts vs N (path halving + union by size unless noted)\n")
        L.append("| input | N | provisional labels | find calls | steps/find | steps/N | longest find |"
                 "\n|---|---|---|---|---|---|---|")
        for r in d["scaling"]:
            f = r["finds1"] + r["finds2"]
            tag = r["kind"] + (f" ({r['config']})" if "config" in r else "")
            L.append(f"| {tag} | {r['voxels']:,} | {r['provisional']:,} | {f:,} "
                     f"| {(r['steps1'] + r['steps2']) / max(f, 1):.3f} "
                     f"| {(r['steps1'] + r['steps2']) / r['voxels']:.3f} | {r['maxpath1']} |")
        L.append("")
        fig_complexity(d)
    if have("app.json"):
        d = load_json("app.json")
        L.append("## Lesion-counting application (synthetic phantoms)\n")
        L.append(f"{d['scans']} phantom scans of shape {tuple(d['shape'])}, spacing {tuple(d['spacing_mm'])} mm; "
                 f"simulated segmentation output with mean voxel Dice {d['dice_mean']:.3f}.\n")
        L.append("| min volume (mm³) | predicted lesions / scan | false positives / scan | lesion sensitivity |"
                 "\n|---|---|---|---|")
        for t in sorted(d["summary"], key=float):
            s = d["summary"][t]
            L.append(f"| {t} | {s['pred_lesions_mean']:.1f} | {s['fp_per_scan_mean']:.1f} "
                     f"| {s['sensitivity_mean']:.3f} |")
        L.append("")
        fig_app(d)
    if have("real.json"):
        real_section(L)
    with open(os.path.join(RESULTS, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote RESULTS.md")


if __name__ == "__main__":
    write_markdown()
