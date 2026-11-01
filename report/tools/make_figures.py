"""Publication figures for the report.

    python make_figures.py PATH/TO/Task06_Lung

Writes PDF (for LaTeX) and PNG (for quick viewing) into ../figures.
Numbers come from ufccl3d/results/*.json; schematics are drawn from the
actual ufccl code so they cannot disagree with the implementation.
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))   # repository root
OUT = os.path.abspath(os.path.join(HERE, "..", "figures"))
sys.path.insert(0, PROJ)
os.makedirs(OUT, exist_ok=True)

from ufccl import label, stream_components  # noqa: E402
from ufccl.io import hu_window_mask, load_nifti  # noqa: E402
from ufccl.synth import blob_slices, lesion_phantom, noise_slices, stack  # noqa: E402

RED = "#B3262B"
COL = {"ufccl": RED, "scipy": "#4C78A8", "cc3d": "#2A9D8F", "stream": "#E08E0B",
       "tiled": "#7B5EA7", "stats": "#8C6D31", "grey": "#6B6B6B"}
NAME = {"ufccl": "ufccl (ours)", "scipy": "SciPy", "cc3d": "cc3d", "stream": "ufccl streaming",
        "tiled": "ufccl tiled"}

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
W = 6.9  # text width in inches


def J(name):
    with open(os.path.join(PROJ, "results", name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=170, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def panel(ax, tag):
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right")


def label_cmap(n, seed=3, dark_bg=False):
    rng = np.random.default_rng(seed)
    base = plt.get_cmap("tab20")(np.arange(20))[:, :3]
    cols = [base[i % 20] for i in rng.permutation(max(n, 1))]
    bg = (0.09, 0.09, 0.11) if dark_bg else (1, 1, 1)
    return ListedColormap([bg] + cols)


# ------------------------------------------------------------------ causal
def fig_causal():
    from matplotlib.patches import Rectangle
    from ufccl.label import causal_offsets
    fig = plt.figure(figsize=(W, 2.15))
    gw, gh = 0.088, 0.46          # one 3x3 grid, figure fraction
    gap_in, gap_out = 0.012, 0.045
    x0 = 0.012
    layer_names = ["slice $z{-}1$", "slice $z$", "slice $z{+}1$"]
    for g, conn in enumerate((6, 18, 26)):
        nnz = {6: 1, 18: 2, 26: 3}[conn]
        causal = {tuple(o) for o in causal_offsets(conn)}
        n_all = sum(1 for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if 0 < (dz != 0) + (dy != 0) + (dx != 0) <= nnz)
        gx = x0 + g * (3 * gw + 2 * gap_in + gap_out)
        for l, dz in enumerate((-1, 0, 1)):
            ax = fig.add_axes([gx + l * (gw + gap_in), 0.20, gw, gh])
            ax.set_xlim(0, 3)
            ax.set_ylim(3, 0)
            ax.set_aspect("equal")
            ax.axis("off")
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    c = (dz != 0) + (dy != 0) + (dx != 0)
                    if c == 0:
                        fc, ec = RED, "k"
                    elif c <= nnz and (dz, dy, dx) in causal:
                        fc, ec = "#3B6EA8", "k"
                    elif c <= nnz:
                        fc, ec = "#DADADA", "k"
                    else:
                        fc, ec = "white", "#BBBBBB"
                    ax.add_patch(Rectangle((dx + 1, dy + 1), 1, 1, fc=fc, ec=ec, lw=0.7))
                    if c == 0:
                        ax.text(dx + 1.5, dy + 1.55, "v", ha="center", va="center", color="white",
                                fontsize=8, fontweight="bold", style="italic")
            ax.text(1.5, 3.42, layer_names[l], ha="center", va="top", fontsize=6.6)
        fig.text(gx + 1.5 * gw + gap_in, 0.80, f"{conn}-connectivity", ha="center", fontsize=9.5,
                 fontweight="bold")
        fig.text(gx + 1.5 * gw + gap_in, 0.70, f"{len(causal)} causal of {n_all} neighbours",
                 ha="center", fontsize=7.8)
    fig.legend(handles=[Patch(fc=RED, ec="k", lw=0.5, label="current voxel $v=(z,y,x)$"),
                        Patch(fc="#3B6EA8", ec="k", lw=0.5, label="causal neighbour (already labelled)"),
                        Patch(fc="#DADADA", ec="k", lw=0.5, label="later neighbour (not yet visited)"),
                        Patch(fc="white", ec="#BBBBBB", lw=0.5, label="not a neighbour")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=7.2,
               handlelength=1.2, columnspacing=1.2)
    save(fig, "fig_causal")


# --------------------------------------------------------------- streaming
def box(ax, x, y, w, h, text, fc="#F4F4F4", ec="#555555", fs=7.6, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=0.8)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, p, q, color="#333333", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def fig_streaming(task_dir, case):
    ct, spacing, _ = load_nifti(os.path.join(task_dir, "imagesTr", case + ".nii.gz"), dtype=np.float32)
    hu = hu_window_mask(ct, -200.0, 300.0)
    del ct
    table, info = stream_components(iter(hu), 6, return_info=True)
    k = np.array(info["open_per_slice"])
    closed = np.array(info["closed_per_slice"])
    total_seen = np.cumsum(closed) + k
    z = np.arange(len(k))

    fig = plt.figure(figsize=(W, 3.0))
    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    box(ax, 0.2, 8.15, 4.0, 1.1, "slice $z$ of the mask\n$Y\\times X$ (one read)", fc="#EAF1F8", ec=COL["scipy"])
    box(ax, 5.6, 8.15, 4.0, 1.1, "frontier labels of slice $z-1$\nids $1..K$, stats rows", fc="#EAF1F8", ec=COL["scipy"])
    box(ax, 0.2, 6.1, 9.4, 1.35, "1  scan: new labels $K{+}1..M$ for voxels with no causal neighbour;\n"
        "union equivalent labels; add each voxel to its label's stats row", bold=False)
    box(ax, 0.2, 4.35, 9.4, 1.1, "2  fold every non-root stats row into its Union-Find root\n"
        "(add counts and sums, min/max of box and first index)")
    box(ax, 0.2, 2.6, 4.5, 1.2, "3a  root carried by no voxel\nof slice $z$: closed", fc="#FBEAEA", ec=RED)
    box(ax, 5.1, 2.6, 4.5, 1.2, "3b  root still on the\nfrontier: stays open", fc="#EAF5EE", ec=COL["cc3d"])
    box(ax, 0.2, 0.55, 4.5, 1.2, "emit stats row,\nfree the slot", fc="#FBEAEA", ec=RED, bold=True)
    box(ax, 5.1, 0.55, 4.5, 1.2, "renumber open roots $1..K'$,\nremap slice, swap buffers", fc="#EAF5EE", ec=COL["cc3d"], bold=True)
    arrow(ax, (2.2, 8.15), (2.2, 7.47))
    arrow(ax, (7.6, 8.15), (7.6, 7.47))
    arrow(ax, (4.9, 6.1), (4.9, 5.47))
    arrow(ax, (3.0, 4.35), (2.4, 3.82))
    arrow(ax, (6.8, 4.35), (7.4, 3.82))
    arrow(ax, (2.4, 2.6), (2.4, 1.77), color=RED)
    arrow(ax, (7.4, 2.6), (7.4, 1.77), color=COL["cc3d"])
    ax.annotate("", xy=(9.9, 8.7), xytext=(9.9, 1.15), arrowprops=dict(arrowstyle="-|>", lw=0.9,
                color=COL["cc3d"], connectionstyle="arc3,rad=-0.0", ls="--"))
    ax.text(10.05, 5, "next slice", rotation=90, va="center", fontsize=7.2, color=COL["cc3d"])

    ax2 = fig.add_axes([0.655, 0.17, 0.335, 0.74])
    ax2.fill_between(z, total_seen, color="#D9D9D9", label="components seen so far")
    ax2.plot(z, total_seen, color="#8A8A8A", lw=0.8)
    ax2.plot(z, k, color=RED, lw=1.5, label="open (kept in memory)")
    ax2.set_xlabel("slice index $z$")
    ax2.set_ylabel("components")
    ax2.set_title("one CT scan (%s)" % case.replace("_", " "), fontsize=8.5)
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_xlim(0, len(k))
    ax2.set_ylim(0, total_seen.max() * 1.05)
    fig.text(0.0, 1.0, "(a)", fontsize=10, fontweight="bold", va="bottom")
    fig.text(0.60, 1.0, "(b)", fontsize=10, fontweight="bold", va="bottom")
    save(fig, "fig_streaming")
    return dict(peak_open=int(k.max()), total=len(table), slices=len(k))


# ------------------------------------------------------------------- tiled
def fig_tiled():
    n = 24
    yy, xx = np.mgrid[:n, :n]
    m = np.zeros((n, n), bool)
    m |= ((xx - 11.5) / 6.5) ** 2 + ((yy - 6) / 3.2) ** 2 <= 1          # crosses vertical seam
    m |= ((xx - 15) / 3.4) ** 2 + ((yy - 16.5) / 5.6) ** 2 <= 1          # crosses both seams
    m |= ((xx - 4) / 2.8) ** 2 + ((yy - 18) / 2.8) ** 2 <= 1             # stays in one tile
    m |= ((xx - 20) / 1.6) ** 2 + ((yy - 4) / 1.6) ** 2 <= 1             # stays in one tile
    t = n // 2
    local = np.zeros((n, n), int)
    base = 0
    for ty in (0, 1):
        for tx in (0, 1):
            blk = m[ty * t:(ty + 1) * t, tx * t:(tx + 1) * t][None]
            r = label(blk, 6)
            lab = r.labels[0]
            local[ty * t:(ty + 1) * t, tx * t:(tx + 1) * t] = np.where(lab > 0, lab + base, 0)
            base += r.n
    glob = label(m[None], 6).labels[0]
    n_local, n_glob = base, int(glob.max())
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.45))
    cm1 = label_cmap(n_local, seed=11)
    cm2 = label_cmap(n_glob, seed=5)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
    axs[0].imshow(m, cmap=ListedColormap(["white", "#444444"]), interpolation="nearest")
    axs[0].set_title("input mask, 2×2 tiles")
    axs[1].imshow(local, cmap=cm1, interpolation="nearest", vmin=0, vmax=n_local)
    axs[1].set_title(f"step 1: tiles labelled alone\n({n_local} local components)")
    axs[2].imshow(glob, cmap=cm2, interpolation="nearest", vmin=0, vmax=n_glob)
    axs[2].set_title(f"step 2–3: seams merged\n({n_glob} components)")
    for ax in axs:
        ax.axvline(t - 0.5, color="k", lw=1.0, ls=(0, (4, 2)))
        ax.axhline(t - 0.5, color="k", lw=1.0, ls=(0, (4, 2)))
    # halo band drawn on the middle panel
    axs[1].axvspan(t - 1.5, t + 0.5, color=RED, alpha=0.13, lw=0)
    axs[1].axhspan(t - 1.5, t + 0.5, color=RED, alpha=0.13, lw=0)
    axs[1].text(t + 0.9, n - 0.4, "halo", color=RED, fontsize=7.5, va="bottom")
    for ax, tag in zip(axs, "abc"):
        panel(ax, f"({tag})")
    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.84, bottom=0.01)
    save(fig, "fig_tiled")
    return n_local, n_glob


# ---------------------------------------------------------------- phantoms
def fig_phantoms():
    n = 128
    blobs = stack(blob_slices((n, n, n), 16, 0.60, 0))
    les = lesion_phantom((n, n, n), 30, (3, 12), 3, seed=0)
    noise = stack(noise_slices((n, n, n), 0.31, 0))
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.5))
    items = [(blobs, "blobs  (21 % foreground)"), (les, "lesions + vessels  (2.4 %)"),
             (noise, "noise at $p=0.31$  (percolation)")]
    for ax, (vol, title) in zip(axs, items):
        r = label(vol, 6)
        counts = np.bincount(r.labels.ravel())[1:]
        zc = n // 2
        if vol is les:
            zc = int(np.argmax(vol.sum(axis=(1, 2))))
        sl = r.labels[zc]
        cm = label_cmap(r.n, seed=7, dark_bg=False)
        ax.imshow(sl, cmap=cm, vmin=0, vmax=r.n, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{title}\n{r.n:,} components in the $128^3$ volume", fontsize=8.3)
        for s in ax.spines.values():
            s.set_visible(True)
    for ax, tag in zip(axs, "abc"):
        panel(ax, f"({tag})")
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.82, bottom=0.01)
    save(fig, "fig_phantoms")


# ------------------------------------------------------------------ real CT
def ct_window(a, lo=-1000, hi=400):
    return np.clip((a - lo) / (hi - lo), 0, 1)


def vivid_cmap(n):
    cols = plt.get_cmap("tab20")(np.arange(20))
    sat = [cols[i] for i in (0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 17, 19)]
    return ListedColormap([sat[i % len(sat)] for i in range(n)])


def rank_components(lab, keep=60):
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    order = np.argsort(-cnt)[:keep]
    rank = np.zeros(len(cnt), int)
    for i, lid in enumerate(order, 1):
        if cnt[lid] > 0:
            rank[lid] = i
    return rank


def overlay_components(ax, ct_slice, lab_slice, rank, **kw):
    """Largest component (the body) as a dim wash, all others in vivid colour."""
    ax.imshow(ct_window(ct_slice) * 0.55, cmap="gray", vmin=0, vmax=1, **kw)
    r = rank[lab_slice]
    body = np.ma.masked_where(r != 1, r)
    ax.imshow(body, cmap=ListedColormap(["#5D8AA8"]), alpha=0.28, vmin=0, vmax=1, **kw)
    rest = np.ma.masked_where(r < 2, r - 2)
    ax.imshow(rest, cmap=vivid_cmap(40), vmin=0, vmax=39, alpha=1.0, **kw)


def fig_real_ct(task_dir, case):
    ct, sp, _ = load_nifti(os.path.join(task_dir, "imagesTr", case + ".nii.gz"), dtype=np.float32)
    lab_vol, _, _ = load_nifti(os.path.join(task_dir, "labelsTr", case + ".nii.gz"))
    tum = label(lab_vol == 1, 6, stats=True, spacing=sp)
    t = tum.table
    big = int(np.argmax(t.voxels))
    zc, yc, xc = (int(round(v)) for v in t.centroid_vox[big])
    hu = hu_window_mask(ct, -200.0, 300.0)
    full = label(hu, 6)
    rank = rank_components(full.labels)
    kw = dict(origin="lower", interpolation="nearest")
    Z, Y, X = ct.shape
    fig = plt.figure(figsize=(W, 2.2))
    gs = fig.add_gridspec(1, 3, wspace=0.03, left=0.005, right=0.995, top=0.84, bottom=0.005)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    ext_ax = dict(origin="lower", interpolation="bilinear")
    ax = axs[0]
    ax.imshow(ct_window(ct[zc]), cmap="gray", vmin=0, vmax=1, **ext_ax)
    ov = np.ma.masked_where(tum.labels[zc] == 0, tum.labels[zc])
    ax.imshow(ov, cmap=ListedColormap([RED]), alpha=0.8, vmin=0, vmax=1, **ext_ax)
    ax = axs[1]
    asp = sp[0] / sp[1]
    ax.imshow(ct_window(ct[:, yc, :]), cmap="gray", aspect=asp, vmin=0, vmax=1, **ext_ax)
    ov = np.ma.masked_where(tum.labels[:, yc, :] == 0, tum.labels[:, yc, :])
    ax.imshow(ov, cmap=ListedColormap([RED]), alpha=0.8, aspect=asp, vmin=0, vmax=1, **ext_ax)
    ax = axs[2]
    # axial maximum-id projection of all tumour components, each in its own colour and numbered
    proj = tum.labels.max(axis=0)
    ax.imshow(ct_window(ct[zc]) * 0.6, cmap="gray", vmin=0, vmax=1, **ext_ax)
    ax.imshow(np.ma.masked_where(proj == 0, proj - 1), cmap=vivid_cmap(max(len(t), 1)), vmin=0,
              vmax=max(len(t) - 1, 1), interpolation="nearest", origin="lower")
    # equal physical height for all three panels, centred on the body
    body = (rank[full.labels[zc]] == 1).any(axis=1)
    rows = np.flatnonzero(body)
    ymid = 0.5 * (rows.min() + rows.max())
    half = 0.5 * Z * sp[0] / sp[1]
    axs[0].set_ylim(ymid - half, ymid + half)
    # panel (c): zoom on the largest tumour, same aspect ratio as the other panels
    ratio = (Z * sp[0]) / (X * sp[2])
    cy0, cx0 = t.centroid_vox[big][1], t.centroid_vox[big][2]
    axs[2].set_xlim(cx0 - 62, cx0 + 62)
    axs[2].set_ylim(cy0 - 62 * ratio, cy0 + 62 * ratio)
    bar_px = 20.0 / sp[2]
    axs[2].plot([cx0 + 56 - bar_px, cx0 + 56], [cy0 - 62 * ratio + 6] * 2, color="white", lw=2)
    axs[2].text(cx0 + 56 - bar_px / 2, cy0 - 62 * ratio + 9, "20 mm", color="white", fontsize=6.5,
                ha="center", va="bottom")
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        a.invert_xaxis()      # RAS+: patient's right on the image left
        a.set_anchor('N')
        for sp_ in a.spines.values():
            sp_.set_visible(False)
    titles = ["axial CT with tumour annotation", "coronal CT with tumour annotation",
              "%d labelled tumour components\n(zoom, projection over z)" % len(t)]
    for xx, tt in zip((0.18, 0.5, 0.82), titles):
        fig.text(xx, 0.995, tt, ha="center", va="top", fontsize=8.3, linespacing=1.15)
    save(fig, "fig_real_ct")
    # banner for the cover page
    zs = np.linspace(int(Z * 0.25), int(Z * 0.8), 4).astype(int)
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.8))
    for a, z in zip(axs, zs):
        overlay_components(a, ct[z], full.labels[z], rank, **kw)
        a.axis("off")
        a.invert_xaxis()
    fig.subplots_adjust(wspace=0.01, left=0, right=1, top=1, bottom=0)
    fig.savefig(os.path.join(OUT, "banner.png"), dpi=220, facecolor="black")
    plt.close(fig)
    return dict(case=case, n_tumour=len(t), n_hu=int(full.n), shape=list(ct.shape))


# ------------------------------------------------------------ result plots
def fig_scaling():
    rows = J("scaling.json")["rows"]
    kinds = ["blobs", "lesions", "noise"]
    titles = {"blobs": "blobs (21 % fg)", "lesions": "lesions (2.5 % fg)", "noise": "noise (31 % fg)"}
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.45), sharey=True)
    for ax, kind in zip(axs, kinds):
        rr = [r for r in rows if r["kind"] == kind]
        N = np.array([r["voxels"] for r in rr], float)
        series = [("ufccl", [r["conn6"]["ufccl_s"] for r in rr], "-", "o"),
                  ("scipy", [r["conn6"]["scipy_s"] for r in rr], "-", "s"),
                  ("cc3d", [r["conn6"]["cc3d_s"] for r in rr], "-", "^"),
                  ("stream", [r["stream6_s"] for r in rr], "--", "D"),
                  ("tiled", [r["tiled6_s"] for r in rr], "--", "v")]
        for key, y, ls, mk in series:
            ax.loglog(N, y, ls, color=COL[key], marker=mk, ms=3.2, label=NAME[key])
        ref = rr[-1]["conn6"]["ufccl_s"] * N / N[-1]
        ax.loglog(N, ref, ":", color="k", lw=0.9, label="linear in $N$")
        ax.set_title(titles[kind])
        ax.set_xlabel("voxels $N$")
    axs[0].set_ylabel("wall-clock time (s)")
    axs[0].legend(loc="upper left", frameon=False, fontsize=7)
    for ax, tag in zip(axs, "abc"):
        panel(ax, f"({tag})")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_scaling")


def fig_speed_conn():
    rows = J("scaling.json")["rows"]
    big = [r for r in rows if r["n"] == 512]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    keys = ["ufccl", "scipy", "cc3d", "stream", "tiled"]
    getters = {"ufccl": lambda r: r["conn6"]["ufccl_s"], "scipy": lambda r: r["conn6"]["scipy_s"],
               "cc3d": lambda r: r["conn6"]["cc3d_s"], "stream": lambda r: r["stream6_s"],
               "tiled": lambda r: r["tiled6_s"]}
    x = np.arange(len(big))
    w = 0.16
    for i, k in enumerate(keys):
        y = [r["voxels"] / getters[k](r) / 1e6 for r in big]
        b = ax.bar(x + (i - 2) * w, y, w, color=COL[k], label=NAME[k], zorder=3)
        for xi, yi in zip(x + (i - 2) * w, y):
            ax.text(xi, yi + 3, f"{yi:.0f}", ha="center", fontsize=6, rotation=90, va="bottom")
    ax.set_xticks(x, [r["kind"] for r in big])
    ax.set_ylabel("throughput (Mvoxel/s)")
    ax.set_title("$512^3$ volumes, 6-connectivity", fontsize=8.8)
    ax.set_ylim(0, 330)
    ax.legend(ncol=2, frameon=False, fontsize=6.8, loc="upper right")
    ax.grid(axis="x", visible=False)
    panel(ax, "(a)")
    ax = axs[1]
    rr = [r for r in rows if r["kind"] == "blobs"]
    N = [r["voxels"] for r in rr]
    for c, ls, mk in zip((6, 18, 26), ("-", "--", ":"), ("o", "s", "^")):
        ax.loglog(N, [r[f"conn{c}"]["ufccl_s"] for r in rr], ls, color=COL["ufccl"], marker=mk, ms=3,
                  label=f"ufccl {c}")
        ax.loglog(N, [r[f"conn{c}"]["cc3d_s"] for r in rr], ls, color=COL["cc3d"], marker=mk, ms=3,
                  label=f"cc3d {c}")
    ax.set_xlabel("voxels $N$")
    ax.set_ylabel("time (s)")
    ax.set_title("connectivity ablation (blobs)", fontsize=8.8)
    ax.legend(ncol=2, frameon=False, fontsize=6.8, loc="upper left")
    panel(ax, "(b)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_speed_conn")


def fig_memory():
    rows = J("memory.json")["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
    ax = axs[0]
    cube = [r for r in rows if r["experiment"] == "cube"]
    spec = [("ufccl", "ufccl", "-", "o"), ("scipy", "scipy", "-", "s"), ("cc3d", "cc3d", "-", "^"),
            ("tiled", "tiled", "--", "v"), ("stream", "stream", "--", "D")]
    for v, k, ls, mk in spec:
        rr = [r for r in cube if r["variant"] == v]
        ax.loglog([r["voxels"] for r in rr], [max(r["delta_mb"], 0.8) for r in rr], ls, color=COL[k],
                  marker=mk, ms=3.5, label=NAME[k] + (" (labels on disk)" if v == "tiled" else ""))
    ax.set_xlabel("voxels $N$")
    ax.set_ylabel("peak extra memory (MB)")
    ax.set_title("cubes, blobs", fontsize=8.8)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    panel(ax, "(a)")
    ax = axs[1]
    dep = [r for r in rows if r["experiment"] == "depth"]
    for v, k in (("ufccl", "ufccl"), ("stream", "stream")):
        rr = [r for r in dep if r["variant"] == v]
        ax.semilogx([r["shape"][0] for r in rr], [r["delta_mb"] for r in rr], "-o", color=COL[k], ms=3.5,
                    label=NAME[k] + (" in memory" if v == "ufccl" else ""))
    ax.set_xlabel("depth $Z$ (slices of $256\\times256$)")
    ax.set_ylabel("peak extra memory (MB)")
    ax.set_title("fixed slice size, growing depth", fontsize=8.8)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.annotate("grows with the output\ntable (133 k lesions)", xy=(8192, 36), xytext=(1400, 190),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7), color="#333333")
    panel(ax, "(b)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_memory")


def fig_uf():
    d = J("complexity.json")
    sc = d["scaling"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.45))
    ax = axs[0]
    spec = [("blobs", None, "#1f77b4", "blobs"), ("noise", None, RED, "noise"), ("snake", None, "#2ca02c", "snake"),
            ("comb", "halving+size", COL["tiled"], "comb")]
    for kind, cfg, c, nm in spec:
        rr = [r for r in sc if r["kind"] == kind and r.get("config") == cfg]
        ax.semilogx([r["voxels"] for r in rr], [(r["steps1"] + r["steps2"]) / r["voxels"] for r in rr],
                    "-o", ms=2.8, color=c, label=nm)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("voxels $N$")
    ax.set_ylabel("pointer steps per voxel")
    ax.set_title("halving + union by size", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
    panel(ax, "(a)")
    ax = axs[1]
    for cfg, c, ls in (("none+naive", RED, "-"), ("halving+size", COL["cc3d"], "-")):
        rr = [r for r in sc if r["kind"] == "comb" and r.get("config") == cfg]
        ax.loglog([r["voxels"] for r in rr], [max((r["steps1"] + r["steps2"]), 1) for r in rr], ls, marker="o",
                  ms=3, color=c, label=cfg.replace("+", " + "))
    xs = np.array([1.5e3, 5e4])
    ax.loglog(xs, xs ** 2 / 1e3 * 0.9, ":", color="k", lw=0.9)
    ax.text(2.2e4, 9e3, "$\\propto N^{2}$", fontsize=7.5)
    ax.set_xlabel("voxels $N$ (comb input)")
    ax.set_ylabel("total pointer steps")
    ax.set_title("adversarial comb input", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    panel(ax, "(b)")
    ax = axs[2]
    ab = [r for r in d["ablation"] if r["input"] == "blobs"]
    names = [("none", "size"), ("none", "naive"), ("halving", "size"), ("halving", "naive"), ("full", "size"), ("full", "naive")]
    vals = []
    for comp, link in names:
        r = [x for x in ab if x["compress"] == comp and x["link"] == link][0]
        vals.append((r["steps1"] + r["steps2"]) / r["finds1"])
    x = np.arange(len(names))
    colors = [COL["scipy"] if n[1] == "size" else "#C2C2C2" for n in names]
    ax.bar(x, vals, color=colors, edgecolor="k", lw=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{a}\n{b}" for a, b in names], fontsize=6.4)
    ax.set_ylabel("steps per find call")
    ax.set_title("ablation on blobs ($96^3$)", fontsize=8.6)
    ax.grid(axis="x", visible=False)
    panel(ax, "(c)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_uf")


def fig_real_perf():
    d = J("real.json")
    cs = d["cases"]
    vox = np.array([c["voxels"] for c in cs], float) / 1e6
    fig = plt.figure(figsize=(W, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.85, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    for key, k in (("hu_scipy_s", "scipy"), ("hu_cc3d_s", "cc3d"), ("hu_ufccl_s", "ufccl"),
                   ("hu_stream_s", "stream"), ("hu_tiled_s", "tiled")):
        y = np.array([c[key] for c in cs])
        ax.scatter(vox, y, s=7, color=COL[k], label=NAME[k], alpha=0.85, lw=0)
        slope = (vox * y).sum() / (vox * vox).sum()
        ax.plot([0, vox.max()], [0, slope * vox.max()], color=COL[k], lw=0.8, alpha=0.8)
    ax.set_xlabel("scan size (Mvoxel)")
    ax.set_ylabel("time per scan (s)")
    ax.set_title("63 CT scans, soft-tissue mask", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", markerscale=1.4)
    panel(ax, "(a)")
    ax = fig.add_subplot(gs[1])
    r1 = np.array([c["hu_scipy_s"] / c["hu_ufccl_s"] for c in cs])
    r2 = np.array([c["hu_ufccl_s"] / c["hu_cc3d_s"] for c in cs])
    bp = ax.boxplot([r1, r2], widths=0.55, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", ms=2, mfc="none", mec="#555555"),
                    medianprops=dict(color="k", lw=1.2))
    for patch, c in zip(bp["boxes"], (COL["scipy"], COL["cc3d"])):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xticks([1, 2], ["SciPy time\n/ ufccl time", "ufccl time\n/ cc3d time"], fontsize=6.8)
    ax.set_ylabel("per-scan time ratio")
    ax.set_title("relative speed", fontsize=8.6)
    ax.grid(axis="x", visible=False)
    panel(ax, "(b)")
    ax = fig.add_subplot(gs[2])
    mem = d["memory"]
    cases = list(dict.fromkeys(m["case"] for m in mem))
    variants = ["ufccl", "scipy", "cc3d", "tiled", "stream"]
    x = np.arange(len(cases))
    w = 0.16
    for i, v in enumerate(variants):
        y = [[m for m in mem if m["case"] == c and m["variant"] == v][0]["delta_mb"] for c in cases]
        ax.bar(x + (i - 2) * w, y, w, color=COL[v], label=NAME[v], zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x, [c.replace("lung_", "#") for c in cases])
    ax.set_ylabel("peak extra memory (MB)")
    ax.set_title("3 largest scans", fontsize=8.6)
    ax.grid(axis="x", visible=False)
    panel(ax, "(c)")
    save(fig, "fig_real_perf")


def fig_real_lesions():
    rows = list(csv.DictReader(open(os.path.join(PROJ, "results", "real_lesions.csv"))))
    vol = np.array([float(r["volume_mm3"]) for r in rows])
    d = J("real.json")["cases"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35))
    ax = axs[0]
    bins = np.logspace(0, np.log10(vol.max() * 1.2), 22)
    ax.hist(vol, bins=bins, color=RED, alpha=0.85, edgecolor="white", lw=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("component volume (mm$^3$)")
    ax.set_ylabel("components")
    ax.set_title("164 tumour components", fontsize=8.6)
    ax.axvline(np.median(vol), color="k", ls=":", lw=0.9)
    ax.text(np.median(vol) * 1.5, ax.get_ylim()[1] * 0.9, "median\n3 mm$^3$", fontsize=7)
    panel(ax, "(a)")
    ax = axs[1]
    thr = np.logspace(0, 5.6, 60)
    kept = [(vol >= t).sum() for t in thr]
    scans = []
    cases = [r["case"] for r in rows]
    cases_arr = np.array(cases)
    for t in thr:
        scans.append(len(set(cases_arr[vol >= t])))
    ax.semilogx(thr, kept, color=RED, label="components kept")
    ax.semilogx(thr, scans, color=COL["scipy"], label="scans with $\\geq$1 kept")
    ax.set_xlabel("minimum volume (mm$^3$)")
    ax.set_ylabel("count")
    ax.set_title("size filter", fontsize=8.6)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    panel(ax, "(b)")
    ax = axs[2]
    ks = {c: [x[f"tumour_c{c}_n"] for x in d] for c in (6, 18, 26)}
    tot = [sum(ks[c]) for c in (6, 18, 26)]
    b = ax.bar([0, 1, 2], tot, color=[RED, "#D8776B", "#E9B1AA"], edgecolor="k", lw=0.4, zorder=3)
    for xi, t in zip([0, 1, 2], tot):
        ax.text(xi, t + 2, str(t), ha="center", fontsize=7.5)
    ax.set_xticks([0, 1, 2], ["6", "18", "26"])
    ax.set_xlabel("connectivity")
    ax.set_ylabel("components in 63 scans")
    ax.set_ylim(0, 190)
    ax.set_title("effect of connectivity", fontsize=8.6)
    ax.grid(axis="x", visible=False)
    panel(ax, "(c)")
    fig.tight_layout(w_pad=0.9)
    save(fig, "fig_real_lesions")


def fig_app():
    d = J("app.json")
    s = d["summary"]
    thr = sorted(s, key=float)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.3))
    fp = np.array([s[t]["fp_per_scan_mean"] for t in thr])
    se = np.array([s[t]["sensitivity_mean"] for t in thr])
    pr = np.array([s[t]["pred_lesions_mean"] for t in thr])
    x = np.arange(len(thr))
    ax = axs[0]
    ax.bar(x, pr, color="#D6D6D6", edgecolor="k", lw=0.4, label="predicted components", zorder=3)
    ax.bar(x, fp, color=RED, edgecolor="k", lw=0.4, label="of which false positives", zorder=4)
    ax.axhline(s[thr[0]]["gt_lesions_mean"], color="k", ls=":", lw=0.9)
    ax.text(len(thr) - 0.6, s[thr[0]]["gt_lesions_mean"] + 18, "true lesions (17.7)", fontsize=7, ha="right")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(x, thr)
    ax.set_xlabel("minimum lesion volume (mm$^3$)")
    ax.set_ylabel("components per scan")
    ax.set_title("size filtering of a noisy segmentation", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    ax.grid(axis="x", visible=False)
    panel(ax, "(a)")
    ax = axs[1]
    ax.plot(fp, se, "-o", color=COL["scipy"], ms=3.5)
    for t, a, b in zip(thr, fp, se):
        if t in ("0", "5", "80", "160"):
            ax.annotate(t, (a, b), textcoords="offset points", xytext=(4, -9), fontsize=6.6)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("false positives per scan")
    ax.set_ylabel("lesion-wise sensitivity")
    ax.set_ylim(0.85, 0.95)
    ax.set_title("operating points (labels: min. volume)", fontsize=8.6)
    panel(ax, "(b)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_app")


if __name__ == "__main__":
    task = sys.argv[1]
    real = J("real.json")["cases"]
    case = max(real, key=lambda c: (c["tumour_c6_n"], -c["voxels"]))["case"]
    print("showcase case:", case)
    info = {}
    fig_causal()
    info["tiled"] = fig_tiled()
    fig_phantoms()
    info["ct"] = fig_real_ct(task, case)
    mid = sorted(real, key=lambda c: c["voxels"])[len(real) // 2]["case"]
    info["stream"] = fig_streaming(task, mid)
    fig_scaling()
    fig_speed_conn()
    fig_memory()
    fig_uf()
    fig_real_perf()
    fig_real_lesions()
    fig_app()
    print(json.dumps(info))
