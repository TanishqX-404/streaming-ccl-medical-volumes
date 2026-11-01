"""Benchmark driver.

    python bench/run_bench.py correctness|scaling|memory|complexity|app|all

Each section writes a JSON file to results/ (see plot_results.py for figures
and the markdown summary).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from common import (DATA, RESULTS, ROOT, best_of, hardware, save_json,
                    slices_for, volume_array, volume_file)

import cc3d  # noqa: E402
import scipy.ndimage as ndi  # noqa: E402

from ufccl import (label, label_tiled, same_partition, stream_components)  # noqa: E402
from ufccl.instrumented import label_counted  # noqa: E402
from ufccl.synth import (checkerboard, comb, lesion_phantom, random_mask, snake3d)  # noqa: E402

STRUCT = {6: 1, 18: 2, 26: 3}


def sp_label(m, conn):
    return ndi.label(m, ndi.generate_binary_structure(3, STRUCT[conn]))


# ---------------------------------------------------------------- correctness
def correctness():
    rng = np.random.default_rng(2024)
    cases = []
    for i in range(120):
        shape = tuple(int(s) for s in rng.integers(1, 40, 3))
        p = float(rng.choice([0.05, 0.15, 0.3, 0.45, 0.7]))
        cases.append((f"random{i}", random_mask(shape, p, seed=i)))
    cases += [("checkerboard", checkerboard((24, 25, 26))),
              ("snake3d", snake3d(15, 41, 41)),
              ("empty", np.zeros((10, 11, 12), bool)), ("full", np.ones((10, 11, 12), bool))]
    for s in range(4):
        cases.append((f"lesions{s}", lesion_phantom((48, 64, 64), 30, (3, 9), 4, seed=s)))
    cases.append(("blobs64", volume_array("blobs", (64, 64, 64), 1)))
    cases.append(("noise48", volume_array("noise", (48, 48, 48), 1)))

    checks = ["scipy_exact", "cc3d_partition", "stream_table", "tiled_exact", "stats_vs_scipy"]
    tot = {c: [0, 0] for c in checks}
    per_conn = {6: [0, 0], 18: [0, 0], 26: [0, 0]}
    components = 0
    voxels = 0
    for name, m in cases:
        for conn in (6, 18, 26):
            ref, n_ref = sp_label(m, conn)
            r = label(m, conn, stats=True)
            ok = {}
            ok["scipy_exact"] = r.n == n_ref and np.array_equal(r.labels, ref)
            ok["cc3d_partition"] = same_partition(
                r.labels, cc3d.connected_components(m, connectivity=conn))
            ok["stream_table"] = stream_components(iter(m), conn).same_as(r.table)
            tile = tuple(int(t) for t in rng.integers(1, 20, 3))
            lab_t, n_t, tab_t = label_tiled(m, tile, conn)
            ok["tiled_exact"] = n_t == r.n and np.array_equal(lab_t, r.labels) and tab_t.same_as(r.table)
            if n_ref:
                idx = np.arange(1, n_ref + 1)
                ok["stats_vs_scipy"] = bool(
                    np.array_equal(r.table.voxels, ndi.sum(m, ref, idx).astype(int))
                    and np.allclose(r.table.centroid_vox, np.array(ndi.center_of_mass(m, ref, idx))))
            else:
                ok["stats_vs_scipy"] = len(r.table) == 0
            for c in checks:
                tot[c][0] += int(ok[c])
                tot[c][1] += 1
            per_conn[conn][0] += int(all(ok.values()))
            per_conn[conn][1] += 1
            components += n_ref
            voxels += int(m.sum())
            if not all(ok.values()):
                print("MISMATCH", name, conn, ok, flush=True)
    out = {"volumes": len(cases), "comparisons": len(cases) * 3,
           "checks": {c: {"passed": a, "total": b} for c, (a, b) in tot.items()},
           "per_connectivity": {str(k): {"passed": a, "total": b} for k, (a, b) in per_conn.items()},
           "components_checked": int(components), "foreground_voxels_checked": voxels}
    save_json("correctness.json", out)
    print(json.dumps(out, indent=1))


# -------------------------------------------------------------------- scaling
SIZES = [64, 96, 128, 192, 256, 384, 512]
KINDS = ["blobs", "lesions", "noise"]


def scaling(sizes=SIZES, kinds=KINDS):
    res = []
    label(np.ones((4, 4, 4), bool), 6, stats=True)
    label(np.ones((4, 4, 4), bool), 18)
    label(np.ones((4, 4, 4), bool), 26)
    stream_components(iter(np.ones((3, 4, 4), bool)), 6)
    label_tiled(np.ones((4, 4, 4), bool), (2, 2, 2), 6)
    for kind in kinds:
        for n in sizes:
            shape = (n, n, n)
            m = volume_array(kind, shape, seed=0)
            reps = 5 if n <= 192 else 3
            row = {"kind": kind, "n": n, "voxels": int(m.size), "fg_fraction": float(m.mean())}
            ref6 = None
            for conn in (6, 18, 26):
                st = {6: 1, 18: 2, 26: 3}[conn]
                s_sp, (lab_sp, n_sp) = best_of(
                    lambda: ndi.label(m, ndi.generate_binary_structure(3, st)), reps)
                s_cc, (lab_cc, n_cc) = best_of(
                    lambda: cc3d.connected_components(m, connectivity=conn, return_N=True), reps)
                s_uf, r = best_of(lambda: label(m, conn), reps)
                row[f"conn{conn}"] = {
                    "ufccl_s": s_uf, "scipy_s": s_sp, "cc3d_s": s_cc,
                    "components": int(r.n),
                    "exact_vs_scipy": bool(r.n == n_sp and np.array_equal(r.labels, lab_sp)),
                    "counts_agree_cc3d": bool(n_cc == r.n)}
                if conn == 6:
                    ref6 = r
                del lab_sp, lab_cc, r
            s_stats, r_stats = best_of(lambda: label(m, 6, stats=True), reps)
            row["ufccl6_stats_s"] = s_stats
            # post-hoc statistics: separate pass over the finished label volume
            s_post, _ = best_of(lambda: _posthoc_stats(ref6.labels, ref6.n), reps)
            row["posthoc_stats_s"] = s_post
            s_st, tab = best_of(lambda: stream_components(iter(m), 6), reps)
            row["stream6_s"] = s_st
            row["stream6_matches"] = bool(tab.same_as(r_stats.table))
            tile = (128, 128, 128)
            s_ti, (lab_t, n_t, tab_t) = best_of(lambda: label_tiled(m, tile, 6), max(1, reps - 2))
            row["tiled6_s"] = s_ti
            row["tiled6_matches"] = bool(n_t == ref6.n and np.array_equal(lab_t, ref6.labels)
                                         and tab_t.same_as(r_stats.table))
            row["tiles"] = int(np.prod([(n + 127) // 128] * 3))
            res.append(row)
            print(kind, n, {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in row.items() if k.endswith("_s")}, flush=True)
            del m, ref6, lab_t, r_stats
    save_json("scaling.json", {"hardware": hardware(), "rows": res})


def _posthoc_stats(lab, n):
    """Reference: statistics from a second sweep over the label volume."""
    return _posthoc_kernel(lab, n)


from numba import njit  # noqa: E402
from ufccl.unionfind import stat_add, stats_new  # noqa: E402


@njit(cache=True)
def _posthoc_kernel(lab, n):
    Z, Y, X = lab.shape
    S = stats_new(n + 1)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                v = lab[z, y, x]
                if v != 0:
                    stat_add(S, v, z, y, x, (np.int64(z) * Y + y) * X + x)
    return S


# --------------------------------------------------------------------- memory
def probe(variant, path, shape, conn=6):
    import psutil
    cmd = [sys.executable, os.path.join(ROOT, "bench", "mem_probe.py"), variant, path,
           *map(str, shape), str(conn)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert line.strip() == "READY", line
    ps = psutil.Process(p.pid)
    base = ps.memory_info().rss
    peak = base
    p.stdin.write("GO\n")
    p.stdin.flush()
    while p.poll() is None:
        try:
            peak = max(peak, ps.memory_info().rss)
        except psutil.Error:
            break
        time.sleep(0.003)
    out = json.loads(p.stdout.read().strip().splitlines()[-1])
    out.update(baseline_mb=base / 2**20, peak_mb=peak / 2**20, delta_mb=(peak - base) / 2**20)
    return out


def memory():
    rows = []
    variants = ["ufccl", "ufccl+stats", "scipy", "cc3d", "stream", "tiled"]
    for n in (128, 256, 384, 512):
        shape = (n, n, n)
        path = volume_file("blobs", shape)
        for v in variants:
            r = probe(v, path, shape)
            r.update(experiment="cube", variant=v, shape=list(shape), voxels=n ** 3)
            rows.append(r)
            print(v, n, round(r["delta_mb"], 1), "MB", r["n"], flush=True)
    # fixed slice size, growing depth: streaming memory should not grow with Z
    for Z in (128, 512, 2048, 8192):
        shape = (Z, 256, 256)
        path = volume_file("blobs", shape)
        for v in ("ufccl", "stream") if Z <= 2048 else ("stream",):
            r = probe(v, path, shape)
            r.update(experiment="depth", variant=v, shape=list(shape), voxels=int(np.prod(shape)))
            rows.append(r)
            print(v, shape, round(r["delta_mb"], 1), "MB", r["n"], flush=True)
        if Z >= 2048:
            os.remove(path)
    save_json("memory.json", {"hardware": hardware(), "rows": rows})


# ----------------------------------------------------------------- complexity
def complexity():
    # (a) Union-Find ablation at fixed size on three inputs
    n = 96
    inputs = {"blobs": volume_array("blobs", (n,) * 3), "noise": volume_array("noise", (n,) * 3),
              "snake": snake3d(n // 2 + 1, 49, 49), "comb": comb(4001)}
    names_c = {0: "none", 1: "halving", 2: "full"}
    names_l = {0: "size", 1: "index", 2: "naive"}
    label_counted(np.ones((2, 2, 2), bool), 6, 1, 0)
    abl = []
    for iname, m in inputs.items():
        for compress in (0, 1, 2):
            for link in (0, 1, 2):
                s, (roots, nprov, c) = best_of(lambda: label_counted(m, 6, compress, link), 3)
                abl.append(dict(input=iname, voxels=int(m.size), compress=names_c[compress],
                                link=names_l[link], seconds=s, components=roots,
                                provisional=nprov, **c))
                print(iname, names_c[compress], names_l[link], round(s, 3), c["steps1"], flush=True)
    # (b) operation counts versus N for the production configuration
    scal = []
    for kind in ("blobs", "noise"):
        for n in (32, 48, 64, 96, 128, 192, 256):
            m = volume_array(kind, (n,) * 3)
            roots, nprov, c = label_counted(m, 6, 1, 0)
            scal.append(dict(kind=kind, n=n, voxels=int(m.size), components=roots,
                             provisional=nprov, **c))
            print(kind, n, c, flush=True)
    for n in (17, 33, 49, 65, 97):
        m = snake3d(n, n if n % 2 else n + 1, n)
        roots, nprov, c = label_counted(m, 6, 1, 0)
        scal.append(dict(kind="snake", n=n, voxels=int(m.size), components=roots,
                         provisional=nprov, **c))
    for W in (501, 1001, 2001, 4001, 8001, 16001):
        m = comb(W)
        for cfg, (compress, link) in {"halving+size": (1, 0), "none+naive": (0, 2)}.items():
            roots, nprov, c = label_counted(m, 6, compress, link)
            scal.append(dict(kind="comb", config=cfg, n=W, voxels=int(m.size), components=roots,
                             provisional=nprov, **c))
            print("comb", cfg, W, c, flush=True)
    save_json("complexity.json", {"hardware": hardware(), "ablation": abl, "scaling": scal})


# ------------------------------------------------------------------------ app
def app():
    sys.path.insert(0, ROOT)
    from apps.lesion_count import synthetic_study
    res = synthetic_study()
    save_json("app.json", res)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("section", choices=["correctness", "scaling", "memory", "complexity", "app", "all"])
    ap.add_argument("--sizes", type=int, nargs="*")
    a = ap.parse_args()
    run = {"correctness": correctness, "memory": memory, "complexity": complexity, "app": app}
    if a.section == "scaling":
        scaling(a.sizes or SIZES)
    elif a.section == "all":
        for f in (correctness, scaling, memory, complexity, app):
            f()
    else:
        run[a.section]()
