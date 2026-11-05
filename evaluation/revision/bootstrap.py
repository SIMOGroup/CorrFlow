"""Block bootstrap by day: scores with 95% intervals and paired differences between methods.

Hours of one UTC day are resampled together (rain is correlated in time); every method
is evaluated on the same resampled days, so differences are paired. A "group" averages several
training runs (e.g. CorrFlow seeds 0–3) on each resample, so its interval includes day-to-day
sampling; the spread between runs is reported separately (min–max over seeds).

    python bootstrap.py --split val --method corrdiff --method unet \
        --group corrflow=corrflow_s0,corrflow_s1 --compare corrflow:corrdiff --subset sample
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings

import numpy as np

from scores import categorical
from paths import EVAL

SCORES = os.path.join(EVAL, "scores")


def exceedance(rel):
    """Brier score and ROC area from summed reliability counts (K+1, 2) for one threshold.

    Row j holds the pixels where j of the K members exceed the threshold (forecast probability
    j/K) and how many of them were observed above it. Both scores follow exactly from these counts.
    """
    n, hits = rel[:, 0].astype(float), rel[:, 1].astype(float)
    p = np.arange(len(n)) / (len(n) - 1)
    brier = np.sum(n * p ** 2 - 2 * p * hits + hits) / n.sum()
    # ROC: warn when at least j members exceed, j = K, ..., 0; area by the trapezoid rule
    hr = np.concatenate([[0.0], np.cumsum(hits[::-1]) / hits.sum()])
    far = np.concatenate([[0.0], np.cumsum((n - hits)[::-1]) / (n - hits).sum()])
    auc = np.sum(np.diff(far) * (hr[1:] + hr[:-1]) / 2)
    return brier, auc


def delta_index(counts):
    """Rank-histogram flatness: sum over bins of |observed - uniform| frequency (0 = flat).

    Delle Monache et al. (2006); uses the K+1 rank bins of a K-member ensemble.
    """
    f = counts / counts.sum()
    return np.sum(np.abs(f - 1 / len(f)))


def aggregate(z, w):
    """Scores for one method from per-hour arrays, with hour weights w (resampling counts)."""
    n = z["n_pix"] * w
    tot = n.sum()
    mean = lambda key: np.sum(z[key] * n) / tot           # pixel-pooled mean, as in the paper
    k = z["_k"]
    rmse, spread = np.sqrt(mean("mse")), np.sqrt(mean("ens_var"))
    out = {
        "MAE": mean("mae"), "RMSE": rmse, "bias": mean("bias"), "CRPS": mean("crps"),
        "CRPS_fair": mean("crps_fair"), "Brier_q99": mean("brier_q99"),
        "corr": np.nansum(z["corr"] * w) / np.sum(w * ~np.isnan(z["corr"])),  # mean hourly corr
        "spread": spread,
    }
    cat = categorical((z["contingency"] * w[:, None, None]).sum(0))   # each: (n_thresholds,)
    for i, thr in enumerate(z["contingency_thresholds"]):
        for name in ("CSI", "ETS", "freq_bias"):
            out[f"{name}_{thr:g}"] = cat[name][i]
    # Errors by class of target intensity: pooled over the pixels of each class
    cnt = (z["int_count"] * w[:, None]).sum(0)
    edges = z["intensity_edges"]
    for i in range(len(cnt)):
        tag = f"{edges[i]:g}-{edges[i + 1]:g}"
        c = max(cnt[i], 1)
        out[f"MAE_int_{tag}"] = (z["int_abs_err"][:, i] * w).sum() / c
        out[f"bias_int_{tag}"] = (z["int_err"][:, i] * w).sum() / c
        out[f"CRPS_int_{tag}"] = (z["int_crps"][:, i] * w).sum() / c
        out[f"frac_int_{tag}"] = cnt[i] / tot                          # share of points in the class
    # Exceedance of fixed thresholds: Brier score and ROC area per threshold; observed
    # "above" is o > thr as in brier_q99. Low thresholds: AUC is lifted by easy dry pixels.
    rel = (z["reliability"] * w[:, None, None, None]).sum(0)          # (n_thr, K+1, 2)
    for i, thr in enumerate(z["reliability_thresholds"]):
        out[f"Brier_{thr:g}"], out[f"AUC_{thr:g}"] = exceedance(rel[i])
    # Spatial structure, from sums pooled over hours: fractions skill score per threshold
    # and neighborhood size (members verified as fields, and the mean of the member fractions),
    # variogram score per lag and the member/target variogram ratio (< 1: members smoother)
    if "fss_num_mem" in z:
        for kind, label in (("mem", "FSS"), ("ens", "FSSens")):
            num = (z[f"fss_num_{kind}"] * w[:, None, None]).sum(0)
            den = (z[f"fss_den_{kind}"] * w[:, None, None]).sum(0)
            for i, thr in enumerate(z["reliability_thresholds"]):
                for j, size in enumerate(z["fss_sizes"]):
                    out[f"{label}_{thr:g}_n{size}"] = 1 - num[i, j] / den[i, j]
        n_pair = (z["vs_npair"] * w[:, None]).sum(0)
        vs = (z["vs_sq"] * w[:, None]).sum(0) / n_pair
        ratio = (z["vs_ens"] * w[:, None]).sum(0) / (z["vs_obs"] * w[:, None]).sum(0)
        for l, d in enumerate(z["variogram_lags"]):
            out[f"VS_lag{d}"] = vs[l]
            out[f"vgram_ratio_lag{d}"] = ratio[l]
    if k > 1:
        # Rank-histogram flatness: all land pixels (unconditional) and wet pixels only
        out["delta_rank"] = delta_index((z["rank_all"] * w[:, None]).sum(0))
        out["delta_rank_wet"] = delta_index((z["rank_wet"] * w[:, None]).sum(0))
        out["spread_skill"] = spread * np.sqrt((k + 1) / k) / rmse   # 1 = calibrated (Fortin 2014)
        hits = (z["cov_hits"] * w[:, None]).sum(0) / tot
        hits_wet = (z["cov_hits_wet"] * w[:, None]).sum(0) / max(np.sum(z["n_wet"] * w), 1)
        for lv, h, hw in zip(z["coverage_levels"], hits, hits_wet):
            out[f"coverage_{int(lv * 100)}"] = h          # all pixels (inflated by dry pixels)
            out[f"coverage_{int(lv * 100)}_wet"] = hw     # wet pixels only
    return out


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # NaN for rare events without cases
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="val | test | <label>_hourly | <label>_daily")
    ap.add_argument("--method", action="append", default=[])
    ap.add_argument("--group", action="append", default=[], help="name=run1,run2,...")
    ap.add_argument("--compare", action="append", default=[], help="a:b (difference a - b)")
    ap.add_argument("--subset", default="sample", help="sample | heavy | all")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--reference", choices=["era5land", "imerg", "gsmap"], default="era5land")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups = {m: [m] for m in args.method}
    groups.update({g.split("=")[0]: g.split("=")[1].split(",") for g in args.group})
    runs = sorted({r for rs in groups.values() for r in rs})
    data = {}
    for r in runs:
        suffix = "" if args.reference == "era5land" else f"_vs_{args.reference}"
        z = dict(np.load(os.path.join(args.scores, f"{args.split}_{r}{suffix}.npz")))
        z["_k"] = z["rank_all"].shape[1] - 1 if z["rank_all"].shape[1] > 2 else 1
        data[r] = z
    zi = data[runs[0]]["zi"]
    for r in runs:
        assert np.array_equal(data[r]["zi"], zi), f"{r}: different hours"
    sel = (np.ones(len(zi), bool) if args.subset == "all"
           else data[runs[0]][f"tag_{args.subset}"].astype(bool))
    day = (data[runs[0]]["time"] // 86400)[sel]
    for r in runs:
        data[r] = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape else v)
                   for k, v in data[r].items()}
    days, day_idx = np.unique(day, return_inverse=True)
    print(f"{args.split}/{args.subset}: {sel.sum()} hours on {len(days)} days, "
          f"{args.n_boot} day-block resamples")

    def group_scores(w):
        res = {}
        for g, rs in groups.items():
            per_run = [aggregate(data[r], w) for r in rs]
            res[g] = {m: np.mean([p[m] for p in per_run]) for m in per_run[0]}
        return res

    point = group_scores(np.ones(sel.sum()))
    seed_range = {g: {m: (min(aggregate(data[r], np.ones(sel.sum()))[m] for r in rs),
                          max(aggregate(data[r], np.ones(sel.sum()))[m] for r in rs))
                      for m in point[g]} for g, rs in groups.items() if len(rs) > 1}
    rng = np.random.default_rng(2026)
    boots = []
    for _ in range(args.n_boot):
        pick = rng.integers(0, len(days), len(days))              # days with replacement
        w = np.bincount(pick, minlength=len(days))[day_idx].astype(float)
        boots.append(group_scores(w))

    rows = ["subset,metric,method,value,ci_low,ci_high,seed_min,seed_max"]
    for g in groups:
        for m in point[g]:
            b = np.array([bt[g][m] for bt in boots])
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            smin, smax = seed_range.get(g, {}).get(m, (np.nan, np.nan))
            rows.append(f"{args.subset},{m},{g},{point[g][m]:.6g},{lo:.6g},{hi:.6g},{smin:.6g},{smax:.6g}")
    for c in args.compare:
        a, b_ = c.split(":")
        for m in point[a]:
            if m not in point[b_]:
                continue
            d = np.array([bt[a][m] - bt[b_][m] for bt in boots])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append(f"{args.subset},{m},{a} - {b_},{point[a][m] - point[b_][m]:.6g},{lo:.6g},{hi:.6g},,")
    text = "\n".join(rows)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
