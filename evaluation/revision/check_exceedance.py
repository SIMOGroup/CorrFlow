"""Checks for the scores bootstrap.py derives from saved counts (Brier, ROC area, Delta index,
errors by class of target intensity) and for the spatial scores (fractions skill score, variogram
score), each against a direct computation.

1. Synthetic hours: Brier and AUC from the reliability counts equal a direct per-pixel computation
   (AUC as the Mann-Whitney probability, ties counted half), and so do the errors by intensity
   class; Delta is 0 for a flat histogram and
   2K/(K+1) when all ranks fall in one bin; deterministic AUC = (POD + 1 - POFD) / 2.
2. Spatial scores on a small grid with scattered sea cells vs a cell-by-cell computation, and their
   cost on a full-size grid.
3. Real score files (optional, --split): Brier at the p99 threshold from the counts equals the
   per-hour brier_q99 already saved by scores.py.

    python check_exceedance.py [--split val --run corrflow_s0 --run corrdiff]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import rankdata

from bootstrap import SCORES, aggregate, delta_index
from scores import (CONTINGENCY_THRESHOLDS, COVERAGE_LEVELS, FSS_SIZES, INTENSITY_EDGES,
                    RELIABILITY_THRESHOLDS, VARIOGRAM_LAGS, VARIOGRAM_P, crps_ensemble, fss_sums,
                    hour_scores, stack, variogram_sums)
import time


def direct_auc(p, y):
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def synthetic(k, rng):
    h, w, hours = 24, 30, 12
    land = np.ones((h, w), bool)
    recs, ens_all, obs_all = [], [], []
    for _ in range(hours):
        base = np.maximum(rng.gamma(0.4, 3.0, (h, w)) - 0.5, 0)
        obs = np.maximum(base + rng.normal(0, 1.0, (h, w)), 0)
        ens = np.maximum(base[None] + rng.normal(0, 0.8, (k, h, w)), 0)
        recs.append(hour_scores(obs, ens, land, rng))
        ens_all.append(ens.reshape(k, -1))
        obs_all.append(obs.ravel())
    z = stack(recs)
    z["_k"], z["reliability_thresholds"] = k, RELIABILITY_THRESHOLDS
    z["contingency_thresholds"], z["coverage_levels"] = CONTINGENCY_THRESHOLDS, COVERAGE_LEVELS
    z["intensity_edges"] = INTENSITY_EDGES
    z["fss_sizes"], z["variogram_lags"] = FSS_SIZES, VARIOGRAM_LAGS
    out = aggregate(z, np.ones(hours))
    e, o = np.concatenate(ens_all, axis=1), np.concatenate(obs_all)
    worst = 0.0
    for thr in RELIABILITY_THRESHOLDS:
        p, y = np.mean(e > thr, axis=0), o > thr
        if y.sum() == 0 or y.all():
            continue
        brier, auc = np.mean((p - y) ** 2), direct_auc(p, y)
        if k == 1:
            pod, pofd = np.mean(p[y]), np.mean(p[~y])
            assert abs(auc - (pod + 1 - pofd) / 2) < 1e-12
        db, da = abs(out[f"Brier_{thr:g}"] - brier), abs(out[f"AUC_{thr:g}"] - auc)
        worst = max(worst, db, da)
        print(f"  K={k:2d} thr={thr:<7.4g} Brier {out[f'Brier_{thr:g}']:.6f} vs {brier:.6f}"
              f"   AUC {out[f'AUC_{thr:g}']:.6f} vs {auc:.6f}")
    # Errors by class of target intensity vs a direct computation
    err = e.mean(axis=0) - o
    crps = crps_ensemble(e, o)[0]
    cls = np.digitize(o, INTENSITY_EDGES[1:-1])
    for i in range(len(INTENSITY_EDGES) - 1):
        sel = cls == i
        if not sel.any():
            continue
        tag = f"{INTENSITY_EDGES[i]:g}-{INTENSITY_EDGES[i + 1]:g}"
        for name, direct in (("MAE", np.abs(err[sel]).mean()), ("bias", err[sel].mean()),
                             ("CRPS", crps[sel].mean())):
            worst = max(worst, abs(out[f"{name}_int_{tag}"] - direct))
        worst = max(worst, abs(out[f"frac_int_{tag}"] - sel.mean()))
    print(f"  K={k:2d} intensity classes: MAE, bias, CRPS and class shares checked")
    assert worst < 1e-12, worst
    return worst


def direct_fractions(field, valid, thr, size):
    """Fraction of valid cells above thr in the size x size window around each valid cell."""
    nr, nc = valid.shape
    h = size // 2
    out = np.full((nr, nc), np.nan)
    for r in range(nr):
        for c in range(nc):
            if valid[r, c]:
                win = (slice(max(r - h, 0), r + h + 1), slice(max(c - h, 0), c + h + 1))
                out[r, c] = np.sum((field[win] > thr) & valid[win]) / valid[win].sum()
    return out


def spatial(rng):
    """FSS and variogram terms of hour_scores vs a direct cell-by-cell computation."""
    nr, nc, k = 20, 26, 5
    valid = rng.random((nr, nc)) > 0.25                       # scattered sea cells
    obs = np.maximum(rng.gamma(0.5, 4.0, (nr, nc)) - 1, 0)
    ens = np.maximum(0.6 * obs[None] + rng.gamma(0.5, 3.0, (k, nr, nc)) - 1, 0)
    obs[~valid], ens[:, ~valid] = np.nan, np.nan
    rec = hour_scores(obs, ens, valid, rng)
    worst = 0.0
    for i, thr in enumerate(RELIABILITY_THRESHOLDS):
        for j, size in enumerate(FSS_SIZES):
            fo = direct_fractions(obs, valid, thr, size)[valid]
            fe = np.array([direct_fractions(ens[m], valid, thr, size)[valid] for m in range(k)])
            fm = fe.mean(axis=0)
            want = {"fss_num_mem": np.mean(np.sum((fe - fo) ** 2, axis=1)),
                    "fss_den_mem": np.mean(np.sum(fe ** 2, axis=1)) + np.sum(fo ** 2),
                    "fss_num_ens": np.sum((fm - fo) ** 2),
                    "fss_den_ens": np.sum(fm ** 2) + np.sum(fo ** 2)}
            for key, val in want.items():
                worst = max(worst, abs(rec[key][i, j] - val))
    for l, d in enumerate(VARIOGRAM_LAGS):
        sq = n = so = se = 0.0
        for r in range(nr):
            for c in range(nc):
                for r2, c2 in ((r + d, c), (r, c + d)):
                    if r2 < nr and c2 < nc and valid[r, c] and valid[r2, c2]:
                        go = abs(obs[r2, c2] - obs[r, c]) ** VARIOGRAM_P
                        ge = np.mean(np.abs(ens[:, r2, c2] - ens[:, r, c]) ** VARIOGRAM_P)
                        sq, n, so, se = sq + (go - ge) ** 2, n + 1, so + go, se + ge
        for key, val in (("vs_sq", sq), ("vs_npair", n), ("vs_obs", so), ("vs_ens", se)):
            worst = max(worst, abs(rec[key][l] - val))
    print(f"  FSS terms ({len(RELIABILITY_THRESHOLDS)} thresholds x {len(FSS_SIZES)} sizes) and variogram "
          f"terms ({len(VARIOGRAM_LAGS)} lags): largest difference {worst:.2e}")
    assert worst < 1e-9, worst
    # identical forecast and target: FSS = 1 and variogram score 0
    same = hour_scores(obs, obs[None], valid, rng)
    assert np.allclose(same["fss_num_mem"], 0) and np.allclose(same["vs_sq"], 0)
    print("  identical forecast and target: FSS numerators and variogram score are 0: ok")
    # time on a full-size grid (192 x 160, 16 members, about 60% valid cells)
    v = rng.random((192, 160)) > 0.4
    o = np.where(v, np.maximum(rng.gamma(0.4, 3.0, (192, 160)) - 0.5, 0), np.nan)
    e = np.where(v[None], np.maximum(rng.gamma(0.4, 3.0, (16, 192, 160)) - 0.5, 0), np.nan)
    t = time.time(); [fss_sums(o, e, v) for _ in range(5)]; t_fss = (time.time() - t) / 5
    t = time.time(); [variogram_sums(o, e, v) for _ in range(5)]; t_vs = (time.time() - t) / 5
    t = time.time(); [hour_scores(o, e, v, rng) for _ in range(5)]; t_all = (time.time() - t) / 5
    print(f"  full grid, 16 members: FSS {t_fss:.3f} s, variogram {t_vs:.3f} s, all scores {t_all:.3f} s per hour")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--run", action="append", default=[])
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("1. Synthetic hours: counts-based scores vs direct per-pixel computation")
    for k in (16, 1):
        print(f"  K={k}: largest difference {synthetic(k, rng):.2e}")
    print("2. Spatial scores: hour_scores vs a direct cell-by-cell computation")
    spatial(rng)
    assert delta_index(np.full(17, 5)) == 0
    one = np.zeros(17); one[0] = 9
    assert abs(delta_index(one) - 2 * 16 / 17) < 1e-12
    print("  Delta: flat histogram 0, one-bin histogram 2K/(K+1): ok")

    for r in args.run:
        z = dict(np.load(os.path.join(SCORES, f"{args.split}_{r}.npz")))
        z["_k"] = z["rank_all"].shape[1] - 1 if z["rank_all"].shape[1] > 2 else 1
        out = aggregate(z, np.ones(len(z["zi"])))
        thr = RELIABILITY_THRESHOLDS[2]
        a, b = out[f"Brier_{thr:g}"], out["Brier_q99"]
        print(f"3. {args.split}_{r}: Brier_{thr:g} from counts {a:.9f}, saved brier_q99 {b:.9f}, "
              f"relative difference {abs(a - b) / b:.1e}")
        assert abs(a - b) / b < 1e-9
    print("All checks passed.")


if __name__ == "__main__":
    main()
