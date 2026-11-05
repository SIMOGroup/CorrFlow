"""Per-hour scores of every method.

Every score is computed per hour over the valid land pixels and saved per hour, so that the
bootstrap (bootstrap.py) can resample whole days and compare methods on the same hours (paired).
The scores: MAE, bias, RMSE and correlation of the ensemble mean; CRPS and fair CRPS; spread (the
square root of the mean ensemble variance) for the spread-skill ratio; rank histograms, interval
coverage and reliability, with counts; errors by target intensity; exceedance scores; and spatial
structure (fractions skill score and variogram score).

Inputs, all in mm/hr on the 0.1° land grid:
    obs  (H, W) with NaN outside land;  ens (K, H, W), K = 16 for ensembles, 1 for deterministic.
"""
import numpy as np
from scipy.ndimage import uniform_filter

# Observed-intensity bins (mm/hr) for stratified errors; 3.3122 is the pooled land p99 of the
# training years, which also selects the heavy-rain hours (cache_extremes.py, extreme_meta.json).
Q99_MM = 3.3122003078460693
INTENSITY_EDGES = np.array([0.0, 0.1, 1.0, Q99_MM, 10.0, np.inf])
RELIABILITY_THRESHOLDS = np.array([0.1, 1.0, Q99_MM, 10.0])
COVERAGE_LEVELS = np.array([0.5, 0.8, 0.9])
# Rain / no-rain contingency of the ensemble mean (mm/hr; daily totals use the same numbers in mm)
CONTINGENCY_THRESHOLDS = np.array([0.1, 1.0, Q99_MM, 10.0])
# Spatial structure: fractions skill score over square neighborhoods of FSS_SIZES cells
# (0.1° each) at the reliability thresholds; variogram score of order VARIOGRAM_P over pairs of
# cells VARIOGRAM_LAGS cells apart along rows and along columns
FSS_SIZES = np.array([1, 3, 5, 9, 17, 33])
VARIOGRAM_LAGS = np.array([1, 2, 4, 8, 16])
VARIOGRAM_P = 0.5


def crps_ensemble(ens, obs):
    """Per-pixel ensemble CRPS (the standard estimator) and the 'fair' version.

    ens (K, N), obs (N). CRPS = mean|x_i - y| - 1/(2K^2) sum_ij |x_i - x_j|; the fair estimator
    divides the second term by 2K(K-1) instead, which removes the bias of a finite ensemble.
    """
    k = ens.shape[0]
    acc = np.mean(np.abs(ens - obs[None]), axis=0)
    if k == 1:
        return acc, acc
    es = np.sort(ens, axis=0)
    w = (2 * np.arange(k) - k + 1).reshape(k, 1)
    pair = np.sum(w * es, axis=0)                 # = 1/2 sum_ij |x_i - x_j|
    return acc - pair / k**2, acc - pair / (k * (k - 1))


def fss_sums(obs, ens, valid):
    """Fractions skill score terms (Roberts and Lean 2008) per threshold and neighborhood size.

    The fraction of a window is taken over the valid cells inside it, so windows at the domain
    edge or the coast use only the valid cells they contain, and fractions are compared on valid
    cells. Returns the sums over cells of (F - O)^2 and F^2 + O^2, averaged over the members
    ("mem": every member verified as a field) and for the mean of the member fractions ("ens").
    After pooling hours, FSS = 1 - sum(num) / sum(den).
    """
    v = valid.astype(np.float64)
    o = np.where(valid, obs, 0.0)
    e = np.where(valid[None], ens, 0.0)
    shape = (len(RELIABILITY_THRESHOLDS), len(FSS_SIZES))
    num_m, den_m, num_e, den_e = (np.zeros(shape) for _ in range(4))
    for j, size in enumerate(FSS_SIZES):
        cnt = uniform_filter(v, size, mode="constant")[valid]
        for i, thr in enumerate(RELIABILITY_THRESHOLDS):
            fo = uniform_filter((o > thr) * v, size, mode="constant")[valid] / cnt
            fe = uniform_filter((e > thr) * v[None], size=(1, size, size), mode="constant")[:, valid] / cnt
            num_m[i, j] = np.mean(np.sum((fe - fo) ** 2, axis=1))
            den_m[i, j] = np.mean(np.sum(fe ** 2, axis=1)) + np.sum(fo ** 2)
            fm = fe.mean(axis=0)
            num_e[i, j] = np.sum((fm - fo) ** 2)
            den_e[i, j] = np.sum(fm ** 2) + np.sum(fo ** 2)
    return {"fss_num_mem": num_m, "fss_den_mem": den_m, "fss_num_ens": num_e, "fss_den_ens": den_e}


def variogram_sums(obs, ens, valid):
    """Variogram score terms of order p (Scheuerer and Hamill 2015) per lag distance.

    Pairs are valid cells `lag` cells apart along rows or along columns, all with equal weight.
    For each pair the target term |y_a - y_b|^p is compared with its ensemble mean
    (1/K) sum_k |x_a^k - x_b^k|^p. Returns per lag the sum of squared differences, the number of
    pairs and the sums of both terms (their ratio compares the roughness of the members with that
    of the target: below 1 = members smoother than the target).
    """
    o = np.where(valid, obs, 0.0).astype(np.float64)
    e = np.where(valid[None], ens, 0.0).astype(np.float64)
    nl = len(VARIOGRAM_LAGS)
    sq, n_pair, s_obs, s_ens = np.zeros(nl), np.zeros(nl, np.int64), np.zeros(nl), np.zeros(nl)
    for l, d in enumerate(VARIOGRAM_LAGS):
        for a, b in (((slice(0, -d), slice(None)), (slice(d, None), slice(None))),     # along rows
                     ((slice(None), slice(0, -d)), (slice(None), slice(d, None)))):    # along columns
            pair = valid[a] & valid[b]
            go = np.abs(o[b] - o[a])[pair] ** VARIOGRAM_P
            ge = np.mean(np.abs(e[(slice(None),) + b] - e[(slice(None),) + a])[:, pair] ** VARIOGRAM_P,
                         axis=0)
            sq[l] += np.sum((go - ge) ** 2)
            n_pair[l] += pair.sum()
            s_obs[l] += go.sum()
            s_ens[l] += ge.sum()
    return {"vs_sq": sq, "vs_npair": n_pair, "vs_obs": s_obs, "vs_ens": s_ens}


def hour_scores(obs, ens, land, rng):
    """All scores for one hour. Returns a dict of scalars and small count arrays."""
    valid = land & ~np.isnan(obs) & ~np.any(np.isnan(ens), axis=0)
    o = obs[valid].astype(np.float64)
    e = ens[:, valid].astype(np.float64)
    k, n = e.shape
    m = e.mean(axis=0)
    err = m - o
    crps, crps_fair = crps_ensemble(e, o)
    var = e.var(axis=0, ddof=1) if k > 1 else np.zeros(n)
    corr = (np.corrcoef(m, o)[0, 1] if np.std(m) > 0 and np.std(o) > 0 else np.nan)

    out = {
        "n_pix": n,
        "mae": np.mean(np.abs(err)),
        "mse": np.mean(err ** 2),
        "bias": np.mean(err),
        "corr": corr,
        "crps": np.mean(crps),
        "crps_fair": np.mean(crps_fair),
        "ens_var": np.mean(var),                   # spread^2 averaged over pixels
        "obs_mean": np.mean(o),
        "pred_mean": np.mean(m),
        "brier_q99": np.mean((np.mean(e > Q99_MM, axis=0) - (o > Q99_MM)) ** 2),
    }

    # Errors stratified by observed intensity: counts and sums, aggregated later
    b = np.digitize(o, INTENSITY_EDGES[1:-1])
    nb = len(INTENSITY_EDGES) - 1
    out["int_count"] = np.bincount(b, minlength=nb)
    out["int_abs_err"] = np.bincount(b, weights=np.abs(err), minlength=nb)
    out["int_err"] = np.bincount(b, weights=err, minlength=nb)
    out["int_crps"] = np.bincount(b, weights=crps, minlength=nb)
    out["int_sq_err"] = np.bincount(b, weights=err ** 2, minlength=nb)

    if k > 1:
        # Rank of the observation among the members, ties (e.g. all zero) broken at random
        below = np.sum(e < o, axis=0)
        ties = np.sum(e == o, axis=0)
        rank = below + np.floor(rng.random(n) * (ties + 1)).astype(int)
        out["rank_all"] = np.bincount(rank, minlength=k + 1)
        wet = o >= 0.1
        out["rank_wet"] = np.bincount(rank[wet], minlength=k + 1)
        # Central-interval coverage from the empirical member quantiles
        # Coverage over all pixels is inflated by dry pixels (obs 0, all members 0 counts as
        # "inside"), so it is also counted on wet pixels only (obs >= 0.1 mm/hr)
        cov, cov_wet = [], []
        for lv in COVERAGE_LEVELS:
            lo, hi = np.quantile(e, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
            inside = (o >= lo) & (o <= hi)
            cov.append(np.sum(inside))
            cov_wet.append(np.sum(inside & wet))
        out["cov_hits"] = np.array(cov)
        out["cov_hits_wet"] = np.array(cov_wet)
    else:
        out["rank_all"] = np.zeros(2, int)
        out["rank_wet"] = np.zeros(2, int)
        out["cov_hits"] = np.full(len(COVERAGE_LEVELS), -1)
        out["cov_hits_wet"] = np.full(len(COVERAGE_LEVELS), -1)
    out["n_wet"] = int(np.sum(o >= 0.1))

    # Reliability: forecast probability (fraction of members above a threshold) in K+1 levels;
    # per level, number of pixels and number of observed exceedances
    rel = np.zeros((len(RELIABILITY_THRESHOLDS), k + 1, 2), dtype=np.int64)
    for i, thr in enumerate(RELIABILITY_THRESHOLDS):
        p = np.sum(e > thr, axis=0)
        rel[i, :, 0] = np.bincount(p, minlength=k + 1)
        rel[i, :, 1] = np.bincount(p, weights=(o > thr), minlength=k + 1)
    out["reliability"] = rel

    # Contingency of the ensemble mean: hits, misses, false alarms, correct negatives (for CSI and ETS)
    cont = np.zeros((len(CONTINGENCY_THRESHOLDS), 4), dtype=np.int64)
    for i, thr in enumerate(CONTINGENCY_THRESHOLDS):
        f, ob = m >= thr, o >= thr
        cont[i] = [np.sum(f & ob), np.sum(~f & ob), np.sum(f & ~ob), np.sum(~f & ~ob)]
    out["contingency"] = cont

    # Spatial structure of the fields
    out.update(fss_sums(obs, ens, valid))
    out.update(variogram_sums(obs, ens, valid))
    return out


def categorical(cont):
    """Scores from summed contingency counts (..., 4): POD, FAR, CSI, ETS, frequency bias."""
    h, m, f, c = (cont[..., i].astype(float) for i in range(4))
    n = h + m + f + c
    hr = (h + m) * (h + f) / np.maximum(n, 1)                 # hits expected by chance
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"POD": h / (h + m), "FAR": f / (h + f), "CSI": h / (h + m + f),
                "ETS": (h - hr) / (h + m + f - hr), "freq_bias": (h + f) / (h + m)}


def stack(records):
    """List of per-hour dicts to a dict of arrays (first axis = hour)."""
    return {key: np.array([r[key] for r in records]) for key in records[0]}
