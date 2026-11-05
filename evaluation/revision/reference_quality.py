"""How good are ERA5 and ERA5-Land over the study area?

Scores ERA5-Land (the training target) and ERA5 (bilinear in mm/hr) against the independent
products IMERG (Final run only) and GSMaP (gauge-adjusted), over every hour of the given years on
the land grid, hourly and as daily totals. Uses no model output, so the test year may be included.

Writes per-hour and per-day score files in the score_sets.py format to
$CORRFLOW_OUT/evaluation/quality/, readable by bootstrap.py
(--scores .../quality --split <label>_hourly or <label>_daily).

    python reference_quality.py --years 2024 2025 --label y2024_2025
"""
import argparse
import os
import sys
import time

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scores import (hour_scores, stack, INTENSITY_EDGES, RELIABILITY_THRESHOLDS,  # noqa: E402
                    COVERAGE_LEVELS, CONTINGENCY_THRESHOLDS, FSS_SIZES, VARIOGRAM_LAGS,
                    VARIOGRAM_P)
from independent import reference_hour                                             # noqa: E402
from era5 import era5_bilinear_mm                                                  # noqa: E402

from paths import ZARR, EVAL                                                       # noqa: E402

OUT = os.path.join(EVAL, "quality")
SOURCES = ("era5land", "era5")
REFS = ("imerg", "gsmap")


def save(path, recs, times):
    np.savez_compressed(path, zi=np.arange(len(times)), time=np.array(times, dtype=np.int64),
                        tag_sample=np.ones(len(times), bool), intensity_edges=INTENSITY_EDGES,
                        reliability_thresholds=RELIABILITY_THRESHOLDS, coverage_levels=COVERAGE_LEVELS,
                        contingency_thresholds=CONTINGENCY_THRESHOLDS,
                        fss_sizes=FSS_SIZES, variogram_lags=VARIOGRAM_LAGS, variogram_p=VARIOGRAM_P, **stack(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    t_all = ds["time"].values
    zis = np.where(np.isin(ds["time"].dt.year.values, args.years))[0]
    if args.limit:
        zis = zis[:args.limit]
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    rng = np.random.default_rng(0)
    hourly = {(s, r): ([], []) for s in SOURCES for r in REFS}        # (records, times)
    daily = {(s, r): ([], []) for s in SOURCES for r in REFS}
    acc = {}                                                          # (source, ref) -> day sums
    print(f"{args.label}: {len(zis)} hours", flush=True)
    t0 = time.time()
    for n, zi in enumerate(zis):
        t = t_all[zi]
        fields = {"era5land": np.expm1(ds["tp"].isel(time=int(zi)).values),
                  "era5": era5_bilinear_mm(t)}
        # hour ending at t belongs to the UTC day of (t - 1 h)
        day = (np.datetime64(t, "h") - np.timedelta64(1, "h")).astype("datetime64[D]")
        for r in REFS:
            ref = reference_hour(r, t)
            obs = None if ref is None else np.where(land, ref, np.nan).astype(np.float32)
            for s in SOURCES:
                if obs is not None:
                    hourly[(s, r)][0].append(hour_scores(obs, fields[s][None], land, rng))
                    hourly[(s, r)][1].append(np.datetime64(t, "s").astype(np.int64))
                a = acc.setdefault((s, r), {"day": day, "n": 0, "src": 0.0, "ref": 0.0})
                if a["day"] != day:                                   # close the previous day
                    if a["n"] == 24:
                        o = np.where(land, a["ref"], np.nan).astype(np.float32)
                        daily[(s, r)][0].append(hour_scores(o, a["src"][None], land, rng))
                        daily[(s, r)][1].append(np.datetime64(a["day"], "s").astype(np.int64))
                    acc[(s, r)] = a = {"day": day, "n": 0, "src": 0.0, "ref": 0.0}
                if obs is not None:
                    a["n"] += 1
                    a["src"] = a["src"] + fields[s]
                    a["ref"] = a["ref"] + ref
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(zis)}  {(time.time() - t0) / (n + 1):.3f} s/hour", flush=True)

    os.makedirs(args.out, exist_ok=True)
    for (s, r), (recs, times) in hourly.items():
        if recs:
            save(os.path.join(args.out, f"{args.label}_hourly_{s}_vs_{r}.npz"), recs, times)
        dr, dt_ = daily[(s, r)]
        if dr:
            save(os.path.join(args.out, f"{args.label}_daily_{s}_vs_{r}.npz"), dr, dt_)
        print(f"{s} vs {r}: {len(recs)} hours, {len(dr)} complete days")


if __name__ == "__main__":
    main()
