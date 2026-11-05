"""Score every method on a set of hours; one file of per-hour scores per method and split.

Hour sets are those of the cached results (paths.RESULTS, from cache_extremes.py and cache_sample.py):
  train = the 1,497-hour training sample of 2017-2023 (in-sample scores)
  val  = cached 2024 hours (development set; used to build and check this code)
  test = 2025: the month-stratified sample of 1,999 hours (unconditional) ∪ the 2,711 heavy-rain
         hours (spatial target p99 > 3.31 mm/hr); each hour is tagged with the sets it belongs to.
The test year is opened only with --allow-test, after every checkpoint has been chosen.

Methods ("name=source"):
  cache:<field>        a field of the cached npz: unet_mm, cd_mm (CorrDiff), cf_mm (CorrFlow seed 0)
  preds:<folder>       predictions from predict.py: <folder>/<split>/<zi>.npz with field "mm"
  bilinear_log         expm1 of the network input, i.e. ERA5 interpolated in log(1 + p) space
  bilinear_mm          conventional bilinear interpolation of the 0.25° ERA5 rain in mm/hr (Bilinear)
  (cache:truth_mm scores ERA5-Land itself, useful with --reference imerg|gsmap)
--reference imerg|gsmap scores against the independent products instead of ERA5-Land
(independent.py); output files get the suffix _vs_imerg / _vs_gsmap.

    python score_sets.py --split val --method corrdiff=cache:cd_mm --method corrflow_s0=cache:cf_mm
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scores import (hour_scores, stack, INTENSITY_EDGES, RELIABILITY_THRESHOLDS,  # noqa: E402
                    COVERAGE_LEVELS, CONTINGENCY_THRESHOLDS, FSS_SIZES, VARIOGRAM_LAGS,
                    VARIOGRAM_P)
from era5 import era5_bilinear_mm                                                   # noqa: E402
from independent import reference_hour                                                     # noqa: E402

from paths import RESULTS as PAPER, ZARR, EVAL                                     # noqa: E402

OUT = os.path.join(EVAL, "scores")


def hour_set(split):
    """Sorted hour indices and their tags (sample / heavy) for a split."""
    if split == "val":
        zis = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(f"{PAPER}/cache/val/*.npz"))
        return np.array(zis), {"sample": np.ones(len(zis), bool)}
    if split == "train":                     # the paper's 1,497-hour training sample (in-sample scores)
        zis = sorted(json.load(open(f"{PAPER}/fullperiod_sample_idxs.json"))["sample_idxs"]["train"])
        return np.array(zis), {"sample": np.ones(len(zis), bool)}
    sample = set(json.load(open(f"{PAPER}/fullperiod_sample_idxs.json"))["sample_idxs"]["test"])
    heavy = set(json.load(open(f"{PAPER}/extreme_meta.json"))["extreme_idxs"]["test"])
    zis = np.array(sorted(sample | heavy))
    return zis, {"sample": np.isin(zis, list(sample)), "heavy": np.isin(zis, list(heavy))}


def load_member(source, split, zi, cache, tp_coarse):
    if source.startswith("cache:"):
        x = cache[source.split(":", 1)[1]]
    elif source.startswith("preds:"):
        x = np.load(os.path.join(source.split(":", 1)[1], split, f"{zi}.npz"))["mm"]
    elif source == "bilinear_log":
        x = np.clip(np.expm1(tp_coarse), 0, None)
    elif source == "bilinear_mm":
        x = era5_bilinear_mm(tp_coarse)          # here tp_coarse carries the hour (datetime64)
    else:
        raise ValueError(source)
    x = np.asarray(x, dtype=np.float32)
    return x[None] if x.ndim == 2 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--method", action="append", required=True, help="name=source")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N hours only (smoke test)")
    ap.add_argument("--reference", choices=["era5land", "imerg", "gsmap"], default="era5land",
                    help="what the methods are scored against (imerg: Final run hours only)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The 2025 evaluation year is closed until every checkpoint is chosen. "
                 "Add --allow-test only then.")

    zis, tags = hour_set(args.split)
    if args.limit:
        zis, tags = zis[:args.limit], {k: v[:args.limit] for k, v in tags.items()}
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    times = ds["time"].values[zis]
    methods = dict(m.split("=", 1) for m in args.method)
    need_coarse = any(src == "bilinear_log" for src in methods.values())
    print(f"{args.split}: {len(zis)} hours, methods {list(methods)}", flush=True)

    recs = {name: [] for name in methods}
    kept = []
    rng = {name: np.random.default_rng(12345) for name in methods}   # rank tie-breaking
    t0 = time.time()
    for i, zi in enumerate(zis):
        cache = np.load(f"{PAPER}/cache/{args.split}/{zi}.npz")
        obs = cache["truth_mm"]
        land = ~np.isnan(obs)
        if args.reference != "era5land":
            ref = reference_hour(args.reference, times[i])
            if ref is None:                       # e.g. IMERG Late run (Oct–Dec 2025): not used
                continue
            obs = np.where(land, ref, np.nan).astype(np.float32)
        kept.append(i)
        tp_coarse = ds["tp_coarse"].isel(time=int(zi)).values if need_coarse else None
        for name, src in methods.items():
            ens = load_member(src, args.split, zi, cache,
                              times[i] if src == "bilinear_mm" else tp_coarse)
            recs[name].append(hour_scores(obs, ens, land, rng[name]))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(zis)}  {(time.time() - t0) / (i + 1):.2f} s/hour", flush=True)

    kept = np.array(kept, dtype=int)
    print(f"scored {len(kept)} of {len(zis)} hours against {args.reference}")
    zis, times, tags = zis[kept], times[kept], {k: v[kept] for k, v in tags.items()}
    suffix = "" if args.reference == "era5land" else f"_vs_{args.reference}"
    os.makedirs(args.out, exist_ok=True)
    for name, src in methods.items():
        path = os.path.join(args.out, f"{args.split}_{name}{suffix}.npz")
        np.savez_compressed(path, zi=zis, time=times.astype("datetime64[s]").astype(np.int64),
                            source=src, reference=args.reference, intensity_edges=INTENSITY_EDGES,
                            reliability_thresholds=RELIABILITY_THRESHOLDS,
                            coverage_levels=COVERAGE_LEVELS,
                            contingency_thresholds=CONTINGENCY_THRESHOLDS,
                            fss_sizes=FSS_SIZES, variogram_lags=VARIOGRAM_LAGS, variogram_p=VARIOGRAM_P,
                            **{f"tag_{k}": v for k, v in tags.items()}, **stack(recs[name]))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
