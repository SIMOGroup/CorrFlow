"""Consistency of the ERA5 input and the ERA5-Land target, year by year and month by month.

For hours sampled every 13 h (all times of day), the spatial correlation over land between the
ERA5 input field at hour t (tp_coarse, the model input) and the ERA5-Land target at hour t + lag
(tp, lag = -2..+2 h) is computed in log1p space, with the mean absolute difference in mm/hr at
lag 0. A year whose fields are shifted in time peaks at a lag other than 0.

    python year_consistency.py --out $CORRFLOW_OUT/evaluation/tables/year_consistency
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from paths import ZARR

LAGS = [-2, -1, 0, 1, 2]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output stem: _hours.csv, _years.csv, _months.csv")
    ap.add_argument("--stride", type=int, default=13)
    args = ap.parse_args()
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    times = pd.to_datetime(ds["time"].values)
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    n = len(times)
    rows = []
    for t in range(2, n - 2, args.stride):
        x = ds["tp_coarse"].isel(time=t).values[land]
        y = ds["tp"].isel(time=slice(t - 2, t + 3)).values[:, land]          # lags -2..+2
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue
        rec = {"time": times[t], "year": times[t].year, "month": times[t].month,
               "mae_mm_lag0": float(np.mean(np.abs(np.expm1(x) - np.expm1(y[2])))),
               "era5_mean_mm": float(np.mean(np.expm1(x))), "era5land_mean_mm": float(np.mean(np.expm1(y[2])))}
        for i, lag in enumerate(LAGS):
            rec[f"corr_lag{lag:+d}"] = corr(x, y[i])
        rows.append(rec)
        if len(rows) % 1000 == 0:
            print(f"  {len(rows)} hours", flush=True)
    df = pd.DataFrame(rows)
    cols = [f"corr_lag{lag:+d}" for lag in LAGS]
    df["best_lag"] = df[cols].idxmax(axis=1).str.replace("corr_lag", "").astype(int)
    df.to_csv(args.out + "_hours.csv", index=False)
    agg = {**{c: "mean" for c in cols}, "mae_mm_lag0": "mean", "era5_mean_mm": "mean", "era5land_mean_mm": "mean"}
    years = df.groupby("year").agg(agg)
    years["hours"] = df.groupby("year").size()
    years["share_best_lag0"] = df.assign(z=df.best_lag == 0).groupby("year")["z"].mean()
    years.to_csv(args.out + "_years.csv")
    months = df[df.year >= 2023].groupby(["year", "month"]).agg(agg)
    months["share_best_lag0"] = df[df.year >= 2023].assign(z=lambda d: d.best_lag == 0).groupby(["year", "month"])["z"].mean()
    months.to_csv(args.out + "_months.csv")
    pd.set_option("display.width", 200)
    print(years.round(4).to_string())


if __name__ == "__main__":
    main()
