"""Which field is inconsistent in July-November 2024: the ERA5 input or the ERA5-Land target?

GSMaP (gauge-adjusted, independent of both) arbitrates. For hours sampled every 13 h in 2024 and
2025, the spatial correlation over land (log1p space) of each zarr field with GSMaP in the same
hour is computed and averaged by month. The field whose correlation with GSMaP drops in the
inconsistent months is the faulty one.

    python which_side.py --out $CORRFLOW_OUT/evaluation/tables/which_side
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from independent import reference_hour      # noqa: E402

from paths import ZARR                     # noqa: E402


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=13)
    args = ap.parse_args()
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    times = pd.to_datetime(ds["time"].values)
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    rows = []
    for t in np.where(times.year >= 2024)[0][::args.stride]:
        g = reference_hour("gsmap", ds["time"].values[t])
        if g is None:
            continue
        g = np.log1p(np.clip(g[land], 0, None))
        e5 = ds["tp_coarse"].isel(time=int(t)).values[land]
        el = ds["tp"].isel(time=int(t)).values[land]
        if not (np.isfinite(e5).all() and np.isfinite(el).all() and np.isfinite(g).all()):
            continue
        rows.append({"time": times[t], "year": times[t].year, "month": times[t].month,
                     "era5_vs_gsmap": corr(e5, g), "era5land_vs_gsmap": corr(el, g), "era5_vs_era5land": corr(e5, el)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out + "_hours.csv", index=False)
    m = df.groupby(["year", "month"])[["era5_vs_gsmap", "era5land_vs_gsmap", "era5_vs_era5land"]].mean()
    m["hours"] = df.groupby(["year", "month"]).size()
    m.to_csv(args.out + "_months.csv")
    pd.set_option("display.width", 200)
    print(m.round(3).to_string())


if __name__ == "__main__":
    main()
