"""Is our ERA5-Land (or our ERA5) faulty in July-November 2024? Compare with Earth Engine's own copy.

    --part fetch    (needs earthengine-api and EE_PROJECT, a Google Cloud project with Earth Engine)
                    ERA5-Land hourly total_precipitation_hourly from Earth Engine
                    (ECMWF/ERA5_LAND/HOURLY) on our 0.1 deg grid for sample days
    --part compare  spatial correlation over land, log1p space, of the Earth
                    Engine field with our zarr ERA5-Land (tp) and ERA5 input (tp_coarse) at lags -1..+1 h

Sample days: 3-4 June 2024 (consistent month), 3-4 July 2024 (inconsistent), 3-4 July 2025 (consistent).
"""
import argparse
import os

import numpy as np

LON0, LAT1, NX, NY = 102.05, 25.05, 160, 192              # cell edges of the 0.1 deg target grid
DAYS = ["2024-06-03", "2024-06-04", "2024-07-03", "2024-07-04", "2025-07-03", "2025-07-04"]
from paths import ZARR


def fetch(out):
    import datetime as dt
    import ee
    ee.Initialize(project=os.environ["EE_PROJECT"])
    grid = {"dimensions": {"width": NX, "height": NY}, "crsCode": "EPSG:4326",
            "affineTransform": {"scaleX": 0.1, "shearX": 0, "translateX": LON0,
                                "shearY": 0, "scaleY": -0.1, "translateY": LAT1}}
    col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select("total_precipitation_hourly")
    fields, starts = [], []
    for d in DAYS:
        day0 = dt.datetime.fromisoformat(d)
        for h in range(24):
            s = day0 + dt.timedelta(hours=h)
            img = col.filterDate(s.isoformat(), (s + dt.timedelta(minutes=30)).isoformat()).first()
            a = ee.data.computePixels({"expression": ee.Image(img).unmask(-1).toFloat(),
                                       "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            fields.append(np.asarray(a["total_precipitation_hourly"], np.float32))
            starts.append(np.datetime64(s, "s"))
        print(d, "fetched", flush=True)
    np.savez_compressed(out, tp_m=np.stack(fields), time_start=np.array(starts),
                        source="ECMWF/ERA5_LAND/HOURLY total_precipitation_hourly (m), Earth Engine")


def compare(path):
    import xarray as xr
    z = np.load(path)
    ee_mm = z["tp_m"] * 1000.0
    starts = z["time_start"]
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    times = ds["time"].values.astype("datetime64[s]")
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    rows = []
    for k, s in enumerate(starts):
        f = ee_mm[k]
        if (f[land] < 0).any():
            continue
        e = np.log1p(np.clip(f[land], 0, None))
        rec = {"ee_time_start": str(s)}
        for lag in (-1, 0, 1):                              # zarr hour = EE start + 1 h + lag
            i = np.searchsorted(times, s + np.timedelta64(3600 * (1 + lag), "s"))
            if i >= len(times) or times[i] != s + np.timedelta64(3600 * (1 + lag), "s"):
                continue
            rec[f"ours_era5land_lag{lag:+d}"] = c(e, ds["tp"].isel(time=int(i)).values[land])
            rec[f"ours_era5_lag{lag:+d}"] = c(e, ds["tp_coarse"].isel(time=int(i)).values[land])
        rows.append(rec)
    import pandas as pd
    df = pd.DataFrame(rows)
    df["day"] = df.ee_time_start.str[:10]
    pd.set_option("display.width", 200)
    print(df.groupby("day").mean(numeric_only=True).round(3).to_string())
    df.to_csv(path.replace(".npz", "_compare.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["fetch", "compare"], required=True)
    ap.add_argument("--file", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.file), exist_ok=True)
    fetch(a.file) if a.part == "fetch" else compare(a.file)
