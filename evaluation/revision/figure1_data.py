"""Data for the study-area figure.

    --part dem   elevation and bathymetry on the model domain at 1 arc-minute: ETOPO1 (bedrock) from
                 Earth Engine (needs earthengine-api and EE_PROJECT); falls back to GMTED2010 (land only)
    --part rain  ERA5-Land heavy-rain frequency over the training years 2017-2023: mean hours per year
                 with hourly rain above tau99 = 3.31 mm/hr (pooled training land p99, extreme_meta.json),
                 plus the mean annual total, on the 0.1 deg target grid

Only training years are used, so the figure carries no information from the 2025 test year.
"""
import argparse
import json
import os

import numpy as np

# Model domain: cell edges of the 0.1 deg target grid (centers 5.9-25.0 N, 102.1-118.0 E)
LON0, LON1, LAT0, LAT1 = 102.05, 118.05, 5.85, 25.05
from paths import ZARR, RESULTS

META = os.path.join(RESULTS, "extreme_meta.json")
TRAIN_YEARS = range(2017, 2024)


def dem(out):
    import ee
    ee.Initialize(project=os.environ["EE_PROJECT"])
    w, h = round((LON1 - LON0) * 60), round((LAT1 - LAT0) * 60)          # 1 arc-minute cells
    grid = {"dimensions": {"width": w, "height": h}, "crsCode": "EPSG:4326",
            "affineTransform": {"scaleX": 1 / 60, "shearX": 0, "translateX": LON0,
                                "shearY": 0, "scaleY": -1 / 60, "translateY": LAT1}}
    for asset, band in [("NOAA/NGDC/ETOPO1", "bedrock"), ("USGS/GMTED2010", "be75")]:
        try:
            arr = ee.data.computePixels({"expression": ee.Image(asset).select(band).toFloat(),
                                         "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            break
        except Exception as e:                                             # noqa: BLE001
            print(f"{asset}: {e}")
    else:
        raise SystemExit("no elevation data")
    elev = np.asarray(arr[band], dtype=np.float32)
    lat = LAT1 - (np.arange(h) + 0.5) / 60
    lon = LON0 + (np.arange(w) + 0.5) / 60
    np.savez_compressed(out, elev=elev, lat=lat, lon=lon,
                        source=f"{asset} band {band}, Earth Engine computePixels, 1 arc-minute")
    print(f"{asset}: {elev.shape}, {np.nanmin(elev):.0f} to {np.nanmax(elev):.0f} m -> {out}")


def rain(out):
    import xarray as xr
    q99 = float(json.load(open(META))["Q99_MM"])
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    years = ds["time"].dt.year.values
    idx = np.where(np.isin(years, list(TRAIN_YEARS)))[0]
    thr = np.log1p(q99)                                                    # tp is log1p(mm/hr)
    count = np.zeros(ds["tp"].shape[1:], np.int64)
    total = np.zeros(ds["tp"].shape[1:], np.float64)
    for s in range(0, len(idx), 1000):
        x = ds["tp"].isel(time=idx[s:s + 1000]).values                     # (n, H, W), NaN off land
        count += np.nansum(x > thr, axis=0)
        total += np.nansum(np.expm1(x), axis=0)
        if s % 10000 == 0:
            print(f"  {s + len(x)}/{len(idx)} hours", flush=True)
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    n_years = len(TRAIN_YEARS)
    np.savez_compressed(out, hours_per_year=np.where(land, count / n_years, np.nan),
                        annual_mm=np.where(land, total / n_years, np.nan),
                        lat=ds["latitude"].values, lon=ds["longitude"].values, q99_mm=q99,
                        n_hours=len(idx), years=np.array(list(TRAIN_YEARS)))
    v = count[land] / n_years
    print(f"tau99 = {q99:.4f} mm/hr, {len(idx)} hours; hours/year above tau99 over land: "
          f"median {np.median(v):.1f}, p90 {np.percentile(v, 90):.1f}, max {v.max():.1f} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dem", "rain"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dem(a.out) if a.part == "dem" else rain(a.out)
