"""Download independent rainfall references (IMERG V07, GSMaP v8 gauge) over Vietnam from Earth Engine.

An independent check of ERA5-Land and of the downscaled fields. Both products sit on the same global 0.1° grid (cell edges on whole tenths of a degree);
the box below is cut on that native grid, so nothing is resampled here.

- IMERG V07 (NASA/GPM_L3/IMERG_V07), band `precipitation` (mm/hr), half-hourly. Status
  "permanent" = Final run (monthly gauge calibration); "provisional" = Late run. In October 2026
  the Final run reached 2025-09-30, so Oct–Dec 2025 were Late run; the status of every step is
  saved so the evaluation can separate them.
- GSMaP v8 (JAXA/GPM_L3/GSMaP/v8/operational), bands `hourlyPrecipRateGC` (gauge-adjusted with
  NOAA CPC daily gauges) and `hourlyPrecipRate` (satellite only), mm/hr, hourly; all 2024–2025
  steps were "permanent" (standard MVK product) in October 2026.

One file per UTC day: <out>/<product>/YYYYMMDD.npz with
  precip (n_steps, [n_bands,] 210, 180) float32 mm/hr, rows north to south (25.95 to 5.05°N),
  columns west to east (101.05 to 118.95°E); time_start (n_steps,) int64 ms since 1970 (UTC, start of
  the accumulation period); status (n_steps,) str; lat, lon (cell centres); bands.
Existing day files are skipped, so the job can be rerun to resume.

    python download_independent.py --product imerg --start 2024-01-01 --end 2026-01-01
"""
import argparse
import datetime as dt
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np

PRODUCTS = {
    "imerg": dict(collection="NASA/GPM_L3/IMERG_V07", bands=["precipitation"], steps_per_day=48),
    "gsmap": dict(collection="JAXA/GPM_L3/GSMaP/v8/operational",
                  bands=["hourlyPrecipRateGC", "hourlyPrecipRate"], steps_per_day=24),
}
# Box on the native 0.1° grid: 5–26°N, 101–119°E (target grid 5.9–25.0°N, 102.1–118.0°E + margin)
WEST, NORTH, NX, NY, RES = 101.0, 26.0, 180, 210, 0.1
GRID = {"dimensions": {"width": NX, "height": NY},
        "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": WEST,
                            "shearY": 0, "scaleY": -RES, "translateY": NORTH},
        "crsCode": "EPSG:4326"}
LAT = np.round(NORTH - RES * (np.arange(NY) + 0.5), 2).astype(np.float32)
LON = np.round(WEST + RES * (np.arange(NX) + 0.5), 2).astype(np.float32)


def fetch_day(product, day, out_dir, retries=6):
    """Download one UTC day; returns (day, n_steps, message)."""
    path = os.path.join(out_dir, day.strftime("%Y%m%d") + ".npz")
    if os.path.exists(path):
        return day, None, "exists"
    spec = PRODUCTS[product]
    t0 = ee.Date(day.isoformat())
    col = (ee.ImageCollection(spec["collection"]).filterDate(t0, t0.advance(1, "day"))
           .sort("system:time_start").select(spec["bands"]))
    for attempt in range(retries):
        try:
            meta = ee.Dictionary({"t": col.aggregate_array("system:time_start"),
                                  "s": col.aggregate_array("status")}).getInfo()
            n = len(meta["t"])
            if n == 0:
                return day, 0, "no images"
            arr = ee.data.computePixels({"expression": col.toBands(),
                                         "fileFormat": "NUMPY_NDARRAY", "grid": GRID})
            names = arr.dtype.names               # "<index>_<band>", in collection order
            stack = np.stack([arr[nm] for nm in names]).astype(np.float32)   # (n*b, NY, NX)
            nb = len(spec["bands"])
            stack = stack.reshape(n, nb, NY, NX)
            if nb == 1:
                stack = stack[:, 0]
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, precip=stack, time_start=np.array(meta["t"], dtype=np.int64),
                                status=np.array(meta["s"]), lat=LAT, lon=LON,
                                bands=np.array(spec["bands"]), collection=spec["collection"])
            os.replace(tmp, path)
            msg = "" if n == spec["steps_per_day"] else f"expected {spec['steps_per_day']}"
            return day, n, msg
        except Exception as exc:                 # rate limits, timeouts: back off and retry
            wait = 10 * 2 ** attempt
            if attempt == retries - 1:
                return day, None, f"FAILED: {str(exc)[:150]}"
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=list(PRODUCTS), required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-01-01", help="exclusive")
    ap.add_argument("--out", required=True, help="the satellite folder, $CORRFLOW_DATA/satellite")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                    help="Google Cloud project with Earth Engine access (default: $EE_PROJECT)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    out_dir = os.path.join(args.out, args.product)
    os.makedirs(out_dir, exist_ok=True)
    d0, d1 = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    days = [d0 + dt.timedelta(i) for i in range((d1 - d0).days)]
    print(f"{args.product}: {len(days)} days into {out_dir}", flush=True)

    failed, short = [], []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_day, args.product, d, out_dir) for d in days]
        for i, fut in enumerate(as_completed(futures), 1):
            day, n, msg = fut.result()
            if msg.startswith("FAILED") or msg == "no images":
                failed.append((day, msg))
            elif msg.startswith("expected"):
                short.append((day, n))
            if i % 50 == 0 or msg.startswith("FAILED"):
                print(f"  {i}/{len(days)}  {day}  {n}  {msg}", flush=True)
    print(f"{args.product}: done. failed {len(failed)}, incomplete days {len(short)}", flush=True)
    for day, msg in failed[:20]:
        print("  failed", day, msg)
    for day, n in short[:20]:
        print("  incomplete", day, n)


if __name__ == "__main__":
    main()
