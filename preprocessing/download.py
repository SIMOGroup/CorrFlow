"""Download raw ERA5 and ERA5-Land total precipitation from the Copernicus CDS.

Requires a CDS API key in ~/.cdsapirc. Existing complete files are skipped.
  ERA5 (input):       one file per year,  RAW_INPUT_TP/era5_sl_tp_{year}.nc
  ERA5-Land (target): one file per month, RAW_OUTPUT_DIR/era5_land_tp_{year}_{month:02d}.nc,
                      plus December of the year before the first year, which the
                      de-accumulation in preprocess.ipynb needs as a boundary.
"""
import traceback
from pathlib import Path

import cdsapi

RAW_INPUT_TP   = Path("/mnt/data/khaiht/data/vietnam/input_tp")
RAW_OUTPUT_DIR = Path("/mnt/data/khaiht/data/vietnam/output")

YEARS = list(range(2017, 2026))
DAYS  = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]

# [North, West, South, East], both wider than the model domain
ERA5_AREA      = [30, 95, 0, 135]
ERA5_LAND_AREA = [25, 100, 5, 125]


def should_download(path, min_size_mb):
    """True unless the file exists and is at least min_size_mb (smaller means a partial download)."""
    if not path.exists():
        return True
    size_mb = path.stat().st_size / 1024**2
    if size_mb < min_size_mb:
        print(f"Re-downloading {path.name}: only {size_mb:.1f} MB, probably incomplete")
        return True
    print(f"Skipping {path.name} ({size_mb:.1f} MB)")
    return False


def retrieve(client, dataset, request, out_file):
    print(f"Downloading {out_file.name}")
    try:
        client.retrieve(dataset, request, str(out_file))
    except Exception:
        print(f"Failed: {out_file.name}")
        traceback.print_exc()


def download_era5(client):
    for year in YEARS:
        out_file = RAW_INPUT_TP / f"era5_sl_tp_{year}.nc"
        if should_download(out_file, min_size_mb=50):
            retrieve(client, "reanalysis-era5-single-levels", {
                "product_type": "reanalysis",
                "variable": "total_precipitation",
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": DAYS,
                "time": HOURS,
                "area": ERA5_AREA,
                "data_format": "netcdf",
            }, out_file)


def download_era5_land(client):
    months = [(YEARS[0] - 1, 12)] + [(y, m) for y in YEARS for m in range(1, 13)]
    for year, month in months:
        out_file = RAW_OUTPUT_DIR / f"era5_land_tp_{year}_{month:02d}.nc"
        if should_download(out_file, min_size_mb=5):
            retrieve(client, "reanalysis-era5-land", {
                "variable": "total_precipitation",
                "year": str(year),
                "month": f"{month:02d}",
                "day": DAYS,
                "time": HOURS,
                "area": ERA5_LAND_AREA,
                "data_format": "netcdf",
            }, out_file)


def main():
    RAW_INPUT_TP.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    download_era5(client)
    download_era5_land(client)
    print("All downloads complete.")


if __name__ == "__main__":
    main()
