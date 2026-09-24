import cdsapi
from pathlib import Path
import traceback

# ERA5 Downloader
# Designed for SLURM batch execution

client = cdsapi.Client()

# Common settings
years = list(range(2017, 2026))

# [North, West, South, East]
area = [30, 95, 0, 135]

months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
times = [f"{h:02d}:00" for h in range(24)]

out_root = Path("/mnt/data/khaiht/data/vietnam/input_tp")
out_root.mkdir(parents=True, exist_ok=True)

# Helper
def should_download(path: Path, min_size_mb=50):
    """
    Skip if file exists and looks valid.
    min_size_mb protects against corrupted partial files.
    """
    if not path.exists():
        return True

    size_mb = path.stat().st_size / (1024**2)

    if size_mb < min_size_mb:
        print(f"⚠️ Small/corrupted file detected ({size_mb:.1f} MB)")
        print(f"🔄 Re-downloading: {path.name}")
        return True

    print(f"✅ Exists, skipping: {path.name} ({size_mb:.1f} MB)")
    return False


def download_single_level(var_long, var_short, year):
    out_file = out_root / f"era5_sl_{var_short}_{year}.nc"

    if not should_download(out_file):
        return

    request = {
        "product_type": "reanalysis",
        "variable": var_long,
        "year": str(year),
        "month": months,
        "day": days,
        "time": times,
        "area": area,
        "data_format": "netcdf",
    }

    print("=" * 60)
    print(f"⬇️ Downloading: {out_file.name}")
    print("=" * 60)

    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(out_file),
    )

    print(f"✅ Finished: {out_file.name}")


def download_pressure_level(var_long, short_name, level, year):
    out_file = out_root / f"era5_pl_{short_name}_{year}.nc"

    if not should_download(out_file):
        return

    request = {
        "product_type": "reanalysis",
        "variable": var_long,
        "pressure_level": level,
        "year": str(year),
        "month": months,
        "day": days,
        "time": times,
        "area": area,
        "data_format": "netcdf",
    }

    print("=" * 60)
    print(f"⬇️ Downloading: {out_file.name}")
    print("=" * 60)

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        request,
        str(out_file),
    )

    print(f"✅ Finished: {out_file.name}")


# Variables

single_level_vars = {
    # "mean_sea_level_pressure": "msl",
    # "10m_u_component_of_wind": "u10",
    # "10m_v_component_of_wind": "v10",
    # "total_column_water_vapour": "tcwv",
    # "2m_temperature": "t2m",
    # "2m_dewpoint_temperature": "d2m",
    "total_precipitation": "tp",
}

pressure_level_vars = {
    "geopotential": {"short": "z500", "level": "500"},
    "temperature": {"short": "t850", "level": "850"},
    "specific_humidity": {"short": "q850", "level": "850"},
    "u_component_of_wind": {"short": "u850", "level": "850"},
    "v_component_of_wind": {"short": "v850", "level": "850"},
    "vertical_velocity": {"short": "omega500", "level": "500"},
}


# Main
def main():

    # Single-level variables
    for var_long, var_short in single_level_vars.items():
        for year in years:
            try:
                download_single_level(var_long, var_short, year)

            except Exception as e:
                print(f"❌ Failed: {var_long} {year}")
                print(str(e))
                traceback.print_exc()

    # Pressure-level variables
    # Uncomment if needed
    """
    for var_long, info in pressure_level_vars.items():
        for year in years:
            try:
                download_pressure_level(
                    var_long,
                    info["short"],
                    info["level"],
                    year
                )

            except Exception as e:
                print(f"❌ Failed: {var_long} {year}")
                print(str(e))
                traceback.print_exc()
    """

    print("\n🎉 All downloads complete.")


if __name__ == "__main__":
    main()