"""Data loading and plotting helpers for Section 7 of preprocess.ipynb.

Each figure cell starts with `from viz_utils import *`, so it runs on a fresh
kernel without the earlier cells. Shared settings (SHOW_PEAK_STAR, SEA_LABEL,
RAIN_COLORS, ...) live here, so changing one updates every figure. Keep this
file next to the notebook.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature

__all__ = [
    # Standard library / imported modules
    "Path",
    "np",
    "pd",
    "xr",
    "plt",
    "mcolors",
    "gridspec",
    "mdates",
    "ccrs",
    "cfeature",

    # Configuration
    "RAW_INPUT_TP",
    "RAW_OUTPUT_DIR",
    "PROCESSED_OUTPUT",
    "LAT_MIN",
    "LAT_MAX",
    "LON_MIN",
    "LON_MAX",
    "ALL_YEARS",
    "TRAIN_YEARS",
    "VAL_YEARS",
    "TEST_YEARS",
    "FULL_EXTENT",

    # Helper functions
    "crop",

    # Peak marker
    "SHOW_PEAK_STAR",
    "SHOW_GRIDLINES",
    "plot_peak_star",

    # Rain colormap
    "RAIN_BOUNDS",
    "RAIN_COLORS",
    "RAIN_CMAP",
    "RAIN_NORM",

    # Dataset precipitation colormap
    "PRECIP_SCALE",
    "PRECIP_CMAP",
    "PRECIP_NORM",
    "precip_norm",

    # Cartopy features
    "LAND_10M",
    "COAST_10M",
    "BORDERS_10M",

    # Island / sea labels
    "SEA_LABEL",
    "add_vietnam_islands",
    "add_east_sea_label",
    "add_map_labels",
    "add_gridlines",
    "setup_map_ax",

    # Data loading
    "get_land_mask",
    "load_fine_tp",
    "load_coarse_tp_masked",

    # Analysis
    "compute_annual_stats",
    "find_extreme_events",
]

# Config, duplicated from Section 1 of the notebook so figure cells
# don't depend on it
RAW_INPUT_TP     = Path("/mnt/data/khaiht/data/vietnam/input_tp")
RAW_OUTPUT_DIR   = Path("/mnt/data/khaiht/data/vietnam/output")
PROCESSED_OUTPUT = Path("/mnt/data/khaiht/data/vietnamvip_processed/output")

LAT_MIN, LAT_MAX = 5.8, 25.0
LON_MIN, LON_MAX = 102.0, 118.0

ALL_YEARS   = list(range(2017, 2026))
TRAIN_YEARS = list(range(2017, 2024))
VAL_YEARS   = [2024]
TEST_YEARS  = [2025]

FULL_EXTENT = (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)


def crop(ds):
    """Crop to the Vietnam bounding box. ERA5 latitudes run north to south, hence slice(MAX, MIN)."""
    return ds.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))


# Peak-star toggle
# Set False to hide the peak-pixel star on extreme-event maps, snapshots,
# animation frames, and the top-4 panel.
SHOW_PEAK_STAR = False

# Gridlines and tick labels on maps; add_gridlines(ax, show=...) overrides per call.
SHOW_GRIDLINES = True


def plot_peak_star(ax, lon, lat, transform, color="lime", ms=16, zorder=10, **kw):
    """Draw the peak-pixel star marker, respecting SHOW_PEAK_STAR. Returns the
    Line2D artist (or None if disabled) so callers can update it in animations."""
    if not SHOW_PEAK_STAR:
        return None
    line, = ax.plot(lon, lat, marker="*", color=color, ms=ms,
                     markeredgecolor="black", markeredgewidth=0.8,
                     transform=transform, zorder=zorder, **kw)
    return line


# Rainfall colormap shared by every intensity map in Section 7 (multi-year max,
# p99, extreme snapshots, animation, top-4 panel). Blue for light rain, red for
# extremes; it matches the palette of the results figures in results.ipynb.
RAIN_BOUNDS = [0.1, 0.5, 1., 2., 5., 10., 20., 40., 80., 150.]
RAIN_COLORS = ["#c8eeff", "#75c6f5", "#2196c4", "#65d47e",
               "#f5e642", "#f5a623", "#e84c2b", "#b01a1a", "#6b0f0f"]
RAIN_CMAP = mcolors.ListedColormap(RAIN_COLORS)
RAIN_CMAP.set_under("#f7f7f7")
RAIN_CMAP.set_over("#3d0000")
RAIN_NORM = mcolors.BoundaryNorm(RAIN_BOUNDS, ncolors=RAIN_CMAP.N)


# Precipitation colormap and colour scale for the dataset figures.
# Continuous "Blues"; PRECIP_SCALE picks the norm used by precip_norm() and
# PRECIP_NORM:
#   "log"    - LogNorm(0.1, vmax): keeps wide-range maps (annual-max vs p99) readable
#              on one shared scale. Default.
#   "linear" - Normalize(0, vmax): the plain per-panel look for narrow-range figures.
#   "banded" - discrete BoundaryNorm over RAIN_BOUNDS.
# A cell may override per call, e.g. precip_norm(vmax_max, scale="linear").
PRECIP_SCALE = "log"

import copy as _copy
PRECIP_CMAP = _copy.copy(plt.cm.Blues)   # dry pixels below vmin render near-white
PRECIP_CMAP.set_under("#f7f7f7")


def precip_norm(vmax=150.0, scale=None):
    """Return a norm for precipitation maps. scale=None follows the global
    PRECIP_SCALE; otherwise pass "log", "linear", or "banded"."""
    scale = PRECIP_SCALE if scale is None else scale
    if scale == "linear":
        return mcolors.Normalize(vmin=0.0, vmax=vmax)
    if scale == "banded":
        return mcolors.BoundaryNorm(RAIN_BOUNDS, ncolors=PRECIP_CMAP.N)
    if scale == "log":
        return mcolors.LogNorm(vmin=0.1, vmax=vmax)
    raise ValueError(f"PRECIP scale must be 'log', 'linear', or 'banded', got {scale!r}")


PRECIP_NORM = precip_norm()   # default norm, follows PRECIP_SCALE


# Cartopy static features
LAND_10M = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", facecolor="#f5f5f0", edgecolor="none", zorder=0)
COAST_10M = cfeature.NaturalEarthFeature(
    "physical", "coastline", "10m", facecolor="none", edgecolor="#222222",
    linewidth=0.8, zorder=1)
BORDERS_10M = cfeature.NaturalEarthFeature(
    "cultural", "admin_0_boundary_lines_land", "10m", facecolor="none",
    edgecolor="#555555", linewidth=0.6, zorder=2)


# Hoàng Sa / Trường Sa markers and East Sea label, placed as in the map
# utilities of results.ipynb.
_ISLES = [
    (16.50, 112.00, "Hoàng Sa", "(Paracel Is.)"),
    (9.90, 114.20, "Trường Sa", "(Spratly Is.)"),
]

# See the "Sea naming" note above §7.5 in the notebook. Changing this string
# updates every map.
SEA_LABEL = "South China Sea"


def add_vietnam_islands(ax, transform, fontsize=8, marker_size=5):
    # Hoang Sa is marked at its true position with the label offset up-right.
    # Truong Sa is only ~3° from the eastern edge, so its label is right-aligned
    # (ha="right") to grow into the map and stay on-panel in narrow multi-panel
    # figures.
    _bbox = dict(facecolor="white", alpha=0.7, edgecolor="#aaaaaa",
                 linewidth=0.3, pad=1, boxstyle="round,pad=0.3")
    for lat, lon, vname, ename in _ISLES:
        if vname == "Trường Sa":
            dot_lon, dot_lat = lon + 0.6, lat + 0.6
            ax.plot(dot_lon, dot_lat, marker="o", color="#FFEE44", markersize=marker_size,
                    markeredgecolor="#333333", markeredgewidth=0.5,
                    transform=transform, zorder=8)
            ax.text(dot_lon + 0.25, dot_lat + 0.25, f"{vname}\n{ename}",
                    transform=transform, fontsize=fontsize, color="#111111", zorder=9,
                    bbox=_bbox)
        else:
            dot_lon, dot_lat = lon, lat
            ax.plot(dot_lon, dot_lat, marker="o", color="#FFEE44", markersize=marker_size,
                    markeredgecolor="#333333", markeredgewidth=0.5,
                    transform=transform, zorder=8)
            ax.text(dot_lon + 0.25, dot_lat + 0.25, f"{vname}\n{ename}",
                    transform=transform, fontsize=fontsize, color="#111111", zorder=9,
                    bbox=_bbox)


def add_east_sea_label(ax, transform, extent=FULL_EXTENT, fontsize=11):
    """Faint diagonal sea label, shown only when it fits on-panel."""
    if extent[0] < 113 and extent[3] > 13:
        ax.text(114.0, 14.0, SEA_LABEL, fontsize=fontsize, color="#4fa3ff",
                 alpha=0.4, ha="center", va="center", rotation=25,
                 transform=transform, zorder=4)


def add_map_labels(ax, transform, extent=FULL_EXTENT, fontsize=8, marker_size=5):
    """Add the East Sea label and the island markers."""
    add_east_sea_label(ax, transform, extent, fontsize=fontsize + 3.0)
    add_vietnam_islands(ax, transform, fontsize=fontsize, marker_size=marker_size)


def add_gridlines(ax, show=None, labels=True):
    """Dashed lat/lon gridlines, respecting the global SHOW_GRIDLINES toggle.
    show=True/False overrides the global; labels=False draws lines without tick
    labels (useful for dense multi-panel figures). Returns the gridliner or None."""
    if not (SHOW_GRIDLINES if show is None else show):
        return None
    gl = ax.gridlines(draw_labels=labels, linewidth=0.4, alpha=0.5, linestyle="--")
    if labels:
        gl.top_labels = gl.right_labels = False
    return gl


def setup_map_ax(ax, proj, extent=FULL_EXTENT, gridlines=True):
    """Draw borders and coastline, set the extent, and add gridlines and labels."""
    ax.add_feature(BORDERS_10M)
    ax.add_feature(COAST_10M)
    ax.set_extent(list(extent), crs=proj)
    if gridlines:
        add_gridlines(ax)
    add_map_labels(ax, transform=proj, extent=extent)


# Data loaders, cached per process. Cells can call them in any order, and
# each still works on a cold kernel.
_land_mask_cache = {}


def get_land_mask(years=ALL_YEARS):
    if "mask" not in _land_mask_cache:
        ref_file = PROCESSED_OUTPUT / f"era5_land_tp_vietnam_{years[0]}.nc"
        ds = xr.open_dataset(ref_file)
        mask = (~ds["tp"].isel(time=0).isnull()).values.astype(bool)
        fine_lat = ds["latitude"].values
        fine_lon = ds["longitude"].values
        ds.close()
        _land_mask_cache["mask"] = (mask, fine_lat, fine_lon)
    return _land_mask_cache["mask"]


_fine_cache = {}


def load_fine_tp(years=ALL_YEARS):
    """ERA5-Land 0.1° TP (mm/hr, log1p already inverted) for the given years."""
    key = tuple(years)
    if key not in _fine_cache:
        files = [PROCESSED_OUTPUT / f"era5_land_tp_vietnam_{y}.nc" for y in years]
        ds = xr.open_mfdataset(files, combine="by_coords")
        _fine_cache[key] = np.expm1(ds["tp"])
    return _fine_cache[key]


_coarse_cache = {}


def load_coarse_tp_masked(years=ALL_YEARS):
    """ERA5 0.25° TP (mm/hr), land-masked via the ERA5-Land mask (nearest-neighbour).
    Returns (tp_coarse_mm, fine_land_da, fine_land_coarse)."""
    key = tuple(years)
    if key in _coarse_cache:
        return _coarse_cache[key]

    ref_file = PROCESSED_OUTPUT / f"era5_land_tp_vietnam_{years[0]}.nc"
    ds_ref = xr.open_dataset(ref_file)
    fine_land_da = (~ds_ref["tp"].isel(time=0).isnull()).astype("float32")
    ds_ref.close()

    coarse_list = []
    for year in years:
        fn = RAW_INPUT_TP / f"era5_sl_tp_{year}.nc"
        ds_y = xr.open_dataset(fn)
        if "valid_time" in ds_y.dims:
            ds_y = ds_y.rename({"valid_time": "time"})
        tp_c = (ds_y["tp"] * 1000.0).clip(min=0.0)
        tp_c = tp_c.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
        ds_y.close()
        coarse_list.append(tp_c)

    tp_coarse_mm = xr.concat(coarse_list, dim="time")
    fine_land_coarse = fine_land_da.interp(
        latitude=tp_coarse_mm.latitude, longitude=tp_coarse_mm.longitude, method="nearest")
    tp_coarse_mm = tp_coarse_mm.where(fine_land_coarse > 0)

    _coarse_cache[key] = (tp_coarse_mm, fine_land_da, fine_land_coarse)
    return _coarse_cache[key]


# Analysis routines shared by several figure cells
def compute_annual_stats(years=ALL_YEARS):
    """Per-year mean/p99/p99.9/max/wet-fraction + peak location. Returns df_annual."""
    records = []
    for year in years:
        fn = PROCESSED_OUTPUT / f"era5_land_tp_vietnam_{year}.nc"
        ds_y = xr.open_dataset(fn)
        tp = np.expm1(ds_y["tp"])

        total_hrs = int(tp.sizes["time"])
        mean_val = float(tp.mean())
        p99 = float(tp.quantile(0.99))
        p999 = float(tp.quantile(0.999))
        max_val = float(tp.max())
        wet_frac = float((tp > 0.1).mean())

        flat_idx = tp.argmax(dim=["time", "latitude", "longitude"])
        t_max = str(tp.time.values[int(flat_idx["time"])])[:16]
        lat_m = float(tp.latitude.values[int(flat_idx["latitude"])])
        lon_m = float(tp.longitude.values[int(flat_idx["longitude"])])
        ds_y.close()

        split = "TRAIN" if year in TRAIN_YEARS else ("VAL" if year in VAL_YEARS else "TEST")
        records.append({
            "Year": year, "Split": split, "Hours": total_hrs,
            "Mean (mm/hr)": round(mean_val, 4),
            "p99 (mm/hr)": round(p99, 2),
            "p99.9 (mm/hr)": round(p999, 2),
            "Max (mm/hr)": round(max_val, 2),
            "Wet fraction": round(wet_frac, 4),
            "Peak time (UTC)": t_max,
            "Peak lat": round(lat_m, 2),
            "Peak lon": round(lon_m, 2),
        })
    return pd.DataFrame(records)


def find_extreme_events(years=ALL_YEARS):
    """Per-year domain-max hourly event + matched coarse comparison. Returns df_extreme,
    sorted by fine max descending (row 0 = the single biggest event)."""
    tp_coarse_mm, _, _ = load_coarse_tp_masked(years)
    records = []
    for year in years:
        fn = PROCESSED_OUTPUT / f"era5_land_tp_vietnam_{year}.nc"
        ds_y = xr.open_dataset(fn)
        tp = np.expm1(ds_y["tp"])
        max_val = float(tp.max())
        flat_idx = tp.argmax(dim=["time", "latitude", "longitude"])
        t_max = str(tp.time.values[int(flat_idx["time"])])[:16]
        lat_m = float(tp.latitude.values[int(flat_idx["latitude"])])
        lon_m = float(tp.longitude.values[int(flat_idx["longitude"])])
        ds_y.close()

        t_dt = pd.Timestamp(t_max)
        tp_snap = tp_coarse_mm.sel(time=t_dt, method="nearest")
        tp_c_pt = float(tp_snap.sel(latitude=lat_m, longitude=lon_m, method="nearest").values)
        tp_c_dommax = float(tp_snap.max().values)

        records.append({
            "Year": year,
            "Time (UTC)": t_max,
            "Fine lat": round(lat_m, 1),
            "Fine lon": round(lon_m, 1),
            "Fine max (mm/hr)": round(max_val, 2),
            "Coarse @ pt (mm/hr)": round(tp_c_pt, 2),
            "Coarse dom-max (mm/hr)": round(tp_c_dommax, 2),
            "Ratio fine/coarse-max": round(max_val / (tp_c_dommax + 1e-6), 1),
        })
    return pd.DataFrame(records).sort_values("Fine max (mm/hr)", ascending=False).reset_index(drop=True)


print("viz_utils loaded "
      f"(SHOW_PEAK_STAR={SHOW_PEAK_STAR}, SEA_LABEL='{SEA_LABEL}').")
