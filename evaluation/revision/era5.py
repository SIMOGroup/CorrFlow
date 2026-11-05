"""The original 0.25° ERA5 hourly rain, bilinearly interpolated in mm/hr to the ERA5-Land grid.

This is conventional interpolation of rain amounts (the Bilinear baseline). Interpolating log(1 + p) instead, as the
network input does, is not an interpolation of rain amounts. Files: paths.RAW_ERA5, `tp` in m, `valid_time` = end of
the hour (same convention as ERA5-Land and the zarr store).
"""
from functools import lru_cache

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from paths import RAW_ERA5 as RAW, ZARR


@lru_cache(maxsize=1)
def _target_points():
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    lat, lon = ds["latitude"].values, ds["longitude"].values
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return np.stack([la.ravel(), lo.ravel()], axis=-1), (len(lat), len(lon))


@lru_cache(maxsize=2)
def _year(year):
    ds = xr.open_dataset(RAW.format(year=year))
    index = {t: i for i, t in enumerate(ds["valid_time"].values.astype("datetime64[s]").astype(np.int64))}
    return ds, index


def era5_bilinear_mm(time):
    """ERA5 rain (mm) in the hour ending at `time`, on the 192 × 160 ERA5-Land grid."""
    t = np.datetime64(time, "s")
    ds, index = _year(int(str(t)[:4]))
    field = ds["tp"].isel(valid_time=index[t.astype(np.int64)]).values * 1000.0   # m to mm
    lat, lon = ds["latitude"].values, ds["longitude"].values
    if lat[0] > lat[-1]:                                                            # ascending
        lat, field = lat[::-1], field[::-1]
    pts, shape = _target_points()
    interp = RegularGridInterpolator((lat, lon), np.clip(field, 0, None))
    return interp(pts).reshape(shape).astype(np.float32)
