"""IMERG / GSMaP on the ERA5-Land 0.1° grid and hours, as an independent reference.

Grid: the satellite cells (centres at x.x5°, see preprocessing/download_independent.py) are offset by 0.05° from
the ERA5-Land cells (centres at whole tenths), so each ERA5-Land cell overlaps exactly four
satellite cells, a quarter each: its value is their mean (exact conservative regridding).

Time: ERA5 and ERA5-Land label an hour by its END (value at T = rain in [T-1h, T)); IMERG and
GSMaP label a step by its START. So the hour ending at T is
    IMERG: mean of the two half-hourly rates starting at T-60 and T-30 min (mm/hr = mm in the hour)
    GSMaP: the hourly rate starting at T-60 min (gauge-adjusted band hourlyPrecipRateGC)
IMERG is used only where both half-hours are Final run ("permanent"); otherwise None is returned.
"""
import datetime as dt
import os
from functools import lru_cache

import numpy as np

from paths import SATELLITE as DATA
# Target grid (vietnam_data.zarr): lat 25.0 to 5.9 (192), lon 102.1 to 118.0 (160).
# Satellite box: rows 25.95 to 5.05 (210), cols 101.05 to 118.95 (180).
R0, C0, NY, NX = 9, 10, 192, 160          # first satellite row/col touching the target grid


def to_target_grid(field):
    """(210, 180) satellite field to the (192, 160) ERA5-Land grid (mean of the 4 overlapping cells)."""
    a = field[R0:R0 + NY + 1, C0:C0 + NX + 1]
    return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])


@lru_cache(maxsize=8)
def _day(product, yyyymmdd):
    path = os.path.join(DATA, product, f"{yyyymmdd}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    precip = z["precip"]
    if product == "gsmap":
        precip = precip[:, list(z["bands"]).index("hourlyPrecipRateGC")]
    starts = {int(t): i for i, t in enumerate(z["time_start"])}
    return precip, starts, z["status"]


def _step(product, start):
    """Field (210, 180) of the step starting at `start` (UTC datetime), with its status."""
    day = _day(product, start.strftime("%Y%m%d"))
    if day is None:
        return None, None
    precip, starts, status = day
    i = starts.get(int(start.replace(tzinfo=dt.timezone.utc).timestamp() * 1000))
    return (None, None) if i is None else (precip[i], status[i])


def reference_hour(product, end):
    """Rain (mm) in the hour ending at `end` (numpy datetime64 or datetime), on the target grid."""
    end = dt.datetime.utcfromtimestamp(np.datetime64(end, "s").astype(np.int64))
    if product == "imerg":
        parts = [_step("imerg", end - dt.timedelta(minutes=m)) for m in (60, 30)]
        if any(f is None or s != "permanent" for f, s in parts):
            return None                      # missing, or Late run (not used)
        return to_target_grid(0.5 * (parts[0][0] + parts[1][0]))
    if product == "gsmap":
        f, s = _step("gsmap", end - dt.timedelta(hours=1))
        return None if f is None else to_target_grid(f)
    raise ValueError(product)
