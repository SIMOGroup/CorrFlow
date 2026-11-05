"""Copy score files under a new split label with two extra tags for bootstrap.py --subset:
tag_affected = valid time in the period in which ERA5 and ERA5-Land come from different ERA5
production runs (1 July - 17 November 2024; ECMWF re-ran the final ERA5 after an ERA5T snow-depth
issue and did not reproduce ERA5-Land), tag_clean = all other hours (or days).

    python period_tags.py --scores $CORRFLOW_OUT/evaluation/scores --src val --dst val2024
"""
import argparse
import glob
import os

import numpy as np

AFFECTED = (np.datetime64("2024-07-01T00:00:00"), np.datetime64("2024-11-18T00:00:00"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--src", required=True, help="split label of the files to copy, e.g. val")
    ap.add_argument("--dst", required=True, help="new split label, e.g. val2024")
    ap.add_argument("--names", nargs="*", default=None, help="only these method names (default: all)")
    ap.add_argument("--any-year", action="store_true",
                    help="apply the same calendar window (1 July - 17 November) to every year, e.g. to compare "
                         "2025 hours from the same months as the clean 2024 period")
    ap.add_argument("--within", default=None, help="limit both tags to an existing subset, e.g. sample")
    ap.add_argument("--drop-zi", type=int, nargs="*", default=[],
                    help="hours (zarr index) left out of every subset, e.g. 64959, the one cached 2024 hour "
                         "outside the 1,000-hour validation sample")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.scores, f"{args.src}_*.npz")))
    if args.names:
        files = [f for f in files if os.path.basename(f)[len(args.src) + 1:-4].split("_vs_")[0] in args.names]
    for f in files:
        z = dict(np.load(f))
        t = z["time"].astype("datetime64[s]")
        if args.any_year:
            md = (t - t.astype("datetime64[Y]")).astype("timedelta64[s]")      # time since 1 January
            yr = t.astype("datetime64[Y]")
            start = (np.datetime64("2024-07-01") - np.datetime64("2024-01-01")).astype("timedelta64[s]")
            stop = (np.datetime64("2024-11-18") - np.datetime64("2024-01-01")).astype("timedelta64[s]")
            leap = ((yr.astype(int) + 1970) % 4 == 0)                           # 2017-2025: rule holds
            shift = np.where(leap, 0, 86400).astype("timedelta64[s]")           # non-leap years: one day less
            aff = (md >= start - shift) & (md < stop - shift)
        else:
            aff = (t >= AFFECTED[0]) & (t < AFFECTED[1])
        keep = ~np.isin(z["zi"], args.drop_zi) if "zi" in z else np.ones(len(t), bool)
        if args.within:
            keep &= z[f"tag_{args.within}"].astype(bool)
        z["tag_affected"], z["tag_clean"] = aff & keep, ~aff & keep
        if "tag_sample" in z:
            z["tag_sample"] = z["tag_sample"].astype(bool) & keep
        out = os.path.join(args.scores, args.dst + os.path.basename(f)[len(args.src):])
        np.savez_compressed(out, **z)
        print(f"{os.path.basename(out)}: {int((aff & keep).sum())} affected, {int((~aff & keep).sum())} clean, "
              f"{int((~keep).sum())} dropped ({str(t.min())[:10]} to {str(t.max())[:10]})")


if __name__ == "__main__":
    main()
