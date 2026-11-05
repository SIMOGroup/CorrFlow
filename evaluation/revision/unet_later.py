"""The last U-Net checkpoint on the paper's hours, as a sensitivity check (no model is chosen from it).

The frozen U-Net is step 315,136. The same run went on to step 10,315,008. This writes the fields of the later checkpoint in
the predict.py format, so that score_sets.py and bootstrap.py treat it like every other method:
    <out>/unet_10315008/<split>/<zi>.npz with "mm" (1, H, W) float32, NaN outside land.
On the validation hours it also saves, per hour and for the three surviving checkpoints, the land-mean squared error in the
normalized log space, the quantity the validation loss measures (--nmse-out). Runs on CPU in float32.

    python unet_later.py --split test --allow-test --out $CORRFLOW_OUT/evaluation/preds
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "training")))
sys.path.insert(0, HERE)
for k, v in {"MODULUS_DISTRIBUTED_INITIALIZATION_METHOD": "ENV", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
             "MASTER_ADDR": "127.0.0.1"}.items():
    os.environ.setdefault(k, v)
from train_corrflow import CORRDIFF_ROOT                         # noqa: E402,F401  (sets the import path)
from score_sets import hour_set, ZARR, PAPER                     # noqa: E402
from modulus import Module                                       # noqa: E402

from paths import STATS, RUNS                                    # noqa: E402

BASE = RUNS["regression"] + "/"
CKPTS = {315136: BASE + "checkpoints_regression_best/UNet.0.315136.mdlus",
         300032: BASE + "checkpoints_regression_best/UNet.0.300032.mdlus",
         10315008: BASE + "checkpoints_regression/UNet.0.10315008.mdlus"}
LATER = 10315008
NAME = "unet_10315008"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nmse-out", default=None, help="val only: npz with the per-hour normalized MSE of all three checkpoints")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The 2025 evaluation year needs --allow-test; open it only after every checkpoint is chosen.")

    st = json.load(open(STATS))
    m_in, s_in = st["input"]["tp_coarse"]["mean"], st["input"]["tp_coarse"]["std"]
    m_out, s_out = st["output"]["tp"]["mean"], st["output"]["tp"]["std"]
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    shape = land.shape
    keys = list(CKPTS) if args.nmse_out else [LATER]
    nets = {k: Module.from_checkpoint(CKPTS[k]).eval() for k in keys}
    zis = hour_set(args.split)[0]
    if args.limit:
        zis = zis[:args.limit]
    out_dir = os.path.join(args.out, NAME, args.split)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"checkpoint": CKPTS[LATER], "step": LATER, "members": 1, "inference": "regression pass, CPU, float32"},
              open(os.path.join(args.out, NAME, "source.json"), "w"), indent=2)
    print(f"{args.split}: {len(zis)} hours, threads {torch.get_num_threads()}", flush=True)

    nmse = {k: [] for k in keys}
    t0 = time.time()
    for i, zi in enumerate(zis):
        x = np.nan_to_num(ds["tp_coarse"].isel(time=int(zi)).values, nan=0.0)
        img_lr = torch.from_numpy(((x - m_in) / (s_in + 1e-6))[None, None].astype(np.float32))
        target = None
        if args.nmse_out:
            obs = np.load(f"{PAPER}/cache/{args.split}/{zi}.npz")["truth_mm"]
            target = (np.log1p(obs) - m_out) / (s_out + 1e-6)
        for k, net in nets.items():
            with torch.no_grad():
                n = net(torch.zeros(1, 1, *shape), img_lr, torch.zeros(1)).numpy()[0, 0]
            if target is not None:
                nmse[k].append(float(((n - target)[land] ** 2).mean()))
            if k == LATER:
                mm = np.clip(np.expm1(n * (s_out + 1e-6) + m_out), 0, None)
                np.savez_compressed(os.path.join(out_dir, f"{zi}.npz"), mm=np.where(land, mm, np.nan)[None].astype(np.float32))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(zis)}  {(time.time() - t0) / (i + 1):.2f} s/hour", flush=True)
    if args.nmse_out:
        np.savez_compressed(args.nmse_out, zi=zis, time=ds["time"].values[zis].astype("datetime64[s]").astype(np.int64),
                            **{f"nmse_{k}": np.array(v) for k, v in nmse.items()})
    print(f"done: {len(zis)} hours -> {out_dir}")


if __name__ == "__main__":
    main()
