"""Predictions of a training run (a CorrFlow seed or the deterministic corrector CorrDet) on the paper's hours.

The checkpoint is the one with the lowest fixed validation loss in <run>/val_history.csv,
unless --checkpoint is given. CorrFlow: one Euler step from t = 0, 16 members, member k drawn
with noise seed k: the same inference noise as the cached CorrFlow seed 0 predictions
(cache_extremes.py, cache_sample.py), so runs differ only by training. CorrDet: one member.

Writes <out>/<name>/<split>/<zi>.npz with "mm" (K, 192, 160) float32, NaN outside land.
The hours are those of score_sets.py; --split test needs --allow-test.

    python predict.py --run $CORRFLOW_OUT/corrflow_seed1 --name corrflow_s1 --split val
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "training")))
sys.path.insert(0, HERE)
from train_corrflow import CORRDIFF_ROOT, FlowUNet, DetUNet          # noqa: E402
from score_sets import hour_set, ZARR                                 # noqa: E402
from modulus import Module                                            # noqa: E402
from omegaconf import OmegaConf                                       # noqa: E402

from paths import STATS, REG_CKPT, EVAL                               # noqa: E402

OUT = os.path.join(EVAL, "preds")
K_ENS = 16


def best_checkpoint(run):
    """Checkpoint with the lowest fixed validation loss (falls back to all checks if none fixed)."""
    rows = list(csv.DictReader(open(os.path.join(run, "val_history.csv"))))
    fixed = [r for r in rows if r.get("fixed", "1") == "1"] or rows
    best = min(fixed, key=lambda r: float(r["val_loss"]))
    step = int(best["step"])
    # Weights only (FlowUNet / DetUNet): checkpoint.0.<step>.pt beside them is the optimizer state
    hits = sorted(f for f in glob.glob(os.path.join(run, "checkpoints_flow*", "*UNet.0.*.pt"))
                  if int(re.search(r"\.(\d+)\.pt$", f).group(1)) == step)
    if not hits:
        sys.exit(f"No checkpoint file for best step {step} in {run}")
    return hits[0], step, float(best["val_loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder (with val_history.csv)")
    ap.add_argument("--name", required=True, help="e.g. corrflow_s1, deterministic_s1 (CorrDet)")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--checkpoint", default=None, help="override the automatic choice")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The 2025 evaluation year is closed until every checkpoint is chosen; add --allow-test only then.")

    for k, v in {"MODULUS_DISTRIBUTED_INITIALIZATION_METHOD": "ENV", "RANK": "0",
                 "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1"}.items():
        os.environ.setdefault(k, v)
    device = torch.device("cuda")

    if args.checkpoint:
        ckpt, step, vloss = args.checkpoint, int(re.search(r"\.(\d+)\.pt$", args.checkpoint).group(1)), None
    else:
        ckpt, step, vloss = best_checkpoint(args.run)
    det = os.path.basename(ckpt).startswith("DetUNet")
    print(f"{args.name}: {ckpt} (step {step:,}, fixed val {vloss})", flush=True)

    from hydra import initialize_config_dir, compose
    with initialize_config_dir(version_base="1.2", config_dir=os.path.join(CORRDIFF_ROOT, "conf")):
        cfg = compose(config_name="vietnam_config_training_corrflow_revision")
    backbone = OmegaConf.to_container(cfg.model.model_args)
    backbone.setdefault("checkpoint_level", 0)

    st = json.load(open(STATS))
    m_in, s_in = st["input"]["tp_coarse"]["mean"], st["input"]["tp_coarse"]["std"]
    m_out, s_out = st["output"]["tp"]["mean"], st["output"]["tp"]["std"]
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    shape = land.shape

    reg = Module.from_checkpoint(REG_CKPT).eval().to(device)
    c_cond = 2                                     # regression mean + coarse input
    net = (DetUNet(shape, c_cond, 1, **backbone) if det
           else FlowUNet(shape, 1 + 1 + c_cond, 1, **backbone)).to(device)
    state = torch.load(ckpt, map_location=device)
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
    net.eval()

    zis, _ = hour_set(args.split)
    if args.limit:
        zis = zis[:args.limit]
    out_dir = os.path.join(args.out, args.name, args.split)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"checkpoint": ckpt, "step": step, "fixed_val_loss": vloss, "members": 1 if det else K_ENS,
               "inference": "deterministic" if det else "one Euler step from t=0, member k noise seed k"},
              open(os.path.join(args.out, args.name, "source.json"), "w"), indent=2)

    t0 = time.time()
    with torch.no_grad():
        for i, zi in enumerate(zis):
            path = os.path.join(out_dir, f"{zi}.npz")
            if os.path.exists(path):
                continue
            x = np.nan_to_num(ds["tp_coarse"].isel(time=int(zi)).values, nan=0.0)
            img_lr = torch.from_numpy(((x - m_in) / (s_in + 1e-6))[None, None].astype(np.float32)).to(device)
            mu = reg(torch.zeros(1, 1, *shape, device=device), img_lr, torch.zeros(1, device=device))
            cond = torch.cat([mu, img_lr], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if det:
                    pred = mu + net(cond).float()
                else:
                    g = torch.Generator(device=device)
                    r = torch.empty(K_ENS, 1, *shape, device=device)
                    for k in range(K_ENS):
                        g.manual_seed(k)
                        r[k] = torch.randn(1, *shape, generator=g, device=device)
                    v0 = net(r, torch.zeros(K_ENS, device=device), cond.expand(K_ENS, -1, -1, -1))
                    pred = mu.expand(K_ENS, -1, -1, -1) + r + v0.float()
            mm = np.clip(np.expm1(pred.float().cpu().numpy()[:, 0] * (s_out + 1e-6) + m_out), 0, None)
            np.savez_compressed(path, mm=np.where(land[None], mm, np.nan).astype(np.float32))
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(zis)}  {(time.time() - t0) / (i + 1):.3f} s/hour", flush=True)
    print(f"done: {len(zis)} hours in {out_dir}")


if __name__ == "__main__":
    main()
