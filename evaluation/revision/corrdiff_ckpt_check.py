"""How do CorrDiff's three kept checkpoints rank with and without the affected 2024 hours? No checkpoint is chosen here.

CorrDiff's checkpoint (17,310,208) was chosen by a validation loss on random 2024 hours that included 1 July - 17 November
2024, the period in which ERA5 and ERA5-Land come from different ERA5 productions.

The EDM loss of ResLoss, lambda(sigma) (D(r + sigma n; c, sigma) - r)^2 averaged over land cells, on the paper's 1,000 validation
hours (621 unaffected, 379 affected). Every checkpoint sees the same inputs: four fixed noise levels, the (k + 0.5) / 4 quantiles of
ln sigma ~ N(0, 1.2^2), and the same noise per hour and level. Paired differences to 17,310,208 carry 95% day-block intervals.
Runs on CPU in float32.

    python corrdiff_ckpt_check.py --out <folder>
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "training")))
sys.path.insert(0, HERE)
for k, v in {"MODULUS_DISTRIBUTED_INITIALIZATION_METHOD": "ENV", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
             "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29513"}.items():
    os.environ.setdefault(k, v)
from train_corrflow import CORRDIFF_ROOT                                # noqa: E402
from fixed_validation import build_fixed_val_set                         # noqa: E402
from score_sets import hour_set                                          # noqa: E402
from modulus import Module                                               # noqa: E402
from modulus.distributed import DistributedManager                       # noqa: E402
from datasets.dataset import init_train_valid_datasets_from_config       # noqa: E402

from paths import RUNS                                                  # noqa: E402

BASE = RUNS["corrdiff"] + "/"
CKPTS = {17310208: BASE + "checkpoints_diffusion_best/EDMPrecondSR.0.17310208.mdlus",
         16455168: BASE + "checkpoints_diffusion_best/EDMPrecondSR.0.16455168.mdlus",
         22440192: BASE + "checkpoints_diffusion/EDMPrecondSR.0.22440192.mdlus"}
CHOSEN = 17310208
EXTRA_HOUR = 64959                     # the cached 2024 set has this one extra hour, not in the paper's 1,000
AFFECTED = (np.datetime64("2024-07-01T00:00:00"), np.datetime64("2024-11-18T00:00:00"))
P_MEAN, P_STD, SIGMA_DATA = 0.0, 1.2, 0.5
N_SIGMA = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N hours only (smoke test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    DistributedManager.initialize()
    device = torch.device("cpu")
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(version_base="1.2", config_dir=os.path.join(CORRDIFF_ROOT, "conf")):
        cfg = compose(config_name="vietnam_config_training_corrflow_revision")
    OmegaConf.resolve(cfg)
    dataset, _, val_ds, _ = init_train_valid_datasets_from_config(
        OmegaConf.to_container(cfg.dataset), {"pin_memory": False, "num_workers": 2},
        batch_size=8, seed=0, validation_dataset_cfg=OmegaConf.to_container(cfg.validation), train_test_split=True)
    land = dataset.land_mask.to(device)[0] > 0.5
    all_times = np.array(val_ds.time(), dtype="datetime64[s]")

    import xarray as xr
    from score_sets import ZARR
    zis = np.array([z for z in hour_set("val")[0] if z != EXTRA_HOUR])
    times = xr.open_dataset(ZARR, engine="zarr", chunks=None)["time"].values[zis].astype("datetime64[s]")
    if args.limit:
        step = max(1, len(zis) // args.limit)      # spread over the year, so both periods appear
        zis, times = zis[::step][:args.limit], times[::step][:args.limit]
    pos = {t: i for i, t in enumerate(all_times)}
    idx = [pos[t] for t in times]
    assert np.array_equal(all_times[idx], times)
    aff = (times >= AFFECTED[0]) & (times < AFFECTED[1])
    print(f"{len(idx)} hours, {int(aff.sum())} affected, {int((~aff).sum())} unaffected, threads {torch.get_num_threads()}", flush=True)

    reg = Module.from_checkpoint(cfg.training.io.regression_checkpoint_path).eval().requires_grad_(False).to(device)
    t0 = time.time()
    vs = build_fixed_val_set(torch.utils.data.Subset(val_ds, idx), reg, device, cfg.model.hr_mean_conditioning,
                             torch.bfloat16, False, batch_size=8, num_workers=2)
    print(f"residuals and conditions built in {time.time() - t0:.0f} s; hr_mean_conditioning {cfg.model.hr_mean_conditioning}", flush=True)

    sigmas = np.exp(P_MEAN + P_STD * norm.ppf((np.arange(N_SIGMA) + 0.5) / N_SIGMA))
    lam = (sigmas ** 2 + SIGMA_DATA ** 2) / (sigmas * SIGMA_DATA) ** 2
    print("sigma", np.round(sigmas, 4), "lambda", np.round(lam, 4), flush=True)

    loss = {}
    for step, path in CKPTS.items():
        net = Module.from_checkpoint(path).eval().requires_grad_(False).to(device)
        rec = np.zeros((len(idx), N_SIGMA))
        t0 = time.time()
        with torch.no_grad():
            for i in range(len(idx)):
                r = vs["residual"][i:i + 1].to(device)
                c = vs["cond"][i:i + 1].to(device)
                for j, s in enumerate(sigmas):
                    n = torch.randn(r.shape, generator=torch.Generator().manual_seed(2024 + 10 * i + j))
                    d = net(r + float(s) * n, c, torch.full((1,), float(s), device=device), None)
                    rec[i, j] = float(lam[j]) * ((d - r)[0, 0][land] ** 2).mean().item()
                if (i + 1) % 100 == 0:
                    print(f"  {step:,}: {i + 1}/{len(idx)}  {(time.time() - t0) / (i + 1):.2f} s/hour", flush=True)
        loss[step] = rec
        print(f"{step:,} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(os.path.join(args.out, "corrdiff_ckpt_loss.npz"), zi=zis, time=times.astype(np.int64), affected=aff,
                        sigmas=sigmas, **{f"loss_{k}": v for k, v in loss.items()})

    days, inv = np.unique(times.astype("datetime64[D]"), return_inverse=True)
    rng = np.random.default_rng(2026)
    w = np.stack([np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))[inv] for _ in range(2000)]).astype(float)
    for label, m in (("all hours", np.ones(len(idx), bool)), ("unaffected", ~aff), ("affected", aff)):
        if not m.any():
            continue
        print(f"\n{label}: {int(m.sum())} hours")
        base = loss[CHOSEN].mean(1)
        for step in CKPTS:
            v = loss[step].mean(1)
            line = f"  {step:>11,}  mean loss {v[m].mean():.5f}  per level " + " ".join(f"{x:.5f}" for x in loss[step][m].mean(0))
            if step != CHOSEN:
                ww = w[:, m]
                ok = ww.sum(1) > 0
                d = ((ww * (v - base)[m]).sum(1)[ok] / ww.sum(1)[ok])
                line += f"  minus 17,310,208: {(v - base)[m].mean():+.5f} [{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}]"
            print(line, flush=True)
    print("done")


if __name__ == "__main__":
    main()
