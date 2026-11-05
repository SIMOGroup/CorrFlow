"""Solver and step-count sweep (--subset sample: random hours; heavy: heavy-rain hours) with spread, and the ensemble-size check.

On a fixed random subset of the unconditional sample hours, for each setting, per-hour scores
against ERA5-Land are written in the score_sets.py format (scores/<split>_steps_<label>.npz), so
bootstrap.py can compare settings on the same hours:

  CorrFlow (each run given):  Euler 1, 5, 10, 25 steps; Heun 5, 10 steps (2 network calls per step);
                              Euler 10 with ocean pixels reset to 0 after every step (does the
                              unsupervised ocean state influence land?);  Euler 1 with 64 members.
  CorrDiff:                   Heun 10, 25, 50, 100 steps (50 = the paper's cached predictions);
                              Heun 50 with 64 members (first --hours-k64 hours).
CorrFlow members use noise seed k for member k, as in predict.py; CorrDiff seeds 0..K-1, as in the
paper's pipeline. Also logs the GPU time per hour (the paper's latencies come from timing.py).

    python steps_sweep.py --split test --allow-test --hours 300 \
        --flow corrflow_s0=$CORRFLOW_OUT/corrflow_seed0/checkpoints_flow_best/FlowUNet.0.20620032.pt \
        --flow corrflow_s1=run:$CORRFLOW_OUT/corrflow_seed1
"""
import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "training")))
sys.path.insert(0, HERE)
from train_corrflow import CORRDIFF_ROOT, FlowUNet                     # noqa: E402
from score_sets import hour_set, PAPER, ZARR                             # noqa: E402
from scores import (hour_scores, stack, INTENSITY_EDGES, RELIABILITY_THRESHOLDS,  # noqa: E402
                    COVERAGE_LEVELS, CONTINGENCY_THRESHOLDS,
                    FSS_SIZES, VARIOGRAM_LAGS, VARIOGRAM_P)
from predict import best_checkpoint, REG_CKPT, STATS                   # noqa: E402

from paths import DIFF_CKPT, EVAL                                       # noqa: E402

OUT = os.path.join(EVAL, "scores")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--hours", type=int, default=300)
    ap.add_argument("--subset", choices=["sample", "heavy"], default="sample",
                    help="sample: random unconditional hours; heavy: heavy-rain hours")
    ap.add_argument("--hours-k64", type=int, default=100)
    ap.add_argument("--flow", action="append", default=[], help="name=<checkpoint> or name=run:<folder>")
    ap.add_argument("--no-corrdiff", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The 2025 evaluation year is closed until every checkpoint is chosen; add --allow-test only then.")
    for k, v in {"MODULUS_DISTRIBUTED_INITIALIZATION_METHOD": "ENV", "RANK": "0",
                 "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1"}.items():
        os.environ.setdefault(k, v)

    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf, open_dict
    from modulus import Module
    from modulus.distributed import DistributedManager
    from modulus.utils.corrdiff import diffusion_step
    from modulus.utils.generative import deterministic_sampler
    DistributedManager.initialize()
    device = torch.device("cuda")

    zis, tags = hour_set(args.split)
    pool = zis[tags[args.subset]]                               # random (unconditional) or heavy-rain hours
    hours = np.sort(np.random.default_rng(2026).choice(pool, size=min(args.hours, len(pool)), replace=False))
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    times = ds["time"].values[hours]
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    shape = land.shape
    st = json.load(open(STATS))
    m_in, s_in = st["input"]["tp_coarse"]["mean"], st["input"]["tp_coarse"]["std"]
    m_out, s_out = st["output"]["tp"]["mean"], st["output"]["tp"]["std"]
    to_mm = lambda z: np.where(land[None], np.clip(np.expm1(z * (s_out + 1e-6) + m_out), 0, None), np.nan)

    with initialize_config_dir(version_base="1.2", config_dir=os.path.join(CORRDIFF_ROOT, "conf")):
        cfg_rev = compose(config_name="vietnam_config_training_corrflow_revision")
        cfg_gen = compose(config_name="vietnam_config_generate")
    backbone = OmegaConf.to_container(cfg_rev.model.model_args)
    backbone.setdefault("checkpoint_level", 0)
    reg = Module.from_checkpoint(REG_CKPT).eval().to(device)
    land_t = torch.from_numpy(land).to(device)

    def inputs(zi):
        x = np.nan_to_num(ds["tp_coarse"].isel(time=int(zi)).values, nan=0.0)
        img_lr = torch.from_numpy(((x - m_in) / (s_in + 1e-6))[None, None].astype(np.float32)).to(device)
        mu = reg(torch.zeros(1, 1, *shape, device=device), img_lr, torch.zeros(1, device=device))
        return img_lr, mu

    @torch.no_grad()
    def corrflow(net, img_lr, mu, steps, solver="euler", k=16, ocean_reset=False):
        cond = torch.cat([mu, img_lr], dim=1).expand(k, -1, -1, -1)
        g = torch.Generator(device=device)
        r = torch.empty(k, 1, *shape, device=device)
        for i in range(k):
            g.manual_seed(i)
            r[i] = torch.randn(1, *shape, generator=g, device=device)
        dt = 1.0 / steps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(steps):
                t = torch.full((k,), s * dt, device=device)
                v1 = net(r, t, cond).float()
                if solver == "heun" and s < steps - 1:
                    v2 = net(r + dt * v1, t + dt, cond).float()
                    r = r + dt * 0.5 * (v1 + v2)
                else:
                    r = r + dt * v1
                if ocean_reset:
                    r = torch.where(land_t, r, torch.zeros_like(r))
        return (mu.expand(k, -1, -1, -1) + r).float()

    @torch.no_grad()
    def corrdiff(net_diff, img_lr, mu, steps, k=16):
        cond = torch.cat([mu, img_lr], dim=1) if cfg_gen.generation.hr_mean_conditioning else img_lr
        b = min(k, 16)                              # members in batches of 16, as in the paper
        res = diffusion_step(net=net_diff, sampler_fn=partial(deterministic_sampler, num_steps=steps, solver="heun"),
                             seed_batch_size=b, img_shape=shape, img_out_channels=1,
                             rank_batches=[torch.arange(i, i + b) for i in range(0, k, b)],
                             img_lr=cond.expand(b, -1, -1, -1), rank=0, device=device, hr_mean=None)
        return mu.expand(k, -1, -1, -1) + res

    settings = []                                  # (label, fn(img_lr, mu) -> (K,H,W) z, n_hours)
    for spec in args.flow:
        name, src = spec.split("=", 1)
        ckpt = best_checkpoint(src[4:])[0] if src.startswith("run:") else src
        net = FlowUNet(shape, 1 + 1 + 2, 1, **backbone).to(device)
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
        net.eval()
        print(f"{name}: {ckpt}", flush=True)
        for steps in (1, 5, 10, 25):
            settings.append((f"{name}_euler{steps}", partial(corrflow, net, steps=steps), len(hours)))
        for steps in (5, 10):
            settings.append((f"{name}_heun{steps}", partial(corrflow, net, steps=steps, solver="heun"), len(hours)))
        settings.append((f"{name}_euler10_oceanreset", partial(corrflow, net, steps=10, ocean_reset=True), len(hours)))
        settings.append((f"{name}_euler1_k64", partial(corrflow, net, steps=1, k=64), len(hours)))
    if not args.no_corrdiff:
        with open_dict(cfg_gen):
            cfg_gen.generation.num_ensembles = 16
        net_diff = Module.from_checkpoint(DIFF_CKPT).eval().to(device)
        for steps in (10, 25, 50, 100):
            settings.append((f"corrdiff_heun{steps}", partial(corrdiff, net_diff, steps=steps), len(hours)))
        settings.append(("corrdiff_heun50_k64", partial(corrdiff, net_diff, steps=50, k=64), args.hours_k64))

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(12345)
    for label, fn, n_h in settings:
        prefix = f"{args.split}_steps" + ("" if args.subset == "sample" else "_heavy")
        path = os.path.join(args.out, f"{prefix}_{label}.npz")
        if os.path.exists(path):
            print(f"exists: {path}"); continue
        recs, gpu_s = [], []
        t_start = time.time()
        for zi in hours[:n_h]:
            obs = np.load(f"{PAPER}/cache/{args.split}/{zi}.npz")["truth_mm"]
            img_lr, mu = inputs(zi)
            torch.cuda.synchronize(); t0 = time.time()
            z = fn(img_lr, mu)
            torch.cuda.synchronize(); gpu_s.append(time.time() - t0)
            recs.append(hour_scores(obs, to_mm(z.cpu().numpy()[:, 0]).astype(np.float32), land, rng))
        sel = np.arange(n_h)
        np.savez_compressed(path, zi=hours[sel], time=times[sel].astype("datetime64[s]").astype(np.int64),
                            tag_sample=np.ones(n_h, bool), subset=args.subset, gpu_seconds=np.array(gpu_s), source=label,
                            intensity_edges=INTENSITY_EDGES, reliability_thresholds=RELIABILITY_THRESHOLDS,
                            coverage_levels=COVERAGE_LEVELS, contingency_thresholds=CONTINGENCY_THRESHOLDS,
                            fss_sizes=FSS_SIZES, variogram_lags=VARIOGRAM_LAGS, variogram_p=VARIOGRAM_P,
                            **stack(recs))
        print(f"{label:<32} {n_h} hours  {np.median(gpu_s):.3f} s/hour (GPU)  "
              f"{(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
