"""Inference cost of every configuration in one job on one GPU: mean, spread and range of the latency.

The timed region is the paper's (solver_ablation.py): the shared regression pass followed by the residual
corrector, bracketed by torch.cuda.synchronize(); input preparation, host transfer and scores are excluded. All
configurations run in the same job, on the same hours; for every hour they run in a shuffled order, so a change in
the load of the shared node affects all of them alike. Each configuration is warmed up first (cuDNN autotuning,
lazy initialization). Precision as in the predictions: CorrFlow and the deterministic corrector in bfloat16
autocast, regression U-Net and CorrDiff as in the paper's pipeline.

Configurations: U-Net alone; deterministic corrector; CorrFlow Euler 1 step with 1 and 16 members, Euler 5/10/25
and Heun 5/10 steps with 16 members; CorrDiff Heun 10/25/50/100 steps with 16 members and 25/50 steps with 1 member.

    python timing.py --split test --allow-test --hours 100
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from functools import partial

import numpy as np
import torch
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "training")))
sys.path.insert(0, HERE)
from train_corrflow import CORRDIFF_ROOT, FlowUNet, DetUNet            # noqa: E402
from score_sets import hour_set, ZARR                                   # noqa: E402
from predict import best_checkpoint, REG_CKPT, STATS                   # noqa: E402

from paths import DIFF_CKPT, FLOW_CKPT, RUNS, EVAL                    # noqa: E402

DET_RUN = RUNS["corrdet_s1"]
OUT = os.path.join(EVAL, "timing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--hours", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma-separated labels (smoke test)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The 2025 evaluation year is closed until every checkpoint is chosen; add --allow-test only then.")
    for k, v in {"MODULUS_DISTRIBUTED_INITIALIZATION_METHOD": "ENV", "RANK": "0",
                 "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1"}.items():
        os.environ.setdefault(k, v)

    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from modulus import Module
    from modulus.distributed import DistributedManager
    from modulus.utils.corrdiff import diffusion_step
    from modulus.utils.generative import deterministic_sampler
    DistributedManager.initialize()
    device = torch.device("cuda")

    zis, tags = hour_set(args.split)
    hours = np.sort(np.random.default_rng(2026).choice(zis[tags["sample"]], size=args.hours, replace=False))
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    shape = land.shape
    st = json.load(open(STATS))
    m_in, s_in = st["input"]["tp_coarse"]["mean"], st["input"]["tp_coarse"]["std"]

    with initialize_config_dir(version_base="1.2", config_dir=os.path.join(CORRDIFF_ROOT, "conf")):
        cfg_rev = compose(config_name="vietnam_config_training_corrflow_revision")
        cfg_gen = compose(config_name="vietnam_config_generate")
    backbone = OmegaConf.to_container(cfg_rev.model.model_args)
    backbone.setdefault("checkpoint_level", 0)

    def load(net, ckpt):
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
        return net.eval().to(device)

    reg = Module.from_checkpoint(REG_CKPT).eval().to(device)
    flow = load(FlowUNet(shape, 1 + 1 + 2, 1, **backbone).to(device), FLOW_CKPT)
    det_ckpt = best_checkpoint(DET_RUN)[0]
    det = load(DetUNet(shape, 2, 1, **backbone).to(device), det_ckpt)
    diff = Module.from_checkpoint(DIFF_CKPT).eval().to(device)

    @torch.no_grad()
    def unet(img_lr):
        return reg(torch.zeros(1, 1, *shape, device=device), img_lr, torch.zeros(1, device=device))

    @torch.no_grad()
    def deterministic(img_lr):
        mu = unet(img_lr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return mu + det(torch.cat([mu, img_lr], dim=1)).float()

    @torch.no_grad()
    def corrflow(img_lr, steps, solver="euler", k=16):
        mu = unet(img_lr)
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
                v1 = flow(r, t, cond).float()
                if solver == "heun" and s < steps - 1:
                    v2 = flow(r + dt * v1, t + dt, cond).float()
                    r = r + dt * 0.5 * (v1 + v2)
                else:
                    r = r + dt * v1
        return mu.expand(k, -1, -1, -1) + r

    @torch.no_grad()
    def corrdiff(img_lr, steps, k=16):
        mu = unet(img_lr)
        cond = torch.cat([mu, img_lr], dim=1) if cfg_gen.generation.hr_mean_conditioning else img_lr
        res = diffusion_step(net=diff, sampler_fn=partial(deterministic_sampler, num_steps=steps, solver="heun"),
                             seed_batch_size=k, img_shape=shape, img_out_channels=1,
                             rank_batches=[torch.arange(k)],
                             img_lr=cond.expand(k, -1, -1, -1).to(memory_format=torch.channels_last),
                             rank=0, device=device, hr_mean=None)
        return mu.expand(k, -1, -1, -1) + res

    configs = [("unet", "U-Net", "-", 0, 1, 1, unet),
               ("deterministic", "Deterministic corrector", "-", 0, 1, 1, deterministic),
               ("corrflow_euler1_k1", "CorrFlow", "Euler", 1, 1, 1, partial(corrflow, steps=1, k=1)),
               ("corrflow_euler1_k16", "CorrFlow", "Euler", 1, 1, 16, partial(corrflow, steps=1))]
    configs += [(f"corrflow_euler{s}_k16", "CorrFlow", "Euler", s, s, 16, partial(corrflow, steps=s)) for s in (5, 10, 25)]
    configs += [(f"corrflow_heun{s}_k16", "CorrFlow", "Heun", s, 2 * s - 1, 16, partial(corrflow, steps=s, solver="heun"))
                for s in (5, 10)]
    configs += [(f"corrdiff_heun{s}_k16", "CorrDiff", "Heun", s, 2 * s - 1, 16, partial(corrdiff, steps=s))
                for s in (10, 25, 50, 100)]
    configs += [(f"corrdiff_heun{s}_k1", "CorrDiff", "Heun", s, 2 * s - 1, 1, partial(corrdiff, steps=s, k=1))
                for s in (25, 50)]
    if args.only:
        keep = args.only.split(",")
        configs = [c for c in configs if c[0] in keep]

    def img(zi):
        x = np.nan_to_num(ds["tp_coarse"].isel(time=int(zi)).values, nan=0.0)
        return torch.from_numpy(((x - m_in) / (s_in + 1e-6))[None, None].astype(np.float32)).to(device)

    import warnings
    warnings.simplefilter("ignore")
    x0 = img(hours[0])
    for c in configs:                                            # warm-up, not timed
        for _ in range(args.warmup):
            c[-1](x0)
    torch.cuda.synchronize()

    times = {c[0]: [] for c in configs}
    rng = np.random.default_rng(31)
    t_start = time.time()
    for n, zi in enumerate(hours):
        x = img(zi)
        for j in rng.permutation(len(configs)):
            label, fn = configs[j][0], configs[j][-1]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(x)
            torch.cuda.synchronize(); times[label].append(time.perf_counter() - t0)
        if n == 0 or (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(hours)} hours  {(time.time() - t_start) / 60:.1f} min", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for label, model, solver, steps, nfe, k, _ in configs:
        a = np.array(times[label])
        rows.append({"config": label, "model": model, "solver": solver, "steps": steps, "nfe_per_member": nfe,
                     "members": k, "hours": len(a), "mean_s": f"{a.mean():.5f}", "sd_s": f"{a.std(ddof=1):.5f}",
                     "median_s": f"{np.median(a):.5f}", "p05_s": f"{np.quantile(a, 0.05):.5f}",
                     "p95_s": f"{np.quantile(a, 0.95):.5f}", "min_s": f"{a.min():.5f}", "max_s": f"{a.max():.5f}"})
        print(f"{label:<24} mean {a.mean():.4f} s  sd {a.std(ddof=1):.4f}  median {np.median(a):.4f}  "
              f"[{a.min():.4f}, {a.max():.4f}]", flush=True)
    with open(os.path.join(args.out, "timing.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(os.path.join(args.out, "timing_per_hour.npz"), hours=hours,
                        **{k: np.array(v) for k, v in times.items()})
    json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "host": platform.node(), "hours": len(hours),
               "warmup_calls_per_config": args.warmup, "flow_checkpoint": FLOW_CKPT, "det_checkpoint": det_ckpt,
               "corrdiff_checkpoint": DIFF_CKPT, "regression_checkpoint": REG_CKPT},
              open(os.path.join(args.out, "timing_env.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
