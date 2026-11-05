"""Noise cancellation and sensitivity near t = 0: does the trained one-step corrector do what Proposition 1 says?

Proposition 1: with independent coupling and the linear path, the Bayes velocity at t = 0 is
v*(r0, 0, c) = E[r1 | c] - r0, so the one-step endpoint r0 + v*(r0, 0, c) = E[r1 | c] does not depend on the
noise r0. The trained networks are checked directly on random 2024 validation hours outside 1 July - 17 November
2024 (residuals and conditioning built exactly as for the fixed validation loss), in the normalized residual space
the network works in, over land pixels, with K = 16 members (noise seed k for member k, as in predict.py).

A. t = 0, the one-step sampler (float32, and bfloat16 autocast as used for the predictions):
   beta       slope of the velocity on the noise, sum (v - v_bar)(r0 - r0_bar) / sum (r0 - r0_bar)^2 over members
              and land pixels (-1: the velocity cancels the noise exactly; 0: it ignores it);
   retained   sqrt(mean member variance of the endpoint / mean member variance of r0)
              (0: the endpoint does not depend on the noise; 1: the noise passes through unchanged);
   linshare   (1 + beta)^2 var(r0) / var(endpoint), the share of the endpoint spread that is a scaled copy of the noise;
   rmse, ssr  RMSE of the member mean against the target residual r1 and the spread-skill ratio
              sqrt((K + 1) / K) * spread / RMSE (as in the paper), for scale.
B. Starting times t near 0 (float32). The state is built as in training, r_t = (1 - t) r0 + t r1, and the endpoint is
   r_t + (1 - t) v(r_t, t, c):  retained(t) = endpoint spread / ((1 - t) noise spread);  gain(t) = RMS change of the
   endpoint per unit RMS change of the state, by a finite difference with a fixed field (0: insensitive, 1: passed
   through);  rmse(t) of the member mean against r1.
C. The sampler's own path, no target used (float32): a first Euler step of size h from t = 0, r_h = r0 + h v(r0, 0, c),
   then the jump to the endpoint, r_h + (1 - h) v(r_h, h, c):  retained and the RMS change from the one-step endpoint.

    python noise_sensitivity.py --hours 200 \
        --model corrflow_s0=$CORRFLOW_OUT/corrflow_seed0/checkpoints_flow_best/FlowUNet.0.20620032.pt \
        --model corrflow_s1=run:$CORRFLOW_OUT/corrflow_seed1
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "training")))
sys.path.insert(0, HERE)
for k, v in {"MODULUS_DISTRIBUTED_INITIALIZATION_METHOD": "ENV", "RANK": "0", "WORLD_SIZE": "1",
             "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29512"}.items():
    os.environ.setdefault(k, v)
from train_corrflow import CORRDIFF_ROOT, FlowUNet                     # noqa: E402
from fixed_validation import build_fixed_val_set                        # noqa: E402
from predict import best_checkpoint                                     # noqa: E402
from modulus import Module                                              # noqa: E402
from modulus.distributed import DistributedManager                      # noqa: E402
from datasets.dataset import init_train_valid_datasets_from_config      # noqa: E402

from paths import EVAL                                                   # noqa: E402

OUT = os.path.join(EVAL, "noise")
EXCLUDE = (np.datetime64("2024-07-01T00:00:00"), np.datetime64("2024-11-18T00:00:00"))
K = 16
T_B = (0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
H_C = (0.01, 0.05, 0.1, 0.2, 0.5)
EPS = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=<checkpoint> or name=run:<folder>")
    ap.add_argument("--hours", type=int, default=200)
    ap.add_argument("--device", default="cuda", help="cpu only for a smoke test")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    DistributedManager.initialize()
    device = torch.device(args.device)
    torch.backends.cudnn.allow_tf32 = False                    # float32 means float32 (finite differences)
    torch.backends.cuda.matmul.allow_tf32 = False

    from hydra import initialize_config_dir, compose
    with initialize_config_dir(version_base="1.2", config_dir=os.path.join(CORRDIFF_ROOT, "conf")):
        cfg = compose(config_name="vietnam_config_training_corrflow_revision")
    OmegaConf.resolve(cfg)
    dataset, _, val_ds, _ = init_train_valid_datasets_from_config(
        OmegaConf.to_container(cfg.dataset), {"pin_memory": False, "num_workers": 2},
        batch_size=16, seed=0, validation_dataset_cfg=OmegaConf.to_container(cfg.validation), train_test_split=True)
    land = dataset.land_mask[0].numpy() > 0.5
    shape = dataset.image_shape()

    # Random clean 2024 hours, fixed seed; residual and conditioning as for the fixed validation loss
    times = np.array(val_ds.time(), dtype="datetime64[s]")
    clean = np.flatnonzero(~((times >= EXCLUDE[0]) & (times < EXCLUDE[1])))
    idx = np.sort(np.random.default_rng(2026).choice(clean, size=min(args.hours, len(clean)), replace=False))
    reg = Module.from_checkpoint(cfg.training.io.regression_checkpoint_path).eval().requires_grad_(False).to(device)
    vs = build_fixed_val_set(torch.utils.data.Subset(val_ds, idx.tolist()), reg, device,
                             cfg.model.hr_mean_conditioning, torch.bfloat16,
                             cfg.training.perf.fp_optimizations.startswith("amp"), batch_size=16, num_workers=2)
    days = times[idx].astype("datetime64[D]")
    print(f"{len(idx)} clean 2024 hours on {len(np.unique(days))} days, {land.sum()} land pixels", flush=True)

    backbone = OmegaConf.to_container(cfg.model.model_args)
    backbone.setdefault("checkpoint_level", 0)
    land_t = torch.from_numpy(land).to(device)
    g = torch.Generator(device=device)
    noise = torch.empty(K, 1, *shape, device=device)
    for k in range(K):
        g.manual_seed(k)
        noise[k] = torch.randn(1, *shape, generator=g, device=device)
    g.manual_seed(999)
    delta = torch.randn(1, 1, *shape, generator=g, device=device)
    rms = lambda x: x[..., land_t].pow(2).mean().sqrt().item()                 # over members and land pixels
    mvar = lambda x: x[:, 0][:, land_t].var(0, unbiased=True).mean().item()     # mean member variance on land

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for spec in args.model:
        name, src = spec.split("=", 1)
        ckpt = best_checkpoint(src[4:])[0] if src.startswith("run:") else src
        net = FlowUNet(shape, 1 + 1 + vs["cond"].shape[1], 1, **backbone).to(device)
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
        net.eval()
        print(f"{name}: {ckpt}", flush=True)
        rec = {q: [] for q in ("beta", "retained", "linshare", "rmse", "ssr",
                               "beta_bf16", "retained_bf16", "rmse_bf16", "ssr_bf16",
                               "retained_t", "gain_t", "rmse_t", "retained_h", "change_h")}
        t0 = time.time()
        with torch.no_grad():
            for i in range(len(idx)):
                r1 = vs["residual"][i:i + 1].to(device).expand(K, -1, -1, -1)
                cond = vs["cond"][i:i + 1].to(device).expand(K, -1, -1, -1)
                v_at = lambda x, t: net(x, torch.full((K,), t, device=device), cond).float()
                var0 = mvar(noise)
                # A. t = 0
                for tag, amp in (("", False), ("_bf16", True)):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        v = v_at(noise, 0.0)
                    e = noise + v
                    dv, dr = v - v.mean(0), noise - noise.mean(0)
                    beta = ((dv * dr)[..., land_t].sum() / dr[..., land_t].pow(2).sum()).item()
                    vare = mvar(e)
                    rmse = rms(e.mean(0, keepdim=True) - r1[:1])
                    rec["beta" + tag].append(beta)
                    rec["retained" + tag].append(np.sqrt(vare / var0))
                    rec["rmse" + tag].append(rmse)
                    rec["ssr" + tag].append(np.sqrt((K + 1) / K * vare) / rmse)
                    if not amp:
                        rec["linshare"].append((1 + beta) ** 2 * var0 / vare)
                        v0, e0 = v, e
                # B. starting times near 0, state built as in training
                ret, gain, err = [], [], []
                for t in T_B:
                    rt = (1 - t) * noise + t * r1
                    e = rt + (1 - t) * v_at(rt, t)
                    e_p = (rt + EPS * delta) + (1 - t) * v_at(rt + EPS * delta, t)
                    ret.append(np.sqrt(mvar(e) / var0) / (1 - t))
                    gain.append(rms(e_p - e) / (EPS * rms(delta)))
                    err.append(rms(e.mean(0, keepdim=True) - r1[:1]))
                rec["retained_t"].append(ret); rec["gain_t"].append(gain); rec["rmse_t"].append(err)
                # C. first Euler step of size h, then the jump to the endpoint
                ret, chg = [], []
                for h in H_C:
                    rh = noise + h * v0
                    e = rh + (1 - h) * v_at(rh, h)
                    ret.append(np.sqrt(mvar(e) / var0)); chg.append(rms(e - e0))
                rec["retained_h"].append(ret); rec["change_h"].append(chg)
                if i == 0 or (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(idx)} hours  beta {rec['beta'][-1]:+.4f}  retained {rec['retained'][-1]:.4f}  "
                          f"{(time.time() - t0) / (i + 1):.2f} s/hour", flush=True)
        arr = {q: np.array(v) for q, v in rec.items()}
        np.savez_compressed(os.path.join(args.out, f"noise_{name}.npz"), checkpoint=ckpt, hours=idx,
                            time=times[idx].astype(np.int64), t_grid=np.array(T_B), h_grid=np.array(H_C), eps=EPS,
                            members=K, **arr)
        # Means over hours with 95% day-block bootstrap intervals (2,000 resamples, same days for every quantity)
        ud, inv = np.unique(days, return_inverse=True)
        draws = np.random.default_rng(7).integers(0, len(ud), size=(2000, len(ud)))
        cnt = np.stack([np.bincount(d, minlength=len(ud)) for d in draws])[:, inv]      # weight of each hour per draw
        for q, a in arr.items():
            cols = [("", a)] if a.ndim == 1 else [(f"@{x:g}", a[:, j]) for j, x in
                                                   enumerate(T_B if q.endswith("_t") else H_C)]
            for suffix, col in cols:
                boot = (cnt * col).sum(1) / cnt.sum(1)
                summary.append({"model": name, "quantity": q + suffix, "mean": f"{col.mean():.6g}",
                                "ci_low": f"{np.quantile(boot, 0.025):.6g}", "ci_high": f"{np.quantile(boot, 0.975):.6g}",
                                "p05": f"{np.quantile(col, 0.05):.6g}", "median": f"{np.median(col):.6g}",
                                "p95": f"{np.quantile(col, 0.95):.6g}", "hours": len(col), "days": len(ud)})
        print(f"{name} done in {(time.time() - t0) / 60:.1f} min: beta {arr['beta'].mean():+.4f}, retained "
              f"{arr['retained'].mean():.4f} (bf16 {arr['retained_bf16'].mean():.4f}), ssr {arr['ssr'].mean():.3f}", flush=True)
        with open(os.path.join(args.out, "noise_summary.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(summary[0]))
            wr.writeheader(); wr.writerows(summary)


if __name__ == "__main__":
    main()
