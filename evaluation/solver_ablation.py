#!/usr/bin/env python3
"""
Table 6 solver ablation on the full extreme test set, using the
matched-architecture CorrFlow.

Latency is GPU inference time only: torch.cuda.synchronize() brackets the
model forward, so host transfer and the MAE/CRPS computation are excluded.
A short warmup absorbs cuDNN autotuning.

Options:
  --latency-only     skip MAE/CRPS; measure inference latency only
                     (writes tab6_ablation_latency.csv instead of the full CSV)
  --max-ts N         use at most N extreme test timesteps (default: full set)
  --diff-ckpt PATH   override the CorrDiff checkpoint
  --flow-ckpt PATH   override the CorrFlow checkpoint
  --out-dir  PATH    override the output directory
"""
import os, sys, time, warnings, json, argparse
from pathlib import Path
from functools import partial

import numpy as np
import xarray as xr
import pandas as pd
import torch

# Modulus v0.9.0 checkout with modulus_patch/ applied
MODULUS_ROOT  = "/home/khaiht/oggy_climate/physicsnemo"
CORRDIFF_ROOT = os.path.join(MODULUS_ROOT, "examples/generative/corrdiff")
if CORRDIFF_ROOT not in sys.path:
    sys.path.insert(0, CORRDIFF_ROOT)

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, open_dict
from modulus.distributed import DistributedManager
from modulus import Module
from modulus.models.diffusion import SongUNetPosEmbd
from modulus.utils.corrdiff import regression_step, diffusion_step
from modulus.utils.generative import deterministic_sampler
from helpers.generate_helpers import get_dataset_and_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--latency-only", action="store_true",
                help="skip MAE/CRPS; only measure inference latency")
ap.add_argument("--max-ts", type=int, default=None,
                help="use at most N extreme test timesteps (default: full set)")
ap.add_argument("--diff-ckpt", default=None, help="override CorrDiff checkpoint")
ap.add_argument("--flow-ckpt", default=None, help="override CorrFlow checkpoint")
ap.add_argument("--out-dir",   default=None, help="override output directory")
args = ap.parse_args()

ZARR_PATH  = "/mnt/data/khaiht/data/vietnam_train_tp_only/vietnam_data.zarr"
STATS_PATH = "/mnt/data/khaiht/data/vietnam_train_tp_only/stats.json"
REG_CKPT   = "/mnt/data/khaiht/outputs/vietnam_regression_hope/checkpoints_regression_best/UNet.0.315136.mdlus"
DIFF_CKPT  = args.diff_ckpt or \
    "/mnt/data/khaiht/outputs/vietnam_diffusion_hope/checkpoints_diffusion_best/EDMPrecondSR.0.17310208.mdlus"
FLOW_CKPT  = args.flow_ckpt or \
    "/mnt/data/khaiht/outputs/vietnam_flow_legend/checkpoints_flow_best/FlowUNet.0.12225024.pt"

# OUT_DIR: where the CSV is written. META_DIR: source of extreme_meta.json.
OUT_DIR    = args.out_dir or "/mnt/data/khaiht/outputs/vietnam_results_1step"
META_DIR   = "/mnt/data/khaiht/outputs/vietnam_results_1step"
os.makedirs(OUT_DIR, exist_ok=True)

# Backbone args: must equal cfg.model.model_args used for both models.
BACKBONE_KWARGS = dict(model_channels=64, channel_mult=[1, 2, 2],
                       attn_resolutions=[16])
HR_MEAN_CONDITIONING = True     # matches model/vietnam_corrflow_model.yaml

TRAIN_YEARS = list(range(2017, 2024))
VAL_YEARS   = [2024]
TEST_YEARS  = [2025]
K_ENS       = 16

with open(STATS_PATH) as fh:
    _st = json.load(fh)
M_IN  = float(_st["input"]["tp_coarse"]["mean"])
S_IN  = float(_st["input"]["tp_coarse"]["std"])
M_OUT = float(_st["output"]["tp"]["mean"])
S_OUT = float(_st["output"]["tp"]["std"])

os.environ["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"] = "ENV"
os.environ["RANK"]        = "0"
os.environ["WORLD_SIZE"]  = "1"
os.environ["LOCAL_RANK"]  = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
DistributedManager.initialize()
dist   = DistributedManager()
device = dist.device
print(f"Device: {device}")

ds_zarr    = xr.open_dataset(ZARR_PATH, engine="zarr", chunks=None)
lat1d_fine = ds_zarr["latitude"].values
lon1d_fine = ds_zarr["longitude"].values
H, W       = len(lat1d_fine), len(lon1d_fine)
_tp_s      = ds_zarr["tp"].isel(time=0).values
land_mask  = ~np.isnan(_tp_s)
del _tp_s

_yr_z   = ds_zarr["time"].dt.year.values
_tr_idx = np.where(np.isin(_yr_z, TRAIN_YEARS))[0]
_tp_tr  = np.expm1(ds_zarr["tp"].isel(time=_tr_idx).values)
_flat   = _tp_tr[~np.isnan(_tp_tr)]
Q90_MM  = float(np.nanquantile(_flat, 0.90))
Q99_MM  = float(np.nanquantile(_flat, 0.99))
del _tp_tr, _flat

with initialize_config_dir(version_base="1.2",
                           config_dir=os.path.join(CORRDIFF_ROOT, "conf")):
    cfg = compose(config_name="vietnam_config_generate")
OmegaConf.resolve(cfg)
with open_dict(cfg):
    cfg.generation.io.reg_ckpt_filename = REG_CKPT
    cfg.generation.io.res_ckpt_filename = DIFF_CKPT
    cfg.generation.num_ensembles        = K_ENS
    cfg.generation.inference_mode       = "all"

_ds_cfg = OmegaConf.to_container(cfg.dataset)
_ds_cfg["time_range"] = None
dataset, _ = get_dataset_and_sampler(
    dataset_cfg=_ds_cfg, times=[], has_lead_time=False)
IMG_SHAPE = dataset.image_shape()
C_OUT     = len(dataset.output_channels())
C_LR      = len(dataset.input_channels())          # coarse tp only
C_COND    = C_LR + (C_OUT if HR_MEAN_CONDITIONING else 0)  # flow conditioning ch.
assert IMG_SHAPE == (H, W), f"Grid mismatch: zarr ({H},{W}) vs dataset {IMG_SHAPE}"

print("Loading UNet ...")
net_reg = (Module.from_checkpoint(REG_CKPT)
           .eval().to(device).to(memory_format=torch.channels_last))
print("Loading CorrDiff ...")
net_diff = (Module.from_checkpoint(DIFF_CKPT)
            .eval().to(device).to(memory_format=torch.channels_last))

N_GRID_CH = 4


class FlowUNet(torch.nn.Module):
    """Matched-architecture flow net: backbone kwargs forwarded, and the
    conditioning may include the regression mean (hr_mean)."""
    def __init__(self):
        super().__init__()
        self.net = SongUNetPosEmbd(
            img_resolution=IMG_SHAPE,
            in_channels=C_OUT + 1 + C_COND + N_GRID_CH,
            out_channels=C_OUT,
            N_grid_channels=N_GRID_CH,
            embedding_type="zero",
            **BACKBONE_KWARGS,
        )

    def forward(self, x, t, cond):
        B, _, Hh, Ww = x.shape
        t_emb = t.view(B, 1, 1, 1).expand(B, 1, Hh, Ww)
        return self.net(
            torch.cat([x, t_emb, cond], dim=1),
            noise_labels=torch.zeros(B, device=x.device),
            class_labels=None)


print("Loading CorrFlow ...")
net_flow = FlowUNet().to(device)
_ckpt = torch.load(FLOW_CKPT, map_location=device)
if any(k.startswith("module.") for k in _ckpt):
    _ckpt = {k.replace("module.", "", 1): v for k, v in _ckpt.items()}
net_flow.load_state_dict(_ckpt)          # strict: a mismatch here means wrong arch
net_flow = net_flow.eval()
del _ckpt

n_flow = sum(p.numel() for p in net_flow.parameters())
n_diff = sum(p.numel() for p in net_diff.parameters())
print("\n--- backbone parameter check --------------------------------------")
print(f"  BACKBONE_KWARGS      : {BACKBONE_KWARGS}")
print(f"  HR_MEAN_CONDITIONING : {HR_MEAN_CONDITIONING}  "
      f"(flow cond channels C_COND={C_COND})")
print(f"  CorrFlow params      : {n_flow/1e6:.2f} M  ({n_flow:,})")
print(f"  CorrDiff params      : {n_diff/1e6:.2f} M  ({n_diff:,})")
print(f"  ratio (flow/diff)    : {n_flow/n_diff:.3f}x")
print("-------------------------------------------------------------------")
if abs(n_flow / n_diff - 1) > 0.05:
    raise RuntimeError(
        f"Backbones differ by >5% (ratio {n_flow/n_diff:.3f}x). The loaded "
        f"FlowUNet is NOT the matched-arch net — check BACKBONE_KWARGS "
        f"against cfg.model.model_args.")
print("All models loaded.")


def make_input_tensor(tp_coarse_log1p):
    x = np.nan_to_num(tp_coarse_log1p, nan=0.0)
    x = (x - M_IN) / (S_IN + 1e-6)
    return torch.from_numpy(
        x[np.newaxis, np.newaxis].astype(np.float32)).to(device)


def pred_to_mm(pred_z):
    mm = np.clip(np.expm1(pred_z * (S_OUT + 1e-6) + M_OUT), 0.0, None)
    return np.where(land_mask[np.newaxis], mm, np.nan).astype(np.float32)


def truth_to_mm(tp_log1p):
    return np.where(land_mask, np.expm1(tp_log1p), np.nan).astype(np.float32)


@torch.no_grad()
def run_unet(img_lr_t):
    dummy  = torch.zeros(1, C_OUT, *IMG_SHAPE, device=device)
    sigma0 = torch.zeros(1, device=device)
    return net_reg(dummy, img_lr_t, sigma0)


@torch.no_grad()
def run_corrdiff(img_lr_t, n_steps=10, solver="heun"):
    seeds = list(range(K_ENS))
    sfn   = partial(deterministic_sampler, num_steps=n_steps, solver=solver)
    mu    = run_unet(img_lr_t)
    cond  = img_lr_t
    if cfg.generation.hr_mean_conditioning:
        cond = torch.cat(
            [mu.expand(img_lr_t.shape[0], -1, -1, -1), img_lr_t], dim=1)
    res = diffusion_step(
        net=net_diff, sampler_fn=sfn,
        seed_batch_size=K_ENS, img_shape=IMG_SHAPE, img_out_channels=C_OUT,
        rank_batches=[torch.tensor(seeds)],
        img_lr=cond.expand(K_ENS, -1, -1, -1).to(
            memory_format=torch.channels_last),
        rank=0, device=device, hr_mean=None)
    return mu.expand(K_ENS, -1, -1, -1) + res, mu


@torch.no_grad()
def run_corrflow(img_lr_t, n_steps=10, solver="euler", seed=0,
                 return_intermediates=False, capture_at=None):
    mu = run_unet(img_lr_t)
    # cond must match training: cat([mu, img_lr]) when hr_mean_conditioning.
    cond1 = torch.cat([mu, img_lr_t], dim=1) if HR_MEAN_CONDITIONING else img_lr_t
    cond  = cond1.expand(K_ENS, -1, -1, -1)
    # Fixed base noise (member k uses seed k at every timestep) keeps CorrFlow
    # reproducible and matches the CorrDiff seeds, range(K_ENS).
    g = torch.Generator(device=device)
    r = torch.empty(K_ENS, C_OUT, *IMG_SHAPE, device=device)
    for k in range(K_ENS):
        g.manual_seed(seed + k)
        r[k] = torch.randn(C_OUT, *IMG_SHAPE, generator=g, device=device)
    dt    = 1.0 / n_steps
    en_amp = device.type == "cuda"
    inter = {}
    if return_intermediates:
        if capture_at is None:
            capture_at = list(range(n_steps + 1))
        if 0 in capture_at:
            inter[0] = r.cpu().clone()
    for i in range(n_steps):
        t_cur = torch.full((K_ENS,), i / n_steps, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=en_amp):
            v0 = net_flow(r, t_cur, cond)
        r = r + v0.float() * dt
        if return_intermediates and (i + 1) in capture_at:
            inter[i + 1] = r.cpu().clone()
    pred = mu.expand(K_ENS, -1, -1, -1) + r
    return (pred, mu, inter) if return_intermediates else (pred, mu)


def _crps_ens(truth, ens):
    k = ens.shape[0]
    if k == 1:
        return float(np.nanmean(np.abs(ens[0] - truth)))
    es  = np.sort(ens, axis=0)
    idx = np.arange(k).reshape(k, 1, 1)
    sp  = np.nansum((2 * idx - k + 1) * es, axis=0)
    acc = np.nanmean(np.abs(ens - truth[np.newaxis]), axis=0)
    return float(np.nanmean(acc - sp / k**2))


def _mae_valid(pred, obs):
    valid = ~np.isnan(pred) & ~np.isnan(obs)
    if valid.sum() == 0:
        return np.nan, 0
    return float(np.mean(np.abs(pred[valid] - obs[valid]))), int(valid.sum())


def _crps_valid(ens, obs):
    valid = ~np.isnan(obs) & ~np.all(np.isnan(ens), axis=0)
    if valid.sum() == 0:
        return np.nan, 0
    k   = ens.shape[0]
    o   = obs[valid]
    e   = ens[:, valid]
    if k == 1:
        return float(np.mean(np.abs(e[0] - o))), int(valid.sum())
    es  = np.sort(e, axis=0)
    idx = np.arange(k).reshape(k, 1)
    sp  = np.sum((2 * idx - k + 1) * es, axis=0)
    acc = np.mean(np.abs(e - o[np.newaxis]), axis=0)
    return float(np.mean(acc - sp / k**2)), int(valid.sum())


def _cache_path(split, zi):
    d = Path(OUT_DIR) / "cache" / split
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{zi}.npz")


def _is_cached(split, zi):
    return os.path.exists(_cache_path(split, zi))


def _run_and_cache(split, zi):
    tp_target = ds_zarr["tp"].isel(time=int(zi)).values
    tp_coarse = ds_zarr["tp_coarse"].isel(time=int(zi)).values
    truth_mm  = truth_to_mm(tp_target)
    img_lr    = make_input_tensor(tp_coarse)
    mu_z      = run_unet(img_lr)
    unet_mm   = pred_to_mm(mu_z.cpu().numpy()[:, 0, :, :])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cd_z, _ = run_corrdiff(img_lr)
        cd_mm   = pred_to_mm(cd_z.cpu().numpy()[:, 0, :, :])
        cf_z, _ = run_corrflow(img_lr)
        cf_mm   = pred_to_mm(cf_z.cpu().numpy()[:, 0, :, :])
    np.savez_compressed(
        _cache_path(split, zi),
        truth_mm=truth_mm, unet_mm=unet_mm, cd_mm=cd_mm, cf_mm=cf_mm)


# Sanity checks before the main loop
print("\n" + "=" * 65)
print("  SANITY CHECKS")
print("=" * 65)

# 1. Config summary
print(f"\n[1] Config")
print(f"    K_ENS                 = {K_ENS}")
print(f"    IMG_SHAPE             = {IMG_SHAPE}")
print(f"    C_OUT / C_LR / C_COND = {C_OUT} / {C_LR} / {C_COND}")
print(f"    HR_MEAN_CONDITIONING  = {HR_MEAN_CONDITIONING}")
print(f"    BACKBONE_KWARGS       = {BACKBONE_KWARGS}")
print(f"    M_IN={M_IN:.4f}  S_IN={S_IN:.4f}")
print(f"    M_OUT={M_OUT:.4f}  S_OUT={S_OUT:.4f}")
print(f"    Q90={Q90_MM:.4f}  Q99={Q99_MM:.4f} mm/hr")
print(f"    OUT_DIR  = {OUT_DIR}")
print(f"    META_DIR = {META_DIR}")

# 2. Checkpoint paths exist
print(f"\n[2] Checkpoint files")
for label, path in [
    ("UNet  ", REG_CKPT),
    ("CorrDiff", DIFF_CKPT),
    ("CorrFlow", FLOW_CKPT),
]:
    ok = os.path.exists(path)
    print(f"    {'✅' if ok else '❌'} {label}: {os.path.basename(path)}")
    if not ok:
        raise FileNotFoundError(
            f"{label.strip()} checkpoint not found:\n  {path}\n"
            f"Hint: *_best/ only holds val-loss improvements. Periodic saves "
            f"live in the non-best checkpoints/ folder — list it and update the path.")

# 3. Zarr store
print(f"\n[3] Zarr store")
print(f"    Variables  : {list(ds_zarr.data_vars)}")
print(f"    Total time : {ds_zarr.sizes['time']:,} timesteps")
print(f"    Grid       : H={H}  W={W}  land={land_mask.sum():,} / {H*W:,}")
assert "tp"        in ds_zarr, "zarr missing 'tp' (ERA5-Land target)"
assert "tp_coarse" in ds_zarr, "zarr missing 'tp_coarse' (ERA5 coarse input)"
_yr_check = ds_zarr["time"].dt.year.values
for split_name, years in [("train", TRAIN_YEARS), ("val", VAL_YEARS), ("test", TEST_YEARS)]:
    n = int(np.isin(_yr_check, years).sum())
    print(f"    {split_name:<6}: {n:,} timesteps  ({years[0]}–{years[-1]})")

# 4. Forward pass on the highest-precipitation test timestep
print(f"\n[4] One-timestep forward pass")
_yrs_check = ds_zarr["time"].dt.year.values
_test_idxs_s = np.where(np.isin(_yrs_check, TEST_YEARS))[0]
_spat_max_s  = np.array([
    float(np.nanmax(np.expm1(ds_zarr["tp"].isel(time=int(i)).values)))
    for i in _test_idxs_s[:500]
])
_zi_sanity  = int(_test_idxs_s[int(np.argmax(_spat_max_s))])
_ts_sanity  = pd.to_datetime(ds_zarr["time"].values[_zi_sanity]).strftime("%Y-%m-%dT%H:%M:%S")
print(f"    Timestep   : {_ts_sanity}  (zarr index {_zi_sanity})")

_tp_c_s  = ds_zarr["tp_coarse"].isel(time=_zi_sanity).values
_tp_t_s  = ds_zarr["tp"].isel(time=_zi_sanity).values
_img_s   = make_input_tensor(_tp_c_s)
_truth_s = truth_to_mm(_tp_t_s)

print(f"    Input tensor : shape={tuple(_img_s.shape)}  "
      f"dtype={_img_s.dtype}  device={_img_s.device}")
print(f"    Input range  : [{float(_img_s.min()):.3f}, {float(_img_s.max()):.3f}]  (z-score expected)")
# make_input_tensor emits the coarse channel(s) only -> compare against C_LR.
assert _img_s.shape == (1, C_LR, H, W), f"Input shape mismatch: {_img_s.shape}"
assert not torch.isnan(_img_s).any(), "NaN in model input tensor"


def _gpu_time(fn):
    """Time one call on the GPU after a single warmup iteration.

    Same method as the ablation loop, so the result is comparable to Table 6.
    Host transfer and denormalization are not timed."""
    fn()                                   # warmup: cuDNN autotune / lazy init
    torch.cuda.synchronize()
    _t  = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - _t


# UNet
_mu_s, _unet_t = _gpu_time(lambda: run_unet(_img_s))
_unet_mm_s = pred_to_mm(_mu_s.cpu().numpy()[:, 0, :, :])[0]
print(f"\n    UNet  ({_unet_t:.3f}s, GPU forward only)")
print(f"      Output z-score range : [{float(_mu_s.min()):.3f}, {float(_mu_s.max()):.3f}]")
print(f"      Output mm/hr  range  : [{np.nanmin(_unet_mm_s):.3f}, {np.nanmax(_unet_mm_s):.3f}]")
assert not np.isnan(_unet_mm_s[land_mask]).any(), "NaN in UNet output on land"
_unet_mae_s = float(np.nanmean(np.abs(_unet_mm_s - _truth_s)))
print(f"      MAE vs truth         : {_unet_mae_s:.4f} mm/hr")
assert 0.0 < _unet_mae_s < 50.0, f"UNet MAE out of expected range: {_unet_mae_s}"

# CorrDiff
_cd_z_s, _cd_t = _gpu_time(lambda: run_corrdiff(_img_s, n_steps=50)[0])
_cd_mm_s = pred_to_mm(_cd_z_s.cpu().numpy()[:, 0, :, :])
print(f"\n    CorrDiff — solver=heun  steps=50  K={K_ENS}  ({_cd_t:.3f}s, GPU forward only)")
print(f"      Ensemble shape       : {_cd_mm_s.shape}  (K, H, W)")
print(f"      Output mm/hr  range  : [{np.nanmin(_cd_mm_s):.3f}, {np.nanmax(_cd_mm_s):.3f}]")
_cd_mean_s = np.nanmean(_cd_mm_s, axis=0)
_cd_mae_s  = float(np.nanmean(np.abs(_cd_mean_s - _truth_s)))
_cd_std_s  = float(np.nanmean(np.nanstd(_cd_mm_s, axis=0)[land_mask]))
print(f"      Ens mean MAE         : {_cd_mae_s:.4f} mm/hr")
print(f"      Mean ens std         : {_cd_std_s:.4f} mm/hr  (>0 = ensemble is diverse)")
assert _cd_mm_s.shape == (K_ENS, H, W), f"CorrDiff shape mismatch: {_cd_mm_s.shape}"
assert not np.isnan(_cd_std_s), "CorrDiff ensemble is all-NaN — checkpoint or denorm may be wrong"

# CorrFlow
_cf_z_s, _cf_t = _gpu_time(lambda: run_corrflow(_img_s, n_steps=1)[0])
_cf_mm_s = pred_to_mm(_cf_z_s.cpu().numpy()[:, 0, :, :])
print(f"\n    CorrFlow — solver=euler  steps=1  K={K_ENS}  ({_cf_t:.3f}s, GPU forward only)")
print(f"      Ensemble shape       : {_cf_mm_s.shape}  (K, H, W)")
print(f"      Output mm/hr  range  : [{np.nanmin(_cf_mm_s):.3f}, {np.nanmax(_cf_mm_s):.3f}]")
_cf_mean_s = np.nanmean(_cf_mm_s, axis=0)
_cf_mae_s  = float(np.nanmean(np.abs(_cf_mean_s - _truth_s)))
_cf_std_s  = float(np.nanmean(np.nanstd(_cf_mm_s, axis=0)[land_mask]))
print(f"      Ens mean MAE         : {_cf_mae_s:.4f} mm/hr")
print(f"      Mean ens std         : {_cf_std_s:.4f} mm/hr  (>0 = ensemble is diverse)")
assert _cf_mm_s.shape == (K_ENS, H, W), f"CorrFlow shape mismatch: {_cf_mm_s.shape}"
assert not np.isnan(_cf_std_s), "CorrFlow ensemble is all-NaN — checkpoint or denorm may be wrong"

# 5. Output and truth ranges
print(f"\n[5] Truth for test timestep {_ts_sanity}")
print(f"    Truth range  : [{np.nanmin(_truth_s):.3f}, {np.nanmax(_truth_s):.3f}] mm/hr")
print(f"    Land pixels  : {int(np.sum(~np.isnan(_truth_s))):,}")
assert np.nanmax(_truth_s) > 0,   "Truth is all zeros — denorm may be wrong"
assert np.nanmax(_truth_s) < 500, "Truth max > 500 mm/hr — denorm likely wrong"
assert np.nanmin(_unet_mm_s) >= -1, "Negative predictions — denorm likely wrong"

# 6. Timing estimate
print(f"\n[6] Timing estimate  (GPU forward only; excludes host transfer + I/O)")
_per_ts = _unet_t + _cd_t + _cf_t
print(f"    Time per timestep : {_per_ts:.3f} s  "
      f"(UNet {_unet_t:.3f} + CorrDiff/Heun50 {_cd_t:.3f} + CorrFlow/Euler1 {_cf_t:.3f})")
_n_train = int(np.isin(_yrs_check, TRAIN_YEARS).sum())
_n_test  = int(np.isin(_yrs_check, TEST_YEARS).sum())
print(f"    Full-period train : ~{_n_train * _per_ts / 3600:.1f} h  ({_n_train:,} ts)  [lower bound]")
print(f"    Full-period test  : ~{_n_test  * _per_ts / 3600:.1f} h  ({_n_test:,} ts)  [lower bound]")

del _img_s, _mu_s, _cd_z_s, _cf_z_s, _tp_c_s, _tp_t_s
del _unet_mm_s, _cd_mm_s, _cf_mm_s, _truth_s, _cd_mean_s, _cf_mean_s

print("\n" + "=" * 65)
print("  ALL SANITY CHECKS PASSED — proceeding to main loop")
print("=" * 65 + "\n")

# Table 6 solver ablation: reads META_DIR/extreme_meta.json, writes a CSV to OUT_DIR

meta_path = os.path.join(META_DIR, "extreme_meta.json")
assert os.path.exists(meta_path), \
    f"extreme_meta.json not found — run cache_extremes.py first.\n{meta_path}"

with open(meta_path) as fh:
    _meta = json.load(fh)
_yrs_zarr    = ds_zarr["time"].dt.year.values
_times_zarr  = ds_zarr["time"].values
extreme_idxs = {k: np.array(v) for k, v in _meta["extreme_idxs"].items()}
test_idxs    = extreme_idxs["test"]        # full extreme test set by default
if args.max_ts:
    rng       = np.random.default_rng(0)
    test_idxs = np.sort(rng.choice(
        test_idxs, size=min(args.max_ts, len(test_idxs)), replace=False))
n_test       = len(test_idxs)
print(f"Using {n_test} extreme test timesteps"
      f"{'  (LATENCY ONLY)' if args.latency_only else ''}\n")

ab_rows = []
WARMUP  = 3   # cuDNN autotune + allocator, so the first timestep isn't timed cold
_z0 = ds_zarr["tp_coarse"].isel(time=int(test_idxs[0])).values

# CorrDiff: Heun with 10, 25, 50, 100 steps
CD_STEPS = [10, 25, 50, 100]
print(f"CorrDiff Heun ablation over {n_test} extreme test timesteps ...")
print(f"{'Model':<12} {'Solver':<6} {'Steps':>5}  {'MAE':>8}  {'CRPS':>8}  {'Lat(s)':>8}")
print("-" * 58)

for n_ab in CD_STEPS:
    maes      = []
    crps_list = []
    gpu_times = []

    # warmup on the first timestep
    for _ in range(WARMUP):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_corrdiff(make_input_tensor(_z0), n_steps=n_ab)
    torch.cuda.synchronize()

    for zi in test_idxs:
        tp_c   = ds_zarr["tp_coarse"].isel(time=int(zi)).values
        img_lr = make_input_tensor(tp_c)
        truth  = (truth_to_mm(ds_zarr["tp"].isel(time=int(zi)).values)
                  if not args.latency_only else None)

        torch.cuda.synchronize()
        _t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cd_z, _ = run_corrdiff(img_lr, n_steps=n_ab)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - _t)

        if not args.latency_only:
            cd_mm = pred_to_mm(cd_z.cpu().numpy()[:, 0, :, :])
            maes.append(
                float(np.nanmean(np.abs(np.nanmean(cd_mm, axis=0) - truth))))
            crps_list.append(_crps_ens(truth, cd_mm))

    latency   = float(np.mean(gpu_times))
    mae_mean  = float(np.mean(maes))      if maes      else np.nan
    crps_mean = float(np.mean(crps_list)) if crps_list else np.nan
    ab_rows.append({
        "model": "CorrDiff", "solver": "heun", "steps": n_ab,
        "MAE": mae_mean, "CRPS": crps_mean, "Latency_s": latency,
    })
    print(f"{'CorrDiff':<12} {'heun':<6} {n_ab:>5}  "
          f"{mae_mean:>8.4f}  {crps_mean:>8.4f}  {latency:>8.2f}", flush=True)

# CorrFlow: Euler with 1, 5, 10, 25 steps
CF_STEPS = [1, 5, 10, 25]
print(f"\nCorrFlow Euler ablation over {n_test} extreme test timesteps ...")

for n_ab in CF_STEPS:
    maes      = []
    crps_list = []
    gpu_times = []

    # warmup on the first timestep
    for _ in range(WARMUP):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_corrflow(make_input_tensor(_z0), n_steps=n_ab, solver="euler")
    torch.cuda.synchronize()

    for zi in test_idxs:
        tp_c   = ds_zarr["tp_coarse"].isel(time=int(zi)).values
        img_lr = make_input_tensor(tp_c)
        truth  = (truth_to_mm(ds_zarr["tp"].isel(time=int(zi)).values)
                  if not args.latency_only else None)

        torch.cuda.synchronize()
        _t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cf_z, _ = run_corrflow(img_lr, n_steps=n_ab, solver="euler")
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - _t)

        if not args.latency_only:
            cf_mm = pred_to_mm(cf_z.cpu().numpy()[:, 0, :, :])
            maes.append(
                float(np.nanmean(np.abs(np.nanmean(cf_mm, axis=0) - truth))))
            crps_list.append(_crps_ens(truth, cf_mm))

    latency   = float(np.mean(gpu_times))
    mae_mean  = float(np.mean(maes))      if maes      else np.nan
    crps_mean = float(np.mean(crps_list)) if crps_list else np.nan
    ab_rows.append({
        "model": "CorrFlow", "solver": "euler", "steps": n_ab,
        "MAE": mae_mean, "CRPS": crps_mean, "Latency_s": latency,
    })
    print(f"{'CorrFlow':<12} {'euler':<6} {n_ab:>5}  "
          f"{mae_mean:>8.4f}  {crps_mean:>8.4f}  {latency:>8.2f}", flush=True)

df_ab = pd.DataFrame(ab_rows)
# Full run keeps the canonical name; latency-only is tagged so it can't
# overwrite a full MAE/CRPS table.
_csv_name = "tab6_ablation_latency.csv" if args.latency_only else "tab6_ablation.csv"
out_path  = os.path.join(OUT_DIR, _csv_name)
df_ab.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("\n=== Table 6 (matched-arch CorrFlow, full extreme test set) ===")
print(df_ab[["model", "solver", "steps", "MAE", "CRPS", "Latency_s"]]
      .round(4).to_string(index=False))

# Headline speedup
try:
    cd50 = df_ab[(df_ab.model == "CorrDiff") & (df_ab.steps == 50)].Latency_s.iloc[0]
    cf1  = df_ab[(df_ab.model == "CorrFlow") & (df_ab.steps == 1)].Latency_s.iloc[0]
    print(f"\nSpeedup (CorrDiff Heun-50 / CorrFlow Euler-1): {cd50 / cf1:.1f}x")
except (IndexError, ZeroDivisionError):
    pass