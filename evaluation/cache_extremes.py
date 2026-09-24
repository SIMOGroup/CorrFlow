#!/usr/bin/env python3
"""Cache model outputs on the heavy-rain subset and pick case-study timestamps.

Writes OUT_DIR/cache/{split}/*.npz and OUT_DIR/extreme_meta.json.
"""
import os, sys, time, warnings, json
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

ZARR_PATH  = "/mnt/data/khaiht/data/vietnam_train_tp_only/vietnam_data.zarr"
STATS_PATH = "/mnt/data/khaiht/data/vietnam_train_tp_only/stats.json"
REG_CKPT   = "/mnt/data/khaiht/outputs/vietnam_regression_hope/checkpoints_regression_best/UNet.0.315136.mdlus"
DIFF_CKPT  = "/mnt/data/khaiht/outputs/vietnam_diffusion_hope/checkpoints_diffusion_best/EDMPrecondSR.0.17310208.mdlus"
FLOW_CKPT  = "/mnt/data/khaiht/outputs/vietnam_flow_hope/checkpoints_flow_best/FlowUNet.0.12225024.pt"
OUT_DIR    = "/mnt/data/khaiht/outputs/vietnam_results_1step"
os.makedirs(OUT_DIR, exist_ok=True)

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
C_COND    = len(dataset.input_channels())
assert IMG_SHAPE == (H, W), f"Grid mismatch: zarr ({H},{W}) vs dataset {IMG_SHAPE}"

print("Loading UNet ...")
net_reg = (Module.from_checkpoint(REG_CKPT)
           .eval().to(device).to(memory_format=torch.channels_last))
print("Loading CorrDiff ...")
net_diff = (Module.from_checkpoint(DIFF_CKPT)
            .eval().to(device).to(memory_format=torch.channels_last))

N_GRID_CH = 4

class FlowUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = SongUNetPosEmbd(
            img_resolution=IMG_SHAPE,
            in_channels=C_OUT + 1 + C_COND + N_GRID_CH,
            out_channels=C_OUT,
            N_grid_channels=N_GRID_CH,
            embedding_type="zero",
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
net_flow.load_state_dict(_ckpt)
net_flow = net_flow.eval()
del _ckpt
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
def run_corrdiff(img_lr_t):
    seeds = list(range(K_ENS))
    sfn   = partial(deterministic_sampler, num_steps=50, solver="heun")
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
def run_corrflow(img_lr_t, n_steps=1, solver="euler",
                 return_intermediates=False, capture_at=None):
    mu   = run_unet(img_lr_t)
    cond = img_lr_t.expand(K_ENS, -1, -1, -1)
    r    = torch.randn(K_ENS, C_OUT, *IMG_SHAPE, device=device)
    dt   = 1.0 / n_steps
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
import time as _time

print("\n" + "=" * 65)
print("  SANITY CHECKS")
print("=" * 65)

# 1. Config summary
print(f"\n[1] Config")
print(f"    K_ENS          = {K_ENS}")
print(f"    IMG_SHAPE      = {IMG_SHAPE}")
print(f"    C_OUT / C_COND = {C_OUT} / {C_COND}")
print(f"    M_IN={M_IN:.4f}  S_IN={S_IN:.4f}")
print(f"    M_OUT={M_OUT:.4f}  S_OUT={S_OUT:.4f}")
print(f"    Q90={Q90_MM:.4f}  Q99={Q99_MM:.4f} mm/hr")
print(f"    OUT_DIR = {OUT_DIR}")

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
        raise FileNotFoundError(f"Checkpoint not found: {path}")

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

# 4. Forward pass on one test timestep
print(f"\n[4] One-timestep forward pass")
_yrs_check = ds_zarr["time"].dt.year.values
_zi_sanity = int(np.where(np.isin(_yrs_check, TEST_YEARS))[0][0])
_ts_sanity = pd.to_datetime(ds_zarr["time"].values[_zi_sanity]).strftime("%Y-%m-%dT%H:%M:%S")
print(f"    Timestep   : {_ts_sanity}  (zarr index {_zi_sanity})")

_tp_c_s  = ds_zarr["tp_coarse"].isel(time=_zi_sanity).values
_tp_t_s  = ds_zarr["tp"].isel(time=_zi_sanity).values
_img_s   = make_input_tensor(_tp_c_s)
_truth_s = truth_to_mm(_tp_t_s)

print(f"    Input tensor : shape={tuple(_img_s.shape)}  "
      f"dtype={_img_s.dtype}  device={_img_s.device}")
print(f"    Input range  : [{float(_img_s.min()):.3f}, {float(_img_s.max()):.3f}]  (z-score expected)")
assert _img_s.shape == (1, C_COND, H, W), f"Input shape mismatch: {_img_s.shape}"
assert not torch.isnan(_img_s).any(), "NaN in model input tensor"

# UNet
_t0 = _time.time()
_mu_s = run_unet(_img_s)
_unet_mm_s = pred_to_mm(_mu_s.cpu().numpy()[:, 0, :, :])[0]
_unet_t    = _time.time() - _t0
print(f"\n    UNet  ({_unet_t:.2f}s)")
print(f"      Output z-score range : [{float(_mu_s.min()):.3f}, {float(_mu_s.max()):.3f}]")
print(f"      Output mm/hr  range  : [{np.nanmin(_unet_mm_s):.3f}, {np.nanmax(_unet_mm_s):.3f}]")
assert not np.isnan(_unet_mm_s[land_mask]).any(), "NaN in UNet output on land"
_unet_mae_s = float(np.nanmean(np.abs(_unet_mm_s - _truth_s)))
print(f"      MAE vs truth         : {_unet_mae_s:.4f} mm/hr")
assert 0.0 < _unet_mae_s < 50.0, f"UNet MAE out of expected range: {_unet_mae_s}"

# CorrDiff
_t0 = _time.time()
_cd_z_s, _ = run_corrdiff(_img_s)
_cd_mm_s   = pred_to_mm(_cd_z_s.cpu().numpy()[:, 0, :, :])
_cd_t      = _time.time() - _t0
print(f"\n    CorrDiff — solver=heun  steps=50  K={K_ENS}  ({_cd_t:.2f}s)")
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
_t0 = _time.time()
_cf_z_s, _ = run_corrflow(_img_s)
_cf_mm_s   = pred_to_mm(_cf_z_s.cpu().numpy()[:, 0, :, :])
_cf_t      = _time.time() - _t0
print(f"\n    CorrFlow — solver=euler  steps=1  K={K_ENS}  ({_cf_t:.2f}s)")
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
print(f"\n[6] Timing estimate")
_per_ts = _unet_t + _cd_t + _cf_t
print(f"    Time per timestep : {_per_ts:.2f} s  "
      f"(UNet {_unet_t:.2f} + CorrDiff/Heun50 {_cd_t:.2f} + CorrFlow/Euler1 {_cf_t:.2f})")
_n_train = int(np.isin(_yrs_check, TRAIN_YEARS).sum())
_n_test  = int(np.isin(_yrs_check, TEST_YEARS).sum())
print(f"    Full-period train : ~{_n_train * _per_ts / 3600:.1f} h  ({_n_train:,} ts)")
print(f"    Full-period test  : ~{_n_test  * _per_ts / 3600:.1f} h  ({_n_test:,} ts)")

# Clean up sanity temporaries
del _img_s, _mu_s, _cd_z_s, _cf_z_s, _tp_c_s, _tp_t_s
del _unet_mm_s, _cd_mm_s, _cf_mm_s, _truth_s, _cd_mean_s, _cf_mean_s

print("\n" + "=" * 65)
print("  ALL SANITY CHECKS PASSED — proceeding to main loop")
print("=" * 65 + "\n")

# Main loop: cache the heavy-rain subset and pick case studies

# Select extreme timesteps
_yrs_zarr   = ds_zarr["time"].dt.year.values
_times_zarr = ds_zarr["time"].values
extreme_idxs = {}

print(f"\nSelecting extremes (threshold = {Q99_MM:.4f} mm/hr) ...")
for split_name, split_years in [
    ("train", TRAIN_YEARS),
    ("val",   VAL_YEARS),
    ("test",  TEST_YEARS),
]:
    idxs_s = np.where(np.isin(_yrs_zarr, split_years))[0]
    tp_log = ds_zarr["tp"].isel(time=idxs_s).values
    spat_q = np.nanquantile(np.expm1(tp_log), 0.99, axis=(1, 2))
    sel    = np.where(spat_q > Q99_MM)[0]
    extreme_idxs[split_name] = idxs_s[sel]
    print(f"  {split_name}: {len(sel):,} extreme / {len(idxs_s):,} total")

# Case 4: strongest NE-monsoon hour
print("\nFinding Case 4 (NE monsoon, Oct-Nov, 14-16N, 107-109E) ...")
_lat_sl = slice(np.searchsorted(lat1d_fine, 14.0),
                np.searchsorted(lat1d_fine, 16.0) + 1)
_lon_sl = slice(np.searchsorted(lon1d_fine, 107.0),
                np.searchsorted(lon1d_fine, 109.0) + 1)
best_val4, best_zi4, best_lat4, best_lon4 = -1.0, None, None, None
for yr4 in [2020, 2022]:
    idxs_yr  = np.where(_yrs_zarr == yr4)[0]
    times_yr = pd.to_datetime(_times_zarr[idxs_yr])
    omon     = (times_yr.month == 10) | (times_yr.month == 11)
    idxs_ym  = idxs_yr[omon]
    if len(idxs_ym) == 0:
        continue
    tp_reg  = np.expm1(
        ds_zarr["tp"].isel(time=idxs_ym).values[:, _lat_sl, _lon_sl])
    peak_ts = np.nanmax(tp_reg, axis=(1, 2))
    bi      = int(np.argmax(peak_ts))
    if peak_ts[bi] > best_val4:
        best_val4 = peak_ts[bi]
        best_zi4  = idxs_ym[bi]
        flat_i    = int(np.nanargmax(tp_reg[bi]))
        n_cols_r  = tp_reg.shape[2]
        best_lat4 = float(lat1d_fine[_lat_sl.start + flat_i // n_cols_r])
        best_lon4 = float(lon1d_fine[_lon_sl.start + flat_i  % n_cols_r])

CASE4_TS = (pd.to_datetime(_times_zarr[best_zi4]).strftime("%Y-%m-%dT%H:%M:%S")
            if best_zi4 is not None else None)
print(f"  Case 4: {CASE4_TS}  "
      f"peak=({best_lat4:.1f}N, {best_lon4:.1f}E)  max={best_val4:.1f} mm/hr")

CASE_META = {
    "case1": ("2024-05-30T15:00:00", "val"),
    "case2": ("2023-05-07T20:00:00", "train"),
    "case3": ("2020-05-23T11:00:00", "train"),
    "case4": (CASE4_TS,              "train"),
}

# Run and cache
_times_pd = pd.to_datetime(_times_zarr)

for run_split, run_idxs in [
    ("test",  extreme_idxs["test"]),
    ("train", extreme_idxs["train"]),
]:
    n  = len(run_idxs)
    t0 = time.time()
    print(f"\n{run_split} extreme: {n} timesteps")
    for ii, zi in enumerate(run_idxs):
        if _is_cached(run_split, zi):
            if (ii + 1) % 100 == 0:
                print(f"  {ii+1}/{n} (cached)")
            continue
        _run_and_cache(run_split, zi)
        if (ii + 1) % 10 == 0:
            rate = (time.time() - t0) / (ii + 1)
            eta  = rate * (n - ii - 1) / 60
            print(f"  {ii+1}/{n}  {rate:.1f} s/ts  eta {eta:.0f} min",
                  flush=True)
    print(f"  Done in {(time.time()-t0)/60:.1f} min.")

for cname, (cts, split_cs) in CASE_META.items():
    if cts is None:
        print(f"{cname}: no timestep found, skipping.")
        continue
    zi_c = int(np.argmin(np.abs(_times_pd - pd.Timestamp(cts))))
    if not _is_cached(split_cs, zi_c):
        print(f"Running {cname} ({cts}) ...")
        _run_and_cache(split_cs, zi_c)
    else:
        print(f"{cname}: cached ({cts})")

# Save metadata for the notebook
meta = {
    "Q90_MM":       Q90_MM,
    "Q99_MM":       Q99_MM,
    "extreme_idxs": {k: v.tolist() for k, v in extreme_idxs.items()},
    "case4_ts":     CASE4_TS,
    "case4_peak":   [best_lat4, best_lon4],
}
meta_path = os.path.join(OUT_DIR, "extreme_meta.json")
with open(meta_path, "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"\nSaved: {meta_path}")
print("\n✅ All extreme inference done.")
