"""Score CorrFlow / deterministic-corrector checkpoints on the fixed 2024 validation set.

Same set and same loss as training with training.io.fixed_validation, so runs whose original
validation was random (e.g. CorrFlow seed 0) can be compared and chosen by one rule. Writes one
CSV row per checkpoint; never touches the test year.

    python rescore_checkpoints.py --out $CORRFLOW_OUT/rescore_corrflow_seed0.csv \
        $CORRFLOW_OUT/corrflow_seed0/checkpoints_flow_best/FlowUNet.0.*.pt
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import torch
from omegaconf import OmegaConf

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training")
sys.path.insert(0, os.path.normpath(TRAINING_DIR))
from train_corrflow import CORRDIFF_ROOT, FlowUNet, DetUNet, N_GRID_CH  # noqa: E402
from fixed_validation import build_fixed_val_set, fixed_val_loss       # noqa: E402

from modulus import Module                                             # noqa: E402
from modulus.distributed import DistributedManager                     # noqa: E402
from datasets.dataset import init_train_valid_datasets_from_config      # noqa: E402

# The dataset code asks DistributedManager for rank and world size: set up one local process,
# as train_corrflow.py does.
for k, v in {"MODULUS_DISTRIBUTED_INITIALIZATION_METHOD": "ENV", "RANK": "0", "WORLD_SIZE": "1",
             "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511"}.items():
    os.environ.setdefault(k, v)
DistributedManager.initialize()

ap = argparse.ArgumentParser()
ap.add_argument("checkpoints", nargs="+", help="checkpoint files or glob patterns")
ap.add_argument("--out", required=True, help="CSV to write")
ap.add_argument("--corrector", choices=["flow", "deterministic"], default="flow")
ap.add_argument("--config-name", default="vietnam_config_training_corrflow_revision")
ap.add_argument("--hour-stride", type=int, default=None, help="default: from the config")
ap.add_argument("--exclude-period", nargs=2, metavar=("START", "END"), default=None,
                help="leave out validation hours with START <= valid time < END (ISO), e.g. the\n"
                     "1 July - 17 November 2024 period in which ERA5 and ERA5-Land come from different ERA5 runs")
args = ap.parse_args()

from hydra import initialize_config_dir, compose                        # noqa: E402
with initialize_config_dir(version_base="1.2", config_dir=os.path.join(CORRDIFF_ROOT, "conf")):
    cfg = compose(config_name=args.config_name)
OmegaConf.resolve(cfg)
fv = cfg.training.io.fixed_validation
hour_stride = args.hour_stride or int(fv.hour_stride)

device = torch.device("cuda")
amp_dtype, enable_amp = torch.bfloat16, cfg.training.perf.fp_optimizations.startswith("amp")
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

files = sorted({f for pat in args.checkpoints for f in glob.glob(pat)},
               key=lambda f: int(re.search(r"\.(\d+)\.pt$", f).group(1)))
if not files:
    sys.exit("No checkpoint files matched.")

dataset, _, validation_dataset, _ = init_train_valid_datasets_from_config(
    OmegaConf.to_container(cfg.dataset), {"pin_memory": True, "num_workers": 4},
    batch_size=int(fv.batch_size), seed=0,
    validation_dataset_cfg=OmegaConf.to_container(cfg.validation), train_test_split=True,
)
land_mask = dataset.land_mask
C_x = len(dataset.output_channels())
C_cond = len(dataset.input_channels()) + (C_x if cfg.model.hr_mean_conditioning else 0)

regression_net = Module.from_checkpoint(cfg.training.io.regression_checkpoint_path)
regression_net.eval().requires_grad_(False).to(device)

t0 = time.time()
val_set = build_fixed_val_set(validation_dataset, regression_net, device,
                              cfg.model.hr_mean_conditioning, amp_dtype, enable_amp,
                              hour_stride=hour_stride, batch_size=int(fv.batch_size))
print(f"Fixed validation set: {len(val_set['hours']):,} hours, built in {time.time() - t0:.0f} s",
      flush=True)
if args.exclude_period:
    import numpy as np
    start, end = (np.datetime64(x) for x in args.exclude_period)
    times = np.array(validation_dataset.time(), dtype="datetime64[s]")[val_set["hours"]]
    keep = ~((times >= start) & (times < end))
    val_set = {"residual": val_set["residual"][torch.from_numpy(keep)],
               "cond": val_set["cond"][torch.from_numpy(keep)],
               "hours": [h for h, k in zip(val_set["hours"], keep) if k]}
    print(f"Excluded {int((~keep).sum()):,} hours in [{start}, {end}); {len(val_set['hours']):,} hours kept", flush=True)

backbone_kwargs = OmegaConf.to_container(cfg.model.model_args)
backbone_kwargs.setdefault("checkpoint_level", 0)
if args.corrector == "flow":
    model = FlowUNet(dataset.image_shape(), C_x + 1 + C_cond, C_x, **backbone_kwargs)
else:
    model = DetUNet(dataset.image_shape(), C_cond, C_x, **backbone_kwargs)
model = model.to(device)
n_params = sum(p.numel() for p in model.parameters())

rows = []
for f in files:
    state = torch.load(f, map_location=device)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)          # strict: a wrong architecture fails here
    t0 = time.time()
    loss = fixed_val_loss(model, val_set, args.corrector, land_mask, device, amp_dtype, enable_amp,
                          t_points=int(fv.t_points), noise_seed=int(fv.noise_seed),
                          batch_size=int(fv.batch_size))
    step = int(re.search(r"\.(\d+)\.pt$", f).group(1))
    rows.append({"checkpoint": f, "step": step, "params_M": round(n_params / 1e6, 2),
                 "fixed_val_loss": f"{loss:.6f}", "hours": len(val_set["hours"]),
                 "t_points": int(fv.t_points) if args.corrector == "flow" else 0,
                 "noise_seed": int(fv.noise_seed),
                 "excluded": " to ".join(args.exclude_period) if args.exclude_period else ""})
    print(f"{step:>12,}  fixed_val_loss {loss:.6f}  ({time.time() - t0:.0f} s)  {f}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
best = min(rows, key=lambda r: float(r["fixed_val_loss"]))
print(f"Lowest fixed validation loss: step {best['step']:,} ({best['fixed_val_loss']}); written to {args.out}")
