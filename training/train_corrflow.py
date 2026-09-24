import os
import sys
import argparse
import time
import psutil
import logging
from pathlib import Path

from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

# Modulus v0.9.0 checkout with modulus_patch/ applied
MODULUS_ROOT  = "/home/khaiht/oggy_climate/physicsnemo"
CORRDIFF_ROOT = os.path.join(MODULUS_ROOT, "examples/generative/corrdiff")
if CORRDIFF_ROOT not in sys.path:
    sys.path.insert(0, CORRDIFF_ROOT)

from modulus import Module
from modulus.models.diffusion import SongUNetPosEmbd
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint

from datasets.dataset import init_train_valid_datasets_from_config
from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)


# Early stopping

class EarlyStopper:
    """Stop training when validation loss stops improving.

    Stops after `patience` consecutive validation checks without an
    improvement larger than `min_delta`. The best loss is tracked so the
    caller knows when to save a best-model checkpoint.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.bad_checks = 0
        self.improved   = False

    def update(self, val_loss: float) -> bool:
        """Record a validation loss; return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.bad_checks = 0
            self.improved   = True
        else:
            self.bad_checks += 1
            self.improved   = False

        return self.bad_checks >= self.patience


# FlowUNet definition

# Number of sinusoidal positional-embedding channels appended by
# SongUNetPosEmbd.forward() before passing to the parent SongUNet layers.
# Must stay in sync with N_grid_channels passed to the constructor.
N_GRID_CH = 4


class FlowUNet(torch.nn.Module):
    """Flow-matching corrector that replaces the EDM diffusion stage of CorrDiff.

    Architecture:
        SongUNetPosEmbd with embedding_type="zero" (no diffusion noise-level
        embedding) and sinusoidal positional grid channels.

    Input to the UNet at forward time:
        [x_t | t_scalar_channel | cond | pos_emb]
         C_x    1                  C_cond  N_GRID_CH

    The constructor receives in_channels = C_x + 1 + C_cond (the "logical"
    count before positional embedding). FlowUNet.__init__ adds N_GRID_CH
    when constructing SongUNetPosEmbd so the first Conv2d has the right shape.
    """

    def __init__(self, img_shape, in_channels, out_channels,
                 n_grid_channels=N_GRID_CH, **backbone_kwargs):
        """backbone_kwargs must carry cfg.model.model_args (model_channels,
        channel_mult, attn_resolutions) and checkpoint_level. Without them
        SongUNetPosEmbd falls back to library defaults (model_channels=128,
        channel_mult=[1,2,2,2]), making this network ~7x larger than the
        CorrDiff denoiser and breaking the controlled comparison."""
        super().__init__()
        self.in_channels_logical = in_channels
        self.out_channels        = out_channels
        self.n_grid_channels     = n_grid_channels

        self.net = SongUNetPosEmbd(
            img_resolution=img_shape,
            in_channels=in_channels + n_grid_channels,
            out_channels=out_channels,
            N_grid_channels=n_grid_channels,
            embedding_type="zero",
            **backbone_kwargs,
        )

    def forward(self, x, t, cond):
        """x: (B, C_x, H, W) residual at flow time t; t: (B,) in [0, 1];
        cond: (B, C_cond, H, W) low-resolution conditioning."""
        B, _, H, W = x.shape
        t_embed = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        x_in    = torch.cat([x, t_embed, cond], dim=1)

        return self.net(
            x_in,
            noise_labels=torch.zeros(B, device=x.device),
            class_labels=None,
        )


# Flow-matching loss

def flow_matching_loss(
    model: torch.nn.Module,
    residual: torch.Tensor,
    cond: torch.Tensor,
    land_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Conditional flow-matching loss (OT straight-path formulation).

    Samples t ~ Uniform[0,1], constructs the linear interpolation between
    base noise r0 and target residual, and regresses the constant velocity.
    Ocean pixels are excluded from the loss via land_mask.

    Args:
        model    : FlowUNet (or DDP-wrapped FlowUNet)
        residual : (B, C_out, H, W)  target residual = img_clean - mu_regression
        cond     : (B, C_cond, H, W) low-res conditioning
        land_mask: (1, H, W) float32, 1=land 0=ocean, or None to use all pixels

    Returns:
        Scalar loss (mean over land pixels only, or all pixels if mask is None).
    """
    B      = residual.shape[0]
    device = residual.device

    r0     = torch.randn_like(residual)
    t      = torch.rand(B, device=device)
    t_view = t.view(B, 1, 1, 1)

    rt     = (1 - t_view) * r0 + t_view * residual
    target = residual - r0

    pred    = model(rt, t, cond)
    sq_err  = (pred - target) ** 2   # (B, C, H, W)

    if land_mask is not None:
        mask = land_mask.to(device)              # (1, H, W)
        # Mean over land pixels only: sum(err * mask) / sum(mask * C * B)
        return (sq_err * mask).sum() / (mask.sum() * sq_err.shape[0] * sq_err.shape[1])

    return sq_err.mean()


# Loss-curve helper

def _save_loss_curve(train_history, val_history, out_path, title, logger):
    """Plot per-step training loss and validation loss on a log scale."""
    try:
        import matplotlib
        matplotlib.use("Agg")           # non-interactive backend for headless nodes
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))

        if train_history:
            t_steps, t_losses = zip(*train_history)
            ax.plot(t_steps, t_losses, color="#4C72B0", alpha=0.35,
                    linewidth=0.8, label="train loss (per step)")

            # Running mean so the trend shows through the noise
            window = max(1, len(t_losses) // 100)
            if window > 1:
                import numpy as np
                kernel    = np.ones(window) / window
                t_smooth  = np.convolve(t_losses, kernel, mode="valid")
                t_s_steps = t_steps[window - 1:]
                ax.plot(t_s_steps, t_smooth, color="#4C72B0", linewidth=1.8,
                        label=f"train loss (running mean, w={window})")

        if val_history:
            v_steps, v_losses = zip(*val_history)
            ax.plot(v_steps, v_losses, color="#DD8452", linewidth=2.0,
                    marker="o", markersize=4, label="val loss")

            # Mark the best validation point
            best_idx  = int(min(range(len(v_losses)), key=lambda i: v_losses[i]))
            ax.axvline(v_steps[best_idx], color="#DD8452", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.scatter([v_steps[best_idx]], [v_losses[best_idx]],
                       color="#DD8452", s=80, zorder=5,
                       label=f"best val = {v_losses[best_idx]:.4f} @ {v_steps[best_idx]:,}")

        ax.set_yscale("log")
        ax.set_xlabel("Images seen (samples)", fontsize=11)
        ax.set_ylabel("Loss (log scale)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
        fig.tight_layout()

        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Loss curve saved → {out_path}")

    except Exception as exc:
        logger.warning(f"Could not save loss curve: {exc}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="vietnam_config_training_corrflow")
    args = parser.parse_args()

    # Distributed env

    if "RANK" not in os.environ:
        os.environ["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"] = "ENV"
        os.environ["RANK"]        = "0"
        os.environ["WORLD_SIZE"]  = "1"
        os.environ["LOCAL_RANK"]  = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

    print(
        {
            k: os.environ.get(k)
            for k in [
                "RANK", "WORLD_SIZE", "LOCAL_RANK",
                "MODULUS_DISTRIBUTED_INITIALIZATION_METHOD",
                "SLURM_PROCID",
            ]
        }
    )

    # Load Hydra config

    from hydra import initialize_config_dir, compose

    with initialize_config_dir(
        version_base="1.2",
        config_dir=os.path.join(CORRDIFF_ROOT, "conf"),
    ):
        cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg))

    # Run directory

    job_name      = cfg.run.name
    run_dir       = f"/mnt/data/khaiht/outputs/{job_name}_hope"
    tb_dir        = os.path.join(run_dir, "tensorboard")
    ckpt_dir      = os.path.join(run_dir, "checkpoints_flow")
    best_ckpt_dir = os.path.join(run_dir, "checkpoints_flow_best")

    os.makedirs(tb_dir,        exist_ok=True)
    os.makedirs(ckpt_dir,      exist_ok=True)
    os.makedirs(best_ckpt_dir, exist_ok=True)
    os.chdir(run_dir)

    print("Job name     :", job_name)
    print("Run dir      :", run_dir)
    print("TensorBoard  :", tb_dir)
    print("Checkpoints  :", ckpt_dir)
    print("Best ckpt    :", best_ckpt_dir)

    # Logging

    log_file = Path(run_dir) / "train.log"

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )

    # Distributed init

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.rank == 0:
        writer = SummaryWriter(log_dir=tb_dir)

    logger  = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)

    print("World size:", dist.world_size)
    print("Device    :", dist.device)

    OmegaConf.resolve(cfg)

    # Dataset config

    dataset_cfg = OmegaConf.to_container(cfg.dataset)

    if hasattr(cfg, "validation"):
        train_test_split       = True
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation)
    else:
        train_test_split       = False
        validation_dataset_cfg = None

    # Training performance options

    fp_optimizations          = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level

    fp16       = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype  = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )

    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    # Dataset loading

    print("Checkpoint dir:", ckpt_dir)

    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 8,
    }

    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
        train_test_split=train_test_split,
    )

    print("Dataset ready")
    if validation_dataset is not None:
        print(f"  Train samples : {len(dataset):,}")
        print(f"  Val samples   : {len(validation_dataset):,}")

    # Land mask from the dataset, kept on CPU and passed to the loss
    land_mask = dataset.land_mask   # (1, H, W) float32, 1=land 0=ocean

    # Channel arithmetic

    dataset_channels = len(dataset.input_channels())
    img_in_channels  = dataset_channels
    img_shape        = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    patch_shape = (None, None)
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)

    logger0.info("Patch-based training disabled (CorrFlow is full-res only)")

    print("img_shape       :", img_shape)
    print("img_out_channels:", img_out_channels)
    print(
        "img_in_channels :", img_in_channels,
        " (LR inputs" + (" + regression mean" if cfg.model.hr_mean_conditioning else "") + ")",
    )

    # FlowUNet construction

    C_x    = img_out_channels
    C_t    = 1
    C_cond = img_in_channels

    # Backbone args from the YAML, exactly as train_corrdiff.py applies them to the
    # diffusion denoiser, so both correctors share one architecture.
    backbone_kwargs = {}
    if hasattr(cfg.model, "model_args"):
        backbone_kwargs.update(OmegaConf.to_container(cfg.model.model_args))
    backbone_kwargs.setdefault("checkpoint_level", songunet_checkpoint_level)
    for k in ("img_channels", "img_out_channels", "img_resolution", "use_fp16",
              "gridtype", "N_grid_channels", "scale_cond_input"):
        backbone_kwargs.pop(k, None)          # not SongUNet constructor args

    model = FlowUNet(
        img_shape=img_shape,
        in_channels=C_x + C_t + C_cond,
        out_channels=C_x,
        **backbone_kwargs,
    ).to(dist.device)
    logger0.info(f"FlowUNet backbone kwargs: {backbone_kwargs}")
    logger0.info(f"FlowUNet params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    model.train().requires_grad_(True)

    logger0.info(
        f"FlowUNet  logical_in={C_x + C_t + C_cond}  "
        f"actual_in={C_x + C_t + C_cond + N_GRID_CH}  "
        f"out={C_x}"
    )

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # Frozen regression net

    regression_checkpoint_path = cfg.training.io.regression_checkpoint_path
    regression_net = Module.from_checkpoint(regression_checkpoint_path)
    regression_net.eval().requires_grad_(False).to(dist.device)
    logger0.info(f"Loaded regression net: {regression_checkpoint_path}")

    # Optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    start_time = time.time()

    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu

    # Early stopping setup

    es_cfg = getattr(cfg.training.io, "early_stopping", None)
    if es_cfg is not None and validation_dataset_iterator is not None:
        early_stopper = EarlyStopper(
            patience=int(es_cfg.patience),
            min_delta=float(getattr(es_cfg, "min_delta", 0.0)),
        )
        logger0.info(
            f"Early stopping enabled  patience={early_stopper.patience}  "
            f"min_delta={early_stopper.min_delta}"
        )
    else:
        early_stopper = None
        if es_cfg is not None:
            logger0.warning(
                "early_stopping is configured but no validation dataset found — "
                "early stopping disabled."
            )

    # Resume checkpoint

    try:
        cur_nimg = load_checkpoint(
            path=ckpt_dir,
            models=model,
            optimizer=optimizer,
            device=dist.device,
        )
        logger0.info(f"Resumed from checkpoint  cur_nimg={cur_nimg:,}")
    except Exception:
        cur_nimg = 0
        logger0.info("No checkpoint found – starting fresh.")

    # Training loop

    logger0.info(
        f"Training CorrFlow for {cfg.training.hp.training_duration:,} images  "
        f"(batch={cfg.training.hp.total_batch_size}, "
        f"accum_rounds={num_accumulation_rounds}, "
        f"world_size={dist.world_size})"
    )

    done                        = False
    average_loss_running_mean   = 0.0
    n_average_loss_running_mean = 1

    # History for end-of-training loss curve (rank-0 only)
    train_loss_history = []   # [(cur_nimg, loss), ...]
    val_loss_history   = []   # [(cur_nimg, loss), ...]

    while not done:

        tick_start_nimg = cur_nimg
        tick_start_time = time.time()

        # Forward and backward with gradient accumulation

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(num_accumulation_rounds):
            img_clean, img_lr, labels, *_ = next(dataset_iterator)
            img_clean = img_clean.to(dist.device).to(torch.float32).contiguous()
            img_lr    = img_lr.to(dist.device).to(torch.float32).contiguous()

            # Evaluate the frozen regression stage inside the mixed-precision
            # region, matching how ResLoss evaluates it in the diffusion loop,
            # so training throughput is measured under identical precision.
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                B, _, H, W = img_lr.shape
                dummy = torch.zeros(
                    B, regression_net.img_out_channels, H, W, device=dist.device
                )
                mu = regression_net(
                    dummy,
                    img_lr=img_lr,
                    sigma=torch.zeros(B, device=dist.device),
                )

            mu       = mu.to(img_clean.dtype)
            residual = img_clean - mu
            # Match CorrDiff's conditioning: ResLoss concatenates the regression
            # mean onto the LR input when hr_mean_conditioning=True.
            cond = torch.cat([mu, img_lr], dim=1) if cfg.model.hr_mean_conditioning else img_lr

            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                loss = flow_matching_loss(model, residual, cond, land_mask=land_mask)

            loss = loss / num_accumulation_rounds
            loss.backward()
            loss_accum += loss.item()

        # Reduce loss across GPUs

        loss_sum = torch.tensor([loss_accum], device=dist.device)
        if dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = (loss_sum / dist.world_size).cpu().item()

        average_loss_running_mean += (
            average_loss - average_loss_running_mean
        ) / n_average_loss_running_mean
        n_average_loss_running_mean += 1

        if dist.rank == 0:
            writer.add_scalar("train/loss",              average_loss,              cur_nimg)
            writer.add_scalar("train/loss_running_mean", average_loss_running_mean, cur_nimg)
            train_loss_history.append((cur_nimg, average_loss))

        # LR schedule: linear warmup, then exponential decay

        lr_rampup = cfg.training.hp.lr_rampup
        for g in optimizer.param_groups:
            progress = min(cur_nimg / lr_rampup, 1.0) if lr_rampup > 0 else 1.0
            decay    = cfg.training.hp.lr_decay ** ((max(cur_nimg - lr_rampup, 0)) // 5_000_000)
            g["lr"]  = cfg.training.hp.lr * progress * decay
            current_lr = g["lr"]

        if dist.rank == 0:
            writer.add_scalar("train/lr", current_lr, cur_nimg)

        handle_and_clip_gradients(
            model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
        )
        optimizer.step()

        cur_nimg += cfg.training.hp.total_batch_size
        done = cur_nimg >= cfg.training.hp.training_duration

        # Progress logging + running-mean reset

        if is_time_for_periodic_task(
            cur_nimg, cfg.training.io.print_progress_freq, done,
            cfg.training.hp.total_batch_size, dist.rank, rank_0_only=True,
        ):
            average_loss_running_mean   = 0.0
            n_average_loss_running_mean = 1

            tick_end_time = time.time()
            elapsed  = tick_end_time - start_time
            tick_dur = tick_end_time - tick_start_time
            n_tick   = max(cur_nimg - tick_start_nimg, 1)

            fields = [
                f"samples {cur_nimg:>12,}",
                f"loss {average_loss:<7.4f}",
                f"loss_running_mean {average_loss_running_mean:<7.4f}",
                f"lr {current_lr:.3e}",
                f"total_sec {elapsed:<7.0f}",
                f"sec/tick {tick_dur:<6.1f}",
                f"sec/sample {tick_dur / n_tick:<7.4f}",
                f"cpu_gb {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}",
                f"peak_gpu_gb {torch.cuda.max_memory_allocated(dist.device) / 2**30:<6.2f}",
                f"peak_gpu_res_gb {torch.cuda.max_memory_reserved(dist.device) / 2**30:<6.2f}",
            ]
            logger0.info("  ".join(fields))
            torch.cuda.reset_peak_memory_stats()

        # Validation

        avg_valid_loss = None

        if validation_dataset_iterator is not None and is_time_for_periodic_task(
            cur_nimg, cfg.training.io.validation_freq, done,
            cfg.training.hp.total_batch_size, dist.rank,
        ):
            model.eval()
            valid_loss_accum = 0.0

            with torch.no_grad():
                for _ in range(cfg.training.io.validation_steps):
                    img_clean_v, img_lr_v, *_ = next(validation_dataset_iterator)
                    img_clean_v = img_clean_v.to(dist.device).to(torch.float32).contiguous()
                    img_lr_v    = img_lr_v.to(dist.device).to(torch.float32).contiguous()

                    with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                        B, _, H, W = img_lr_v.shape
                        dummy_v = torch.zeros(
                            B, regression_net.img_out_channels, H, W, device=dist.device
                        )
                        mu_v = regression_net(
                            dummy_v,
                            img_lr=img_lr_v,
                            sigma=torch.zeros(B, device=dist.device),
                        )
                    mu_v       = mu_v.to(img_clean_v.dtype)
                    residual_v = img_clean_v - mu_v
                    cond_v = (torch.cat([mu_v, img_lr_v], dim=1)
                              if cfg.model.hr_mean_conditioning else img_lr_v)

                    with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                        loss_v = flow_matching_loss(model, residual_v, cond_v, land_mask=land_mask)

                    valid_loss_accum += loss_v.item() / cfg.training.io.validation_steps

            model.train()

            valid_sum = torch.tensor([valid_loss_accum], device=dist.device)
            if dist.world_size > 1:
                torch.distributed.barrier()
                torch.distributed.all_reduce(
                    valid_sum, op=torch.distributed.ReduceOp.SUM
                )
            avg_valid_loss = (valid_sum / dist.world_size).cpu().item()

            if dist.rank == 0:
                writer.add_scalar("val/loss", avg_valid_loss, cur_nimg)
                val_loss_history.append((cur_nimg, avg_valid_loss))

            logger0.info(
                f"  [val]  samples={cur_nimg:,}  val_loss={avg_valid_loss:.4f}  "
                f"best={early_stopper.best_loss:.4f}" if early_stopper
                else f"  [val]  samples={cur_nimg:,}  val_loss={avg_valid_loss:.4f}"
            )

            # Best-model checkpoint
            if early_stopper is not None:
                stop_now = early_stopper.update(avg_valid_loss)

                if dist.rank == 0:
                    writer.add_scalar("val/best_loss",  early_stopper.best_loss,  cur_nimg)
                    writer.add_scalar("val/bad_checks", early_stopper.bad_checks, cur_nimg)

                if early_stopper.improved and dist.rank == 0:
                    save_checkpoint(
                        path=best_ckpt_dir,
                        models=model,
                        optimizer=optimizer,
                        epoch=cur_nimg,
                    )
                    logger0.info(
                        f"  [val]  ✓ new best={early_stopper.best_loss:.4f}  "
                        f"best checkpoint saved → {best_ckpt_dir}"
                    )

                if stop_now:
                    logger0.info(
                        f"Early stopping triggered after {early_stopper.bad_checks} "
                        f"checks without improvement  "
                        f"(patience={early_stopper.patience}, "
                        f"best_loss={early_stopper.best_loss:.4f})"
                    )
                    done = True

        # Periodic checkpoint

        if dist.world_size > 1:
            torch.distributed.barrier()

        if is_time_for_periodic_task(
            cur_nimg, cfg.training.io.save_checkpoint_freq, done,
            cfg.training.hp.total_batch_size, dist.rank, rank_0_only=True,
        ):
            save_checkpoint(
                path=ckpt_dir,
                models=model,
                optimizer=optimizer,
                epoch=cur_nimg,
            )
            logger0.info(f"Checkpoint saved  cur_nimg={cur_nimg:,}")

    # Finalise

    if dist.rank == 0:
        writer.close()
        _save_loss_curve(
            train_history=train_loss_history,
            val_history=val_loss_history,
            out_path=os.path.join(run_dir, "loss_curve.png"),
            title="CorrFlow — training & validation loss",
            logger=logger0,
        )
    logger0.info("CorrFlow training completed.")


if __name__ == "__main__":
    main()
