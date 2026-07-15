import os
import sys
import time
import psutil
import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from modulus import Module
from modulus.models.diffusion import UNet, EDMPrecondSR
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.metrics.diffusion import RegressionLoss, ResLoss, RegressionLossCE
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


###############################################################################
# Early stopping
###############################################################################

class EarlyStopper:
    """
    Tracks validation loss and signals when training should stop.

    Stops when validation loss has not improved by more than `min_delta`
    for `patience` consecutive validation checks.  Also keeps track of the
    best loss seen so far so the caller can decide when to save a
    best-model checkpoint.

    Parameters
    ----------
    patience  : int   – number of validation checks without improvement before stopping
    min_delta : float – minimum absolute improvement that counts as "better"
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.bad_checks = 0          # consecutive checks without improvement
        self.improved   = False      # set True whenever best_loss is updated

    def update(self, val_loss: float) -> bool:
        """
        Feed the latest validation loss.

        Returns True if training should stop, False otherwise.
        Sets self.improved = True if this is a new best.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.bad_checks = 0
            self.improved   = True
        else:
            self.bad_checks += 1
            self.improved   = False

        return self.bad_checks >= self.patience


###############################################################################
# Loss-curve helper
###############################################################################

def _save_loss_curve(train_history, val_history, out_path, title, logger):
    """
    Save a PNG showing training loss (every step) and validation loss (sparse).

    Uses a log-scale y-axis so both the noisy per-step train loss and the
    smoother val loss are readable on the same axes.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # non-interactive backend — safe on HPC
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))

        if train_history:
            t_steps, t_losses = zip(*train_history)
            ax.plot(t_steps, t_losses, color="#4C72B0", alpha=0.35,
                    linewidth=0.8, label="train loss (per step)")

            # Overlay a simple running mean so the trend is visible through noise
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

            # Mark the best val point
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


###############################################################################
# Main
###############################################################################

def main():

    ###########################################################################
    # Distributed ENV
    ###########################################################################

    os.environ["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"] = "ENV"
    os.environ["RANK"]        = "0"
    os.environ["WORLD_SIZE"]  = "1"
    os.environ["LOCAL_RANK"]  = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

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

    ###########################################################################
    # Load Hydra config
    ###########################################################################

    from hydra import initialize, compose

    with initialize(
        version_base="1.2",
        config_path="physicsnemo/examples/generative/corrdiff/conf",
    ):
        cfg = compose(config_name="vietnam_config_training_diffusion")

    print(OmegaConf.to_yaml(cfg))

    ###########################################################################
    # Run directory
    ###########################################################################

    job_name = cfg.run.name
    run_dir  = f"/mnt/data/khaiht/outputs/{job_name}_hope_sanity"
    tb_dir   = os.path.join(run_dir, "tensorboard")
    ckpt_dir = os.path.join(run_dir, f"checkpoints_{cfg.model.name}")
    best_ckpt_dir = os.path.join(run_dir, f"checkpoints_{cfg.model.name}_best")

    os.makedirs(tb_dir,        exist_ok=True)
    os.makedirs(ckpt_dir,      exist_ok=True)
    os.makedirs(best_ckpt_dir, exist_ok=True)
    os.chdir(run_dir)

    print("Job name     :", job_name)
    print("Run dir      :", run_dir)
    print("TensorBoard  :", tb_dir)
    print("Checkpoints  :", ckpt_dir)
    print("Best ckpt    :", best_ckpt_dir)

    ###########################################################################
    # Logging
    ###########################################################################

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

    ###########################################################################
    # Distributed init
    ###########################################################################

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.rank == 0:
        writer = SummaryWriter(log_dir=tb_dir)

    logger  = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)

    print("World size:", dist.world_size)
    print("Device    :", dist.device)

    OmegaConf.resolve(cfg)

    ###########################################################################
    # Dataset config
    ###########################################################################

    dataset_cfg = OmegaConf.to_container(cfg.dataset)

    if hasattr(cfg, "validation"):
        train_test_split       = True
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation)
    else:
        train_test_split       = False
        validation_dataset_cfg = None

    ###########################################################################
    # Training performance options
    ###########################################################################

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

    ###########################################################################
    # Dataset loading
    ###########################################################################

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

    ###########################################################################
    # Dataset channels
    ###########################################################################

    prob_channels    = None
    dataset_channels = len(dataset.input_channels())
    img_in_channels  = dataset_channels
    img_shape        = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index()

    if cfg.model.name in ("patched_diffusion", "lt_aware_patched_diffusion"):
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None

    patch_shape = (patch_shape_y, patch_shape_x)
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)

    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")

    if img_shape[1] != patch_shape[1]:
        img_in_channels += dataset_channels

    print("img_shape       :", img_shape)
    print("patch_shape     :", patch_shape)
    print("img_in_channels :", img_in_channels)
    print("img_out_channels:", img_out_channels)

    ###########################################################################
    # Model configuration
    ###########################################################################

    model_args = {
        "img_out_channels": img_out_channels,
        "img_resolution":   list(img_shape),
        "use_fp16":         fp16,
    }

    standard_model_cfgs = {
        "regression": {
            "img_channels":        4,
            "N_grid_channels":     4,
            "embedding_type":      "zero",
            "checkpoint_level":    songunet_checkpoint_level,
        },
        "lt_aware_ce_regression": {
            "img_channels":        4,
            "N_grid_channels":     4,
            "embedding_type":      "zero",
            "lead_time_channels":  4,
            "lead_time_steps":     9,
            "prob_channels":       prob_channels,
            "checkpoint_level":    songunet_checkpoint_level,
            "model_type":          "SongUNetPosLtEmbd",
        },
        "diffusion": {
            "img_channels":        img_out_channels,
            "gridtype":            "sinusoidal",
            "N_grid_channels":     4,
            "checkpoint_level":    songunet_checkpoint_level,
        },
        "patched_diffusion": {
            "img_channels":        img_out_channels,
            "gridtype":            "learnable",
            "N_grid_channels":     100,
            "checkpoint_level":    songunet_checkpoint_level,
        },
        "lt_aware_patched_diffusion": {
            "img_channels":        img_out_channels,
            "gridtype":            "learnable",
            "N_grid_channels":     100,
            "lead_time_channels":  20,
            "lead_time_steps":     9,
            "checkpoint_level":    songunet_checkpoint_level,
            "model_type":          "SongUNetPosLtEmbd",
        },
    }

    model_args.update(standard_model_cfgs[cfg.model.name])
    if cfg.model.name in ("diffusion", "patched_diffusion", "lt_aware_patched_diffusion"):
        model_args["scale_cond_input"] = cfg.model.scale_cond_input
    if hasattr(cfg.model, "model_args"):
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    ###########################################################################
    # Model creation
    ###########################################################################

    if cfg.model.name == "regression":
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif cfg.model.name == "lt_aware_ce_regression":
        model = UNet(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name == "lt_aware_patched_diffusion":
        model = EDMPrecondSR(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    else:
        model = EDMPrecondSR(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )

    model.train().requires_grad_(True).to(dist.device)
    print("Model created:", model.img_in_channels, "→", model.img_out_channels)

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )

    ###########################################################################
    # Regression net (needed by diffusion/ResLoss)
    ###########################################################################

    if hasattr(cfg.training.io, "regression_checkpoint_path"):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        regression_net = Module.from_checkpoint(regression_checkpoint_path)
        regression_net.eval().requires_grad_(False).to(dist.device)
        print("Loaded regression net:", regression_checkpoint_path)

    ###########################################################################
    # Loss
    ###########################################################################

    patch_num = getattr(cfg.training.hp, "patch_num", 1)

    if cfg.model.name in ("diffusion", "patched_diffusion", "lt_aware_patched_diffusion"):
        loss_fn = ResLoss(
            regression_net=regression_net,
            img_shape_x=img_shape[1],
            img_shape_y=img_shape[0],
            patch_shape_x=patch_shape[1],
            patch_shape_y=patch_shape[0],
            patch_num=patch_num,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
            land_mask=dataset.land_mask,
        )
    elif cfg.model.name == "regression":
        loss_fn = RegressionLoss(land_mask=dataset.land_mask)
    elif cfg.model.name == "lt_aware_ce_regression":
        loss_fn = RegressionLossCE(prob_channels=prob_channels)

    ###########################################################################
    # Optimizer
    ###########################################################################

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

    ###########################################################################
    # Early stopping setup
    ###########################################################################

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

    ###########################################################################
    # Resume checkpoint
    ###########################################################################

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

    ###########################################################################
    # Training loop
    ###########################################################################

    logger0.info(f"Training for {cfg.training.hp.training_duration:,} images...")

    done                        = False
    average_loss_running_mean   = 0.0
    n_average_loss_running_mean = 1

    # History for end-of-training loss curve (rank-0 only)
    train_loss_history = []   # [(cur_nimg, loss), ...]
    val_loss_history   = []   # [(cur_nimg, loss), ...]

    while not done:

        tick_start_nimg = cur_nimg
        tick_start_time = time.time()

        #######################################################################
        # Forward + backward
        #######################################################################

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(num_accumulation_rounds):

            img_clean, img_lr, labels, *lead_time_label = next(dataset_iterator)

            img_clean = img_clean.to(dist.device).to(torch.float32).contiguous()
            img_lr    = img_lr.to(dist.device).to(torch.float32).contiguous()
            labels    = labels.to(dist.device).contiguous()

            if lead_time_label:
                lead_time_label = lead_time_label[0].to(dist.device).contiguous()
            else:
                lead_time_label = None

            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                loss = loss_fn(
                    net=model,
                    img_clean=img_clean,
                    img_lr=img_lr,
                    labels=labels,
                    augment_pipe=None,
                )

            loss = loss.sum() / batch_size_per_gpu
            loss_accum += loss.item() / num_accumulation_rounds
            loss.backward()

        #######################################################################
        # Reduce loss across GPUs
        #######################################################################

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

        #######################################################################
        # LR schedule (linear ramp-up → exponential decay)
        #######################################################################

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

        #######################################################################
        # Progress print + running-mean reset
        #######################################################################

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
                f"train_loss {average_loss:<7.4f}",
                f"train_loss_rm {average_loss_running_mean:<7.4f}",
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

        #######################################################################
        # Validation
        #######################################################################

        avg_valid_loss = None

        if validation_dataset_iterator is not None and is_time_for_periodic_task(
            cur_nimg, cfg.training.io.validation_freq, done,
            cfg.training.hp.total_batch_size, dist.rank,
        ):
            model.eval()
            valid_loss_accum = 0.0

            with torch.no_grad():
                for _ in range(cfg.training.io.validation_steps):
                    img_clean_v, img_lr_v, labels_v, *_ = next(
                        validation_dataset_iterator
                    )
                    img_clean_v = img_clean_v.to(dist.device).to(torch.float32).contiguous()
                    img_lr_v    = img_lr_v.to(dist.device).to(torch.float32).contiguous()
                    labels_v    = labels_v.to(dist.device).contiguous()

                    with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                        loss_v = loss_fn(
                            net=model,
                            img_clean=img_clean_v,
                            img_lr=img_lr_v,
                            labels=labels_v,
                            augment_pipe=None,
                        )

                    valid_loss_accum += (
                        loss_v.sum() / batch_size_per_gpu
                    ).item() / cfg.training.io.validation_steps

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

            # ── Best-model checkpoint ────────────────────────────────────────
            if early_stopper is not None:
                stop_now = early_stopper.update(avg_valid_loss)

                if dist.rank == 0:
                    writer.add_scalar("val/best_loss", early_stopper.best_loss, cur_nimg)
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
                    done = True  # break out of the while-loop cleanly

        #######################################################################
        # Periodic checkpoint
        #######################################################################

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

    ###########################################################################
    # Finalise
    ###########################################################################

    if dist.rank == 0:
        writer.close()
        _save_loss_curve(
            train_history=train_loss_history,
            val_history=val_loss_history,
            out_path=os.path.join(run_dir, "loss_curve.png"),
            title=f"CorrDiff {cfg.model.name} — training & validation loss",
            logger=logger0,
        )
    logger0.info("Training completed.")


if __name__ == "__main__":
    main()
