"""Fixed validation set for choosing checkpoints.

The original validation loss drew a different random subset of 2024 hours, random flow times t
and random noise at every check, so it jumped between about 0.23 and 0.50 and the lowest value
partly picked a lucky draw. Here the set is built once and is identical for every checkpoint and
every run:

- every validation hour (optionally every n-th, for smoke tests),
- for a flow corrector: each hour at the same K flow times t_k = (k + 0.5) / K, with noise drawn
  from a generator reset to the same seed at every check,
- for a deterministic corrector: the squared error of the predicted residual.

The returned value is the mean over all (hour, t) pairs of the land-masked squared error, the same
quantity the training losses minimise.
"""
import torch


@torch.no_grad()
def build_fixed_val_set(validation_dataset, regression_net, device, hr_mean_conditioning,
                        amp_dtype, enable_amp, hour_stride=1, batch_size=32, num_workers=4):
    """Run the frozen regression net once over the validation hours.

    Returns a dict of CPU tensors: residual (N, C_out, H, W) and cond (N, C_cond, H, W), both in
    the normalised space used for training.
    """
    idx = list(range(0, len(validation_dataset), hour_stride))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(validation_dataset, idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    residuals, conds = [], []
    for img_clean, img_lr, *_ in loader:
        img_clean = img_clean.to(device).to(torch.float32)
        img_lr    = img_lr.to(device).to(torch.float32)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
            B, _, H, W = img_lr.shape
            dummy = torch.zeros(B, regression_net.img_out_channels, H, W, device=device)
            mu = regression_net(dummy, img_lr=img_lr, sigma=torch.zeros(B, device=device))
        mu = mu.to(img_clean.dtype)
        residuals.append((img_clean - mu).cpu())
        conds.append((torch.cat([mu, img_lr], dim=1) if hr_mean_conditioning else img_lr).cpu())
    return {"residual": torch.cat(residuals), "cond": torch.cat(conds), "hours": idx}


def _masked_sq_err_sum(pred, target, land_mask):
    """Sum of land-masked squared errors and the number of land values it covers."""
    sq_err = (pred.float() - target) ** 2
    mask = land_mask.to(pred.device)                       # (1, H, W)
    return (sq_err * mask).sum().item(), mask.sum().item() * sq_err.shape[0] * sq_err.shape[1]


@torch.no_grad()
def fixed_val_loss(model, val_set, corrector, land_mask, device, amp_dtype, enable_amp,
                   t_points=4, noise_seed=2024, batch_size=32):
    """Mean land-masked squared error over the fixed set; identical inputs at every call."""
    was_training = model.training
    model.eval()
    residual, cond = val_set["residual"], val_set["cond"]
    n = residual.shape[0]
    total, count = 0.0, 0.0
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    t_grid = [(k + 0.5) / t_points for k in range(t_points)] if corrector == "flow" else [None]
    for t_k in t_grid:
        for s in range(0, n, batch_size):
            res_b  = residual[s:s + batch_size].to(device, non_blocking=True)
            cond_b = cond[s:s + batch_size].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                if corrector == "flow":
                    B = res_b.shape[0]
                    r0 = torch.randn(res_b.shape, generator=gen, device=device)
                    t = torch.full((B,), t_k, device=device)
                    rt = (1 - t_k) * r0 + t_k * res_b
                    pred, target = model(rt, t, cond_b), res_b - r0
                else:
                    pred, target = model(cond_b), res_b
            s_err, s_cnt = _masked_sq_err_sum(pred, target, land_mask)
            total, count = total + s_err, count + s_cnt
    model.train(was_training)
    return total / count
