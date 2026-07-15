# Modulus patch for CorrFlow

These files modify or extend **NVIDIA Modulus v0.9.0**
(https://github.com/NVIDIA/modulus), from a snapshot taken 2026-03-23.

Redistributed under the Apache License 2.0; see `../LICENSE`. NVIDIA's original
copyright headers are retained in every derived file.

## Modified from Modulus v0.9.0

Derived from NVIDIA originals; copyright headers retained.

| File | Change |
|---|---|
| `modulus/metrics/diffusion/loss.py` | Conditional flow-matching residual loss |
| `modulus/models/diffusion/song_unet.py` | Broadcast time channel for the flow corrector |
| `modulus/models/diffusion/unet.py` | Supporting change for the above |
| `examples/generative/corrdiff/conf/sampler/deterministic.yaml` | Heun step counts for the solver sweep |

## New in this work

Not derived from Modulus.

| File | Purpose |
|---|---|
| `examples/generative/corrdiff/datasets/vietnam.py` | ERA5 / ERA5-Land Vietnam dataset |
| `examples/generative/corrdiff/conf/dataset/vietnam.yaml` | Dataset config |
| `examples/generative/corrdiff/conf/generation/vietnam.yaml` | CorrDiff generation |
| `examples/generative/corrdiff/conf/generation/vietnamflow.yaml` | CorrFlow generation |
| `examples/generative/corrdiff/conf/model/vietnam_corrflow_model.yaml` | CorrFlow model |
| `examples/generative/corrdiff/conf/training/vietnam_corrdiff_diffusion.yaml` | CorrDiff diffusion training |
| `examples/generative/corrdiff/conf/training/vietnam_corrdiff_regression.yaml` | Regression training |
| `examples/generative/corrdiff/conf/training/vietnam_corrflow_training.yaml` | CorrFlow training |
| `examples/generative/corrdiff/conf/vietnam_config_generate_corrflow.yaml` | Top-level CorrFlow generation |
| `examples/generative/corrdiff/conf/vietnam_config_training_corrflow.yaml` | Top-level CorrFlow training |
| `examples/generative/corrdiff/conf/vietnam_config_training_diffusion.yaml` | Top-level diffusion |
| `examples/generative/corrdiff/conf/vietnam_config_training_regression.yaml` | Top-level regression |

## Applying

Obtain Modulus v0.9.0, then overlay these files onto the clone:

    cp -r modulus_patch/modulus/*  <modulus-root>/modulus/
    cp -r modulus_patch/examples/* <modulus-root>/examples/

Paths mirror the upstream tree, so the copy lands each file in place.
