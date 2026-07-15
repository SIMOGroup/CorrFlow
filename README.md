# One-step residual flow matching for probabilistic precipitation downscaling

Code for the paper *One-step residual flow matching for probabilistic
precipitation downscaling: accuracy, dispersion, and cost.*

CorrFlow keeps the two-stage residual decomposition of CorrDiff (a regression
U-Net for the conditional mean, plus a generative corrector for the residual)
and replaces the diffusion corrector with conditional flow matching, generating
each ensemble member in a single Euler step.

## Layout

    modulus_patch/   Changes to NVIDIA Modulus v0.9.0 (see modulus_patch/README.md)
    preprocessing/   ERA5 download, log1p + regrid, zarr store, normalization stats
    training/        Regression, diffusion, and flow-matching training

## Data

| | Product | Resolution |
|---|---|---|
| Input | ERA5 hourly total precipitation | 0.25° |
| Target | ERA5-Land hourly total precipitation | 0.1° |

Domain 5.9–25.0°N, 102.1–118.0°E (192 x 160 cells at 0.1°).
Split: train 2017–2023, validation 2024, evaluation 2025.

Both products come from the Copernicus Climate Data Store.
`preprocessing/download_era5.py` expects a CDS API key in `~/.cdsapirc`.

## Pipeline

1. Download: `preprocessing/download_era5.py`
2. Preprocess: `preprocessing/corrdiff_preprocess_tp_only_v1.ipynb`
   applies log1p on each product's native grid, bilinearly regrids ERA5 to the
   ERA5-Land grid, writes the zarr store, and computes normalization statistics
   over training-period land pixels only.
3. Train, regression first, then either corrector on its frozen residuals:

       python training/train.py      --config-name=vietnam_config_training_regression
       python training/train.py      --config-name=vietnam_config_training_diffusion
       python training/train_flow.py --config-name=vietnam_config_training_corrflow

## Paths

These scripts carry absolute paths from the machine the experiments ran on.
Edit before running:

| File | Line | Variable |
|---|---|---|
| `preprocessing/download_era5.py` | 24 | `out_root` |
| `preprocessing/viz_utils.py` | 106-108 | `RAW_INPUT_TP`, `RAW_OUTPUT_DIR`, `PROCESSED_OUTPUT` |
| `training/train.py` | 186 | `run_dir` |
| `training/train_flow.py` | 298 | `run_dir` |

## License

Apache License 2.0; see `LICENSE`. Files under `modulus_patch/` derive from
NVIDIA Modulus and retain NVIDIA's copyright headers.
