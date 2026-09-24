# One-step residual flow matching for probabilistic precipitation downscaling

Code for the paper *One-step residual flow matching for probabilistic
precipitation downscaling: accuracy, dispersion, and cost.*

CorrFlow keeps the two-stage residual decomposition of CorrDiff (a regression
U-Net for the conditional mean, plus a generative corrector for the residual)
and replaces the diffusion corrector with conditional flow matching, generating
each ensemble member in a single Euler step.

## Layout

    modulus_patch/   Changes to NVIDIA Modulus v0.9.0 (see modulus_patch/README.md)
    preprocessing/   ERA5/ERA5-Land download, de-accumulation, log1p + regrid, zarr store, stats
    training/        Regression, diffusion, and flow-matching training
    evaluation/      Scripts that run inference and cache results
    notebooks/       Renders every paper table and figure from those caches

## Setup

Tested with Python 3.10 and PyTorch 2.1.2 (CUDA 12.1). From the root of this
repository:

1. Install PyTorch for your CUDA version (see pytorch.org). For CUDA 12.1:

       pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

2. Install the remaining dependencies, pinned to the versions used for the paper:

       pip install -r requirements.txt

3. Get NVIDIA Modulus v0.9.0, apply the CorrFlow patch, and install it. The
   `launch` extra is required: every training and evaluation script imports
   `modulus.launch.logging`, which needs it.

       git clone https://github.com/NVIDIA/physicsnemo.git ../modulus
       git -C ../modulus checkout v0.9.0
       cp -r modulus_patch/modulus/*  ../modulus/modulus/
       cp -r modulus_patch/examples/* ../modulus/examples/
       pip install -e "../modulus[launch]"

Then point `MODULUS_ROOT` in the training and evaluation scripts at that clone
(see [Paths](#paths)). `modulus_patch/README.md` lists every patched file.

## Data

| | Product | Resolution |
|---|---|---|
| Input | ERA5 hourly total precipitation | 0.25° |
| Target | ERA5-Land hourly total precipitation | 0.1° |

Domain 5.9–25.0°N, 102.1–118.0°E (192 x 160 cells at 0.1°).
Split: train 2017–2023, validation 2024, evaluation 2025.

Both products come from the Copernicus Climate Data Store.
`preprocessing/download.py` downloads both and expects a CDS API key in `~/.cdsapirc`.

## Pipeline

1. Download: `python preprocessing/download.py`
2. Preprocess: `preprocessing/preprocess.ipynb` de-accumulates ERA5-Land to hourly
   totals, applies log1p on each product's native grid, bilinearly regrids ERA5 to
   the ERA5-Land grid, writes the zarr store, and computes normalization statistics
   over training-period land pixels only.
3. Train, regression first, then either corrector on its frozen residuals:

       python training/train_corrdiff.py --config-name=vietnam_config_training_regression
       python training/train_corrdiff.py --config-name=vietnam_config_training_diffusion
       python training/train_corrflow.py --config-name=vietnam_config_training_corrflow

4. Evaluate. The scripts compute and cache; the notebook renders. Run
   `cache_extremes.py` first, since the ablation needs its
   `extreme_meta.json`. All four write to `OUT_DIR`.

       python evaluation/cache_extremes.py
       python evaluation/cache_sample.py
       python evaluation/solver_ablation.py
       python evaluation/power_spectra.py

   Then run `notebooks/results.ipynb` top to bottom.

| Script | Produces |
|---|---|
| `cache_extremes.py` | Heavy-rain subset selection, per-timestep caches, case timestamps |
| `cache_sample.py` | Broad month-stratified sample scores |
| `solver_ablation.py` | Solver sweep and latency (`tab6_ablation.csv`) |
| `power_spectra.py` | Radial power spectra (`psd_results.npz`) |

The notebook reads these and produces the score tables, reliability diagrams,
rank histogram, spread-skill plot, per-hour boxplots, and case-study figures.

## Paths

These scripts carry absolute paths from the machine the experiments ran on.
Edit these variables before running:

| File | Variables |
|---|---|
| `preprocessing/download.py` | `RAW_INPUT_TP`, `RAW_OUTPUT_DIR` |
| `preprocessing/preprocess.ipynb` | `RAW_INPUT_TP`, `RAW_OUTPUT_DIR`, `PROCESSED_INPUT`, `PROCESSED_OUTPUT`, `ZARR_DIR` |
| `preprocessing/viz_utils.py` | `RAW_INPUT_TP`, `RAW_OUTPUT_DIR`, `PROCESSED_OUTPUT` |
| `training/train_corrdiff.py` | `MODULUS_ROOT`, `run_dir` |
| `training/train_corrflow.py` | `MODULUS_ROOT`, `run_dir` |
| `evaluation/cache_extremes.py` | `MODULUS_ROOT`, `OUT_DIR` |
| `evaluation/cache_sample.py` | `MODULUS_ROOT`, `OUT_DIR` |
| `evaluation/solver_ablation.py` | `MODULUS_ROOT`, `OUT_DIR`, `META_DIR` |
| `evaluation/power_spectra.py` | `MODULUS_ROOT`, `OUT_DIR` |
| `notebooks/results.ipynb` | `MODULUS_ROOT`, `ZARR_PATH`, `STATS_PATH`, `RAW_TP_DIR`, `OUT_DIR`, `REG_CKPT`, `DIFF_CKPT`, `FLOW_CKPT` |
| `<modulus-root>/examples/generative/corrdiff/conf/dataset/vietnam.yaml` | `data_path`, `stats_path` |

`MODULUS_ROOT` must point at the Modulus v0.9.0 clone with `modulus_patch/`
applied; the scripts and `results.ipynb` find the CorrDiff datasets and Hydra configs there, so
they can be run from any directory. `solver_ablation.py` also accepts
`--out-dir`.

## License

Apache License 2.0; see `LICENSE`. Files under `modulus_patch/` derive from
NVIDIA Modulus and retain NVIDIA's copyright headers.
