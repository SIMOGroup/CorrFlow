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

Then set the environment variable `MODULUS_ROOT` to that clone (see [Paths](#paths)). `modulus_patch/README.md` lists every patched file.

## Data

| | Product | Resolution |
|---|---|---|
| Input | ERA5 hourly total precipitation | 0.25° |
| Target | ERA5-Land hourly total precipitation | 0.1° |
| Independent check | IMERG V07 Final run; GSMaP v8, gauge-adjusted | 0.1° |

Domain 5.9–25.0°N, 102.1–118.0°E (192 x 160 cells at 0.1°).
Split: train 2017–2023, validation 2024, evaluation 2025.

ERA5 and ERA5-Land come from the Copernicus Climate Data Store.
`preprocessing/download.py` downloads both and expects a CDS API key in `~/.cdsapirc`.
`preprocessing/download_independent.py` downloads IMERG and GSMaP for 2024–2025 from Google Earth
Engine; it needs `earthengine-api` and a Google Cloud project with Earth Engine access
(`--project`, or the environment variable `EE_PROJECT`).

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

   This CorrFlow run is seed 0; the paper uses its checkpoint with the lowest validation loss,
   `FlowUNet.0.20620032.pt`. CorrFlow seeds 1–3 and the deterministic residual corrector CorrDet
   use `vietnam_config_training_corrflow_revision.yaml`: the same model, a seed per run, a budget of
   25 million samples and a fixed validation set (`training/fixed_validation.py`) checked every
   million samples, whose lowest value chooses the checkpoint:

       python training/train_corrflow.py --config-name=vietnam_config_training_corrflow_revision \
           training.seed=1 model.corrector=flow run.out_dir=$CORRFLOW_OUT/corrflow_seed1
       python training/train_corrflow.py --config-name=vietnam_config_training_corrflow_revision \
           training.seed=1 model.corrector=deterministic run.out_dir=$CORRFLOW_OUT/corrdet_seed1

   `evaluation/rescore_checkpoints.py` scores the checkpoints of a run on the same fixed validation set.

4. Evaluate. The scripts compute and cache; the notebook renders. Run
   `cache_extremes.py` first, since the ablation needs its
   `extreme_meta.json`. All four write to `OUT_DIR`. They use the CorrFlow seed 0 checkpoint above.

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

5. Evaluate as in the paper. The scripts in `evaluation/revision/` add the four CorrFlow runs,
   CorrDet, bilinear interpolation in mm/hr, scores against IMERG and GSMaP, and 95% intervals from a
   day-block bootstrap. They read the hour lists and cached predictions of step 4 (set its `OUT_DIR`
   to `$CORRFLOW_OUT/results`) and take every path from `evaluation/revision/paths.py`. The 2025
   evaluation year opens only with `--allow-test`, after every checkpoint has been chosen on 2024.
   From `evaluation/revision/`:

       python predict.py --run $CORRFLOW_OUT/corrflow_seed1 --name corrflow_s1 --split test --allow-test
       python predict.py --run $CORRFLOW_OUT/corrdet_seed1 --name deterministic_s1 --split test --allow-test
       python score_sets.py --split test --allow-test --reference era5land \
           --method unet=cache:unet_mm --method bilinear_mm=bilinear_mm --method corrdiff=cache:cd_mm \
           --method corrflow_s0=cache:cf_mm --method corrflow_s1=preds:$CORRFLOW_OUT/evaluation/preds/corrflow_s1
       python bootstrap.py --split test --subset sample --method unet --method corrdiff --method corrflow_s0 \
           --group corrflow_s0to3=corrflow_s0,corrflow_s1,corrflow_s2,corrflow_s3 \
           --compare corrflow_s0:corrdiff --out $CORRFLOW_OUT/evaluation/tables/test_sample_vs_era5land.csv

   Repeat `predict.py` for seeds 2 and 3, `score_sets.py` with `--reference imerg` and `gsmap`, and
   `bootstrap.py` with `--subset heavy`. Each script documents its options.

| Script (`evaluation/revision/`) | Produces |
|---|---|
| `predict.py`, `unet_later.py` | Predictions of CorrFlow seeds 1–3, CorrDet and the last U-Net checkpoint |
| `score_sets.py`, `scores.py` | Per-hour scores against ERA5-Land, IMERG or GSMaP |
| `bootstrap.py` | Score tables with 95% day-block intervals and paired differences |
| `period_tags.py` | Scores split by period (1 July–17 November against the other months) |
| `reference_quality.py` | ERA5-Land and ERA5 against IMERG and GSMaP |
| `noise_sensitivity.py` | Noise cancellation and sensitivity of the trained networks near t = 0 |
| `steps_sweep.py`, `timing.py` | Sampling steps and solvers; latency of every setting |
| `year_consistency.py`, `which_side.py`, `ee_era5land_check.py`, `corrdiff_ckpt_check.py` | Checks of the ERA5 and ERA5-Land mismatch in 2024 and of the checkpoints |
| `figure1_data.py`, `figure1_study_area.py` | Study-area figure |
| `revision_figures.py` | Calibration, trade-off and spectra figures; rows of the cost table |
| `paper_figures.py` | Case maps and case tables (cells of `notebooks/results.ipynb`) |
| `check_exceedance.py` | Tests of the derived scores against direct computations |

## Paths

These scripts and configs use placeholder paths: `/path/to/data`, `/path/to/outputs` and
`/path/to/modulus`. Set these variables before running:

| File | Variables |
|---|---|
| `preprocessing/download.py` | `RAW_INPUT_TP`, `RAW_OUTPUT_DIR` |
| `preprocessing/preprocess.ipynb` | `RAW_INPUT_TP`, `RAW_OUTPUT_DIR`, `PROCESSED_INPUT`, `PROCESSED_OUTPUT`, `ZARR_DIR` |
| `preprocessing/viz_utils.py` | `RAW_INPUT_TP`, `RAW_OUTPUT_DIR`, `PROCESSED_OUTPUT` |
| `training/train_corrdiff.py` | `MODULUS_ROOT`; the run folder is `$CORRFLOW_OUT/<job name>` |
| `training/train_corrflow.py` | `MODULUS_ROOT`; the run folder is `run.out_dir`, or `$CORRFLOW_OUT/<job name>` |
| `evaluation/revision/paths.py` | `CORRFLOW_DATA`, `CORRFLOW_OUT` (environment variables; the layout is in the file) |
| `evaluation/cache_extremes.py`, `cache_sample.py`, `power_spectra.py` | `MODULUS_ROOT`, `ZARR_PATH`, `STATS_PATH`, `REG_CKPT`, `DIFF_CKPT`, `FLOW_CKPT`, `OUT_DIR` |
| `evaluation/solver_ablation.py` | the same, and `META_DIR` |
| `notebooks/results.ipynb` | `MODULUS_ROOT`, `ZARR_PATH`, `STATS_PATH`, `RAW_TP_DIR`, `OUT_DIR`, `REG_CKPT`, `DIFF_CKPT`, `FLOW_CKPT` |
| `<modulus-root>/examples/generative/corrdiff/conf/dataset/vietnam.yaml` and the `vietnam_config_training_*.yaml` files | `data_path`, `stats_path` |
| `conf/training/vietnam_*_training.yaml`, `conf/training/vietnam_corrdiff_diffusion.yaml` | `regression_checkpoint_path` |
| `conf/generation/vietnam.yaml`, `conf/generation/vietnamflow.yaml` | the checkpoint file names |

`MODULUS_ROOT` (an environment variable, or the variable at the top of each script) must point
at the Modulus v0.9.0 clone with `modulus_patch/` applied; the scripts and `results.ipynb` find the CorrDiff datasets and Hydra configs there, so
they can be run from any directory. `solver_ablation.py` also accepts
`--out-dir`.

## License

Apache License 2.0; see `LICENSE`. Files under `modulus_patch/` derive from
NVIDIA Modulus and retain NVIDIA's copyright headers.
