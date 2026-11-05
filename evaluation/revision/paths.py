"""Where the scripts in this folder read data and trained models, and where they write results.

Set two environment variables before running, or edit the defaults below:

    CORRFLOW_DATA   the data folder
    CORRFLOW_OUT    the folder with the trained runs; results are written below it

Expected layout:

    $CORRFLOW_DATA/vietnam_data.zarr          zarr store from preprocessing/preprocess.ipynb
    $CORRFLOW_DATA/stats.json                 normalization statistics from the same notebook
    $CORRFLOW_DATA/era5/era5_sl_tp_<year>.nc  raw 0.25 deg ERA5 hourly precipitation (preprocessing/download.py)
    $CORRFLOW_DATA/satellite/                 IMERG and GSMaP (preprocessing/download_independent.py)

    $CORRFLOW_OUT/<run>/                      one folder per training run (see RUNS; set the names to your runs)
    $CORRFLOW_OUT/results/                    caches of evaluation/cache_extremes.py and cache_sample.py
    $CORRFLOW_OUT/evaluation/                 everything written by the scripts in this folder
"""
import os

DATA = os.environ.get("CORRFLOW_DATA", "/path/to/data")
OUT = os.environ.get("CORRFLOW_OUT", "/path/to/outputs")

ZARR = os.path.join(DATA, "vietnam_data.zarr")
STATS = os.path.join(DATA, "stats.json")
RAW_ERA5 = os.path.join(DATA, "era5", "era5_sl_tp_{year}.nc")
SATELLITE = os.path.join(DATA, "satellite")

# Training runs, one folder each
RUNS = {
    "regression": os.path.join(OUT, "regression"),
    "corrdiff": os.path.join(OUT, "corrdiff"),
    "corrflow_s0": os.path.join(OUT, "corrflow_seed0"),
    "corrflow_s1": os.path.join(OUT, "corrflow_seed1"),
    "corrflow_s2": os.path.join(OUT, "corrflow_seed2"),
    "corrflow_s3": os.path.join(OUT, "corrflow_seed3"),
    "corrdet_s1": os.path.join(OUT, "corrdet_seed1"),
}

# Checkpoints used in the paper: lowest validation loss of each run
REG_CKPT = os.path.join(RUNS["regression"], "checkpoints_regression_best", "UNet.0.315136.mdlus")
DIFF_CKPT = os.path.join(RUNS["corrdiff"], "checkpoints_diffusion_best", "EDMPrecondSR.0.17310208.mdlus")
FLOW_CKPT = os.path.join(RUNS["corrflow_s0"], "checkpoints_flow_best", "FlowUNet.0.20620032.pt")

RESULTS = os.path.join(OUT, "results")          # hour lists and cached predictions of the paper's models
EVAL = os.path.join(OUT, "evaluation")          # predictions, scores, tables and figures of this folder
