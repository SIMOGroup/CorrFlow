import datetime
import json
from typing import List, Tuple, Union

import torch
import numpy as np
import xarray as xr

from modulus.utils.generative import convert_datetime_to_cftime
from .base import ChannelMetadata, DownscalingDataset


class VietnamDataset(DownscalingDataset):
    """
    Full-grid Vietnam dataset for CorrDiff / CorrFlow.

    Supports an optional `time_range` filter so the same zarr can be used
    for both the training and validation split without duplicating data on disk.

    Parameters
    ----------
    data_path          : str   – path to the consolidated zarr store
    stats_path         : str   – path to stats.json (produced by preprocessing)
    input_variables    : list  – variable names for the LR conditioning tensor
    output_variables   : list  – variable names for the HR target tensor
    invariant_variables: list  – (unused, kept for API parity)
    time_range         : list  – optional [start_year, end_year] inclusive,
                                 e.g. [2017, 2023] for train or [2024, 2024] for val
    """

    def __init__(
        self,
        data_path: str,
        stats_path: str,
        input_variables=None,
        output_variables=None,
        invariant_variables=None,
        time_range: Union[List[int], None] = None,
    ):
        self.ds = xr.open_dataset(
            data_path,
            engine="zarr",
            chunks=None,
        )

        # ── Optional time-range filter ─────────────────────────────────────
        if time_range is not None:
            start_year, end_year = int(time_range[0]), int(time_range[1])
            years = self.ds["time"].dt.year.values
            mask  = (years >= start_year) & (years <= end_year)
            self.ds = self.ds.isel(time=mask)

        self.input_variables    = input_variables
        self.output_variables   = output_variables
        self.invariant_variables = invariant_variables

        self._in_vars  = tuple(self.input_variables)
        self._out_vars = tuple(self.output_variables)

        self.times = self.ds["time"].values

        self.img_shape = (
            self.ds.sizes["latitude"],
            self.ds.sizes["longitude"],
        )

        # ── Land mask ──────────────────────────────────────────────────────
        # Derived from the NaN pattern of the output variable (before any
        # fill).  Shape (1, H, W), float32: 1 = land, 0 = ocean.
        # Kept on CPU; move to GPU inside the loss functions.
        _tp_sample = self.ds[self._out_vars[0]].isel(time=0).values
        self.land_mask = torch.from_numpy(
            (~np.isnan(_tp_sample)).astype(np.float32)
        ).unsqueeze(0)   # (1, H, W)
        del _tp_sample

        # Load stats (small JSON)
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.input_mean,  self.input_std  = _load_stats(stats, input_variables,  "input")
        self.output_mean, self.output_std = _load_stats(stats, output_variables, "output")

    def __len__(self):
        return self.ds.sizes["time"]

    def __getitem__(self, idx):
        ds_t = self.ds.isel(time=idx)

        x = np.stack(
            [ds_t[v].values for v in self._in_vars], axis=0
        ).astype(np.float32)

        y = np.stack(
            [ds_t[v].values for v in self._out_vars], axis=0
        ).astype(np.float32)

        # Fill NaN ocean pixels with 0 before normalization.
        # All input variables have NaN on ocean (set during preprocessing).
        # Without this, NaNs propagate through normalization and poison training.
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        x = self.normalize_input(x)
        y = self.normalize_output(y)

        return (y, x, torch.tensor(idx, dtype=torch.long))

    # ── API ────────────────────────────────────────────────────────────────

    def image_shape(self) -> Tuple[int, int]:
        return self.img_shape

    def input_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v) for v in self.input_variables]

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self):
        datetimes = (
            datetime.datetime.utcfromtimestamp(t.tolist() / 1e9) for t in self.times
        )
        return [convert_datetime_to_cftime(t) for t in datetimes]

    def normalize_input(self, x):
        return (x - self.input_mean) / (self.input_std + 1e-6)

    def denormalize_input(self, x):
        return x * (self.input_std + 1e-6) + self.input_mean

    def normalize_output(self, x):
        return (x - self.output_mean) / (self.output_std + 1e-6)

    def denormalize_output(self, x):
        return x * (self.output_std + 1e-6) + self.output_mean

    def latitude(self) -> np.ndarray:
        lat  = self.ds["latitude"].values
        lon  = self.ds["longitude"].values
        lat2d, _ = np.meshgrid(lat, lon, indexing="ij")
        return lat2d.astype(np.float32)

    def longitude(self) -> np.ndarray:
        lat  = self.ds["latitude"].values
        lon  = self.ds["longitude"].values
        _, lon2d = np.meshgrid(lat, lon, indexing="ij")
        return lon2d.astype(np.float32)


def _load_stats(stats, variables, group):
    mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None]
    std  = np.array([stats[group][v]["std"]  for v in variables])[:, None, None]
    return mean.astype(np.float32), std.astype(np.float32)
