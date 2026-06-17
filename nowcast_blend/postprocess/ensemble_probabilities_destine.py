import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

@dataclass
class RainfallConfig:
    rainfall_var: str
    ens_var: str

    resample_to: str
    thresholds: Optional[List[float]]
    percentiles: Optional[List[float]]
    hours_list: Optional[List[int]]

def load_config(path: str) -> RainfallConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return RainfallConfig(
        rainfall_var=raw["rainfall"]["rainfall_var"],
        ens_var=raw["rainfall"]["ens_var"],
        resample_to=raw["processing"]["resample_to"],
        thresholds=raw.get("thresholds"),
        percentiles=raw.get("percentiles"),
        hours_list=raw.get("hours_list"),
    )

# --- AANPASSEN VOOR OPERATIONEEL ---
config = load_config(r"\configs\zware-buien.yaml")
path = r"RAINFALL_BLENDING_NC.nc"
output_file = Path(r"cRAINFALL_BLENDING_NC_ZB.nc")
# --- AANPASSEN VOOR OPERATIONEEL ---

ds = xr.open_dataset(path)

ds = ds.assign_coords(
    time=ds["time"].values[0] + np.arange(len(ds.time)) * np.timedelta64(1, "h")
)
ds = xr.open_dataset(path)

ds = ds.assign_coords(
    time=ds["time"].values[0] + np.arange(len(ds.time)) * np.timedelta64(1, "h")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logging.info("Start met statistieken berekenen")

# --- Helper function om 2 timesteps te verzekeren ---
def ensure_two_timesteps(da: xr.DataArray) -> xr.DataArray:
    """
    Ensures that a DataArray contains exactly two timesteps.

    If the input has:
    - 3 or more timesteps: selects the first and third timestep
    - 2 timesteps: returns unchanged
    - 1 timestep: duplicates it and applies a small time offset to avoid
    duplicate timestamps
    - 0 timesteps: raises an error


    Arguments
    ---------
    da : xr.DataArray
        Input data array with a time dimension.

    Returns
    -------
    xr.DataArray
        DataArray with exactly two timesteps.
    """
    tlen = da.sizes.get("time", 0)

    if tlen >= 3:
        return da.isel(time=[0, 2])

    if tlen == 2:
        return da

    if tlen == 1:
        t0 = da.time.values[0]
        t1 = t0 + np.timedelta64(1, "s")
        return xr.concat([da, da], dim="time").assign_coords(time=[t0, t1])

    raise ValueError("No timesteps found")

# --- Verwerking van neerslag ---

def process_rainfall(
    XRarray: xr.DataArray,
    config: RainfallConfig
) -> xr.Dataset:
    
    """
    Processes rainfall data to compute aggregated statistics over time.

    The function calculates:
    - Daily aggregated rainfall
    - Threshold exceedance probabilities
    - Percentile-based rainfall metrics
    - Rolling window statistics over multiple time scales

    All outputs are resampled to a common time resolution and aligned
    using a reference model date.

    Intermediate outputs are normalized to ensure consistent time length
    using `ensure_two_timesteps`.


    Arguments
    ---------
    XRarray : xr.DataArray
        Input rainfall dataset with dimensions including time and ensemble.

    config : RainfallConfig
        Configuration object containing:
        - rainfall_var (str): Name of rainfall variable
        - ens_var (str): Name of ensemble dimension
        - resample_to (str): Temporal aggregation frequency
        - thresholds (list[float] | None): Rainfall thresholds in mm
        - percentiles (list[float] | None): Percentiles (0–1 range)
        - hours_list (list[int] | None): Rolling window sizes in hours


    Returns
    -------
    xr.Dataset
        Dataset containing derived rainfall statistics.

        The exact content is fully controlled by the configuration object.
        Available outputs may include:

        - threshold exceedance probabilities (if thresholds are defined)
        - percentile-based rainfall statistics (if percentiles are defined)
        - rolling-window aggregated statistics (if hours_list is defined)

        Only variables enabled in the configuration are computed.

        All variables are aligned to a common time axis and
        constrained to two timesteps per variable.
    """

    results: Dict[str, xr.DataArray] = {}

    ens_var = config.ens_var
    resample_to = config.resample_to

    modelDate = pd.to_datetime(XRarray.time[0].values, "%Y%m%d%H%M%S")

    daily_sum = XRarray.resample(
        time=resample_to,
        skipna=True,
        origin=modelDate,
        label="right"
    ).sum(dim="time")

    # thresholds
    if config.thresholds:
        for th in config.thresholds:
            above_th = daily_sum > th
            chance = (above_th.sum(dim=ens_var) / XRarray.sizes[ens_var]) * 100
            chance.attrs["units"] = "%"
            chance.attrs["long_name"] = f"Chance of rainfall more than {th} mm"
            results[f"precipitation_likelihood_{th}mm"] = ensure_two_timesteps(chance)

    if config.percentiles:
        for p in config.percentiles:
            pval = daily_sum.quantile(dim=ens_var, q=p, skipna=True)
            pval.attrs["units"] = "mm"
            pval = pval.drop_vars("quantile")
            pval.attrs["long_name"] = f"Rainfall {int(p*100)}th percentile"

            results[f"p_{int(p*100)}th_percentile"] = ensure_two_timesteps(pval)

    if config.hours_list:
        for h in config.hours_list:
            if XRarray.sizes["time"] >= h:
                ysum = XRarray.rolling(time=h, min_periods=h).sum().fillna(0.0)

                if config.thresholds:
                    for th in config.thresholds:
                        max_above = (
                            xr.where(ysum > th, 1, 0)
                            .sum(dim=ens_var) / XRarray.sizes[ens_var] * 100
                        )

                        max_resampled = max_above.resample(
                            time=resample_to,
                            skipna=True,
                            origin=modelDate,
                            label="right"
                        ).max(dim="time")
                        max_resampled.attrs["units"] = "%"
                        max_resampled.attrs["long_name"] = f"Chance of rainfall more than {th} mm in {h} hours"
                        results[f"precipitation_likelihood_{th}mm_{h}h"] = ensure_two_timesteps(max_resampled)

                if config.percentiles:
                    for p in config.percentiles:
                        max_p = ysum.quantile(dim=ens_var, q=p, skipna=True)
                        max_p.attrs["units"] = "mm"
                        max_p.attrs["long_name"] = f"Maximum {int(p*100)}th percentile rainfall over {h} hours"
                        max_p = max_p.drop_vars("quantile")

                        max_p = max_p.resample(
                            time=resample_to,
                            skipna=True,
                            origin=modelDate,
                            label="right"
                        ).max(dim="time")

                        results[f"precipitation_{int(p*100)}th_percentile_{h}h"] = ensure_two_timesteps(max_p)

    return xr.Dataset(results)


def add_statistics_to_dataset(
    ds: xr.Dataset,
    XRarray: Union[xr.DataArray, xr.Dataset],
    config: RainfallConfig
) -> xr.Dataset:
    
    """
    Computes rainfall-derived statistics of an xarray Dataset without modifying the input dataset.

    A new Dataset is created, keeping the original time dimension unchanged.
    Variables from the processed statistics are aligned to the original time axis.

    The rainfall statistics are computed using a configuration object that
    defines variable names, aggregation settings, and statistical parameters.

    Arguments
    ---------
    ds : xr.Dataset
        Original dataset containing a 'time' dimension.

    XRarray : xr.DataArray or xr.Dataset
        Input data used for computing rainfall statistics.

    config : RainfallConfig
        Configuration object containing:
        - rainfall_var (str): Name of rainfall variable
        - ens_var (str): Name of ensemble dimension
        - resample_to (str): Temporal aggregation frequency
        - thresholds (list[float] | None): Rainfall thresholds in mm
        - percentiles (list[float] | None): Percentiles (0–1 range)
        - hours_list (list[int] | None): Rolling window sizes in hours

    Returns
    -------
    xr.Dataset
        New dataset containing original variables plus added statistics,
        with the same time dimension as the input dataset.
    """

    new_stats = process_rainfall(XRarray, config)

    def make_time_unique(da: xr.DataArray) -> xr.DataArray:
        if "time" in da.dims:
            times = pd.to_datetime(da.time.values)
            if len(times) != len(np.unique(times)):
                new_times = [t + pd.Timedelta(microseconds=i) for i, t in enumerate(times)]
                da = da.copy()
                da["time"] = new_times
        return da

    ds_out = xr.Dataset(coords={"time": ds.time})

    for var_name, da in new_stats.data_vars.items():
        da = make_time_unique(da)

        if "time" in da.dims:
            if len(da.time) == 1:
                da = da.expand_dims(time=ds_out.time)
                da = da.broadcast_like(ds_out)
            else:
                da = da.reindex(time=ds_out.time, method="nearest")
        else:
            da = da.broadcast_like(ds_out)

        ds_out[var_name] = da


    return ds_out


# --- Main ---
if __name__ == "__main__":

    da = ds[config.rainfall_var]
    da = da.where(np.isfinite(da)).fillna(0)
    da = da.where(da >= 0, 0)

    ds_with_stats = add_statistics_to_dataset(ds, da, config)

    os.makedirs(output_file.parent, exist_ok=True)
    ds_with_stats.to_netcdf(output_file, engine="netcdf4")

    logging.info(f"Dataset met statistieken geschreven: {output_file}")