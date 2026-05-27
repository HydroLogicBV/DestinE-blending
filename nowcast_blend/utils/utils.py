from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import logging

log = logging.getLogger(__name__)


def round_to_5min(dt):
    minutes = dt.minute
    rounded = int(round(minutes / 5.0) * 5)
    diff = rounded - minutes
    return (dt + timedelta(minutes=diff)).replace(second=0, microsecond=0)


def floor_to_30min(dt):
    return dt.replace(minute=(dt.minute // 30) * 30, second=0, microsecond=0)


def ensure_destine_time_dim(ds):
    """
    sometimes the destine files have time dim and sometimes not?
    """
    # Case 1: already good
    if "time" in ds.dims:
        return ds.sortby("time")
    # Case 2: forecast structure (step + valid_time)
    if "valid_time" in ds.coords and "step" in ds.dims:
        ds = ds.assign_coords(time=ds["valid_time"])
        ds = ds.swap_dims({"step": "time"})
        return ds.sortby("time")
    # Case 3: valid_time already a dimension
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
        return ds.sortby("time")
    raise ValueError("Dataset has no usable time coordinate (time or valid_time)")


def validate_destine_time_range(destinE_nlgrid, R_xr, cfg):
    """Check that data covers the radar-based forecast window for blending."""
    expected_times = pd.date_range(
        start=pd.to_datetime(R_xr["time"][-1].values) + timedelta(minutes=5),
        periods=int(cfg.settings.timesteps) + 1,
        freq=f"{int(cfg.settings.timestep_interval)}min",
    )
    expected_start = expected_times[0]
    expected_end = expected_times[-1]
    actual_times = pd.to_datetime(destinE_nlgrid["time"].values)
    actual_start = actual_times.min()
    actual_end = actual_times.max()
    log.debug(f"Expected range: {expected_start} - {expected_end}")
    log.debug(f"Actual range:   {actual_start} - {actual_end}")
    if actual_start > expected_start:
        raise ValueError(f"Data starts too late: {actual_start} > {expected_start}")
    if actual_end < expected_end:
        raise ValueError(f"Data ends too early: {actual_end} < {expected_end}")
    log.info("Time range valid.")


def validate_destine_file(destinE_nlgrid, R_xr, cfg):
    """Return destine file sliced to expected timesteps. Raise if any timestep is missing."""
    # TODO: can just be used to slice instead of redoing validation done in validate_destine_time_range
    log.info(f"Validating and slicing destinE file for expected timesteps")
    # Define expected time range
    expected_times = pd.date_range(
        start=pd.to_datetime(R_xr["time"][-1].values) + timedelta(minutes=5),
        periods=int(cfg.settings.timesteps) + 1,
        freq=f"{int(cfg.settings.timestep_interval)}min",
    )
    log.debug(f"expected times = {expected_times}")
    # Slice dataset
    time_slice = slice(expected_times[0], expected_times[-1])
    destinE_nlgrid = ensure_destine_time_dim(
        destinE_nlgrid
    )  # Ensure time dimension exists and is sorted
    destinE_nlgrid_sel = destinE_nlgrid.sel(time=time_slice)

    actual_times = pd.to_datetime(destinE_nlgrid_sel["time"].values)
    log.debug(f"actual times = {actual_times}")

    if len(actual_times) != len(expected_times):
        raise ValueError(
            f"Wrong number of timesteps: got {len(actual_times)}, expected {len(expected_times)}"
        )
    if not np.all(actual_times == expected_times):
        raise ValueError(
            f"Timestamps mismatch.\nExpected: {expected_times}\nGot: {actual_times}"
        )
    return destinE_nlgrid_sel


def closest_ecmwf_available(dt: datetime) -> str:
    """
    Returns the latest ECMWF cycle available at the given datetime.

    ECMWF init times: 00, 06, 12, 18 UTC
    Availability: ~7 hours after init
    """
    # ECMWF init hours
    init_hours = [0, 6, 12, 18]

    # Availability delay in hours
    availability_delay = 7

    # Adjust datetime back by availability delay
    dt_adjusted = dt - timedelta(hours=availability_delay)

    # Find all init hours <= adjusted hour
    available_inits = [h for h in init_hours if h <= dt_adjusted.hour]

    if available_inits:
        latest_init = max(available_inits)
        dt_init = dt.replace(hour=latest_init, minute=0, second=0, microsecond=0)
    else:
        # No available run today → use previous day's 18Z
        latest_init = 18
        dt_init = (dt - timedelta(days=1)).replace(
            hour=latest_init, minute=0, second=0, microsecond=0
        )

    return dt_init.strftime("%Y%m%d%H")
