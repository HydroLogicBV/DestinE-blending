import logging
log = logging.getLogger(__name__)

import os
import re
import zipfile
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from nowcast_blend.preprocess.preprocess_destine import advection_correction_backward, cdo_to_netcdf

from pysteps.downscaling import rainfarm

from cdo import Cdo
cdo = Cdo()


# Filename pattern of the HydroNet GeoTIFF exports, e.g.
#   Ecmwf.Ensemble.15day_98.0.128.228.1_2026-07-08T18h00m00s_30th Percentile_2026-07-09T00h00m00s_2026-07-09T01h00m00s.tif
# groups: model run, percentile, accumulation-window-start, accumulation-window-end
_HL_TIF_PATTERN = re.compile(
    r"_(\d{4}-\d{2}-\d{2}T\d{2}h\d{2}m\d{2}s)_"
    r"(\d+)th Percentile_"
    r"(\d{4}-\d{2}-\d{2}T\d{2}h\d{2}m\d{2}s)_"
    r"(\d{4}-\d{2}-\d{2}T\d{2}h\d{2}m\d{2}s)\.tif$"
)


def _hl_parse_time(s):
    return pd.to_datetime(s, format="%Y-%m-%dT%Hh%Mm%Ss")


def _validate_ifs_HL_data(precip, source_file, model_run=None, steps=None):
    """Reject an all-nodata HydroNet forecast.

    The API answers a request for a window its model run cannot reach (e.g. a historical
    ``rundate`` against ``ModelRun: "Last"``) with correctly-shaped grids that are entirely
    nodata. Filling those with zeros would feed the blend a silently dry IFS forecast, so
    fail here instead.
    """
    if not np.isnan(precip).all():
        return

    detail = ""
    if model_run is not None and steps is not None and len(steps):
        detail = (
            f" The forecast was issued at {model_run} but data was requested for "
            f"{steps[0]} - {steps[-1]}; a model run cannot cover a window that precedes it."
        )
    raise ValueError(
        f"All values in the HydroNet IFS forecast ({source_file}) are nodata.{detail} "
        f"Request a window the latest model run actually covers: run with "
        f"settings.rt=true, or set rundate to a recent date. Delete the cached "
        f".zip/.nc for this date afterwards, since the filenames do not encode the window."
    )


def pre_process_ifs_HL_data(ifs_zip_HL, ifs_file_HL_nc):
    """Convert the HydroNet ECMWF 15-day ensemble zip (GeoTIFFs) to a single netCDF.

    The zip contains one GeoTIFF per (percentile, hourly-accumulation-step). This
    unpacks them and stacks them into a dataset with the same structure as the
    grib-derived IFS files (``tp`` with dims ``(number, step, latitude, longitude)``,
    latitude descending), where ``number`` holds the requested percentiles (30/50/90).

    Parameters
    ----------
    ifs_zip_HL : str
        Path to the downloaded zip of GeoTIFFs.
    ifs_file_HL_nc : str
        Output path for the combined netCDF.

    Returns
    -------
    xarray.Dataset
        The combined dataset (also written to ``ifs_file_HL_nc``).
    """
    import rasterio

    extract_dir = ifs_zip_HL[:-4] if ifs_zip_HL.endswith(".zip") else ifs_zip_HL + "_extracted"
    log.info(f"Extracting {ifs_zip_HL} -> {extract_dir}")
    with zipfile.ZipFile(ifs_zip_HL) as zf:
        zf.extractall(extract_dir)

    # gather all GeoTIFFs (recursively; skip Synology '@eaDir' sidecar folders)
    tif_paths = []
    for root, _, files in os.walk(extract_dir):
        if "@eaDir" in root:
            continue
        for name in files:
            if name.endswith(".tif"):
                tif_paths.append(os.path.join(root, name))
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFF files found in {ifs_zip_HL}")

    records = []          # (percentile, valid_time, array)
    transform = shape = None
    model_runs = set()
    for path in tif_paths:
        match = _HL_TIF_PATTERN.search(os.path.basename(path))
        if match is None:
            log.warning(f"Skipping unrecognised GeoTIFF name: {os.path.basename(path)}")
            continue
        model_runs.add(match.group(1))
        percentile = int(match.group(2))
        valid_time = _hl_parse_time(match.group(4))  # end of the accumulation window
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            if transform is None:
                transform, shape = src.transform, src.shape
        records.append((percentile, valid_time, arr))
    if not records:
        raise ValueError(f"No GeoTIFF in {ifs_zip_HL} matched the expected HydroNet naming")

    height, width = shape
    # pixel-centre coordinates; transform.e is negative so latitude is descending
    longitude = transform.c + (np.arange(width) + 0.5) * transform.a
    latitude = transform.f + (np.arange(height) + 0.5) * transform.e

    percentiles = sorted({r[0] for r in records})
    times = sorted({r[1] for r in records})

    data = np.full((len(percentiles), len(times), height, width), np.nan, dtype="float32")
    for percentile, valid_time, arr in records:
        data[percentiles.index(percentile), times.index(valid_time)] = arr

    times_index = pd.DatetimeIndex(times)
    if len(model_runs) > 1:
        log.warning(f"GeoTIFFs mix several model runs: {sorted(model_runs)}")
    model_run = _hl_parse_time(sorted(model_runs)[0])

    

    ds = xr.Dataset(
        {"tp": (("number", "step", "latitude", "longitude"), data)},
        coords={
            "number": ("number", percentiles),
            "step": ("step", times_index),
            "latitude": ("latitude", latitude),
            "longitude": ("longitude", longitude),
            "time": model_run,  # forecast init, i.e. the model run the export came from
            "valid_time": ("step", times_index),
        },
    )
    ds["tp"].attrs = {"units": "mm", "long_name": "Accumulated precipitation (hourly)"}
    ds["number"].attrs = {"long_name": "percentile", "units": "%"}
    ds.attrs["model_run"] = str(model_run)

    os.makedirs(os.path.dirname(ifs_file_HL_nc), exist_ok=True)
    ds.to_netcdf(ifs_file_HL_nc)

    print(ds['tp'].values)
    
    _validate_ifs_HL_data(ds['tp'].values, ifs_zip_HL, model_run=model_run, steps=times_index)

    log.info(
        f"Wrote combined IFS-HL netCDF ({len(percentiles)} percentiles x "
        f"{len(times)} steps) to {ifs_file_HL_nc}"
    )
    return ds


# The DestinE path downscales its 0.05 deg grid by 4, i.e. to ~0.0125 deg. The HydroNet
# grid is coarser (0.2 deg), so derive the rainfarm factor from the native spacing
# instead of hardcoding it.
_TARGET_RES_DEG = 0.0125


def _ifs_ds_factor(ds):
    native_res = float(np.abs(np.diff(ds.latitude.values)).mean())
    return max(int(round(native_res / _TARGET_RES_DEG)), 1)


def _ifs_slice_to_blend(ifs_nlgrid, R_xr, timestep_interval, timesteps, source_file):
    """Slice a regridded IFS dataset to the lead times the blending expects."""
    time_slice = slice(
        pd.to_datetime(R_xr['time'][-1].values) + timedelta(minutes=5),
        pd.to_datetime(R_xr['time'][-1].values)
        + timedelta(minutes=timestep_interval * timesteps)
        + timedelta(minutes=5),
    )
    IFS_nlgrid_blend = ifs_nlgrid.sel(time=time_slice)
    log.info(f'IFS_nlgrid_blend time range: {time_slice}')

    len_nwp = len(IFS_nlgrid_blend['time'])
    assert len_nwp == (timesteps + 1), (
        f'Not the correct length timesteps in IFS file ({source_file}), length is '
        f'currently: {len_nwp} while it should be {(timesteps + 1)}'
    )
    return IFS_nlgrid_blend


def pre_process_ifs_data(ifs_file_HL_nc, ifs_file_preprocessed, cfg, date, timestep_interval, timesteps, radar_path, R_xr):
    """Downscale, advection-correct and regrid the HydroNet IFS forecast onto the KNMI radar grid.

    Takes the combined netCDF written by ``pre_process_ifs_HL_data`` (``tp`` with dims
    ``(number, step, latitude, longitude)``, hourly accumulations in mm, ``number`` holding
    the 30/50/90 percentiles) and produces the blend-ready dataset on the radar grid.

    Unlike the grib/MARS source this replaces, the HydroNet data needs no unit conversion
    (already mm, not m), no de-accumulation (already hourly increments, not cumulative) and
    no percentile selection (the API returns the three percentiles directly).
    """
    if os.path.exists(ifs_file_preprocessed):
        if cfg.settings.verbose:
            log.info(f"pre-processed IFS file found: {ifs_file_preprocessed}")
        return _ifs_slice_to_blend(
            xr.open_dataset(ifs_file_preprocessed), R_xr, timestep_interval, timesteps,
            ifs_file_preprocessed,
        )

    if cfg.settings.verbose:
        log.info(f"preprocessing: {ifs_file_HL_nc}")

    IFS_data_raw = xr.open_dataset(ifs_file_HL_nc)
    precipitation = IFS_data_raw['tp']  # (number, step, latitude, longitude), mm/h
    # also catches a cached all-nodata .nc left behind by an earlier bad request
    _validate_ifs_HL_data(
        precipitation.values, ifs_file_HL_nc,
        model_run=IFS_data_raw.attrs.get("model_run"),
        steps=pd.DatetimeIndex(IFS_data_raw.step.values),
    )
    IFS_data_raw['tp'].attrs = {'long_name': 'Accumulated precipitation', 'units': 'mm/h', 'param': '193.1.0'}

    # DOWNSCALE IFS DATA WITH RAINFARM TO RADAR RESOLUTION
    ds_factor = _ifs_ds_factor(IFS_data_raw)
    n_ens, n_steps, height, width = precipitation.shape
    target_shape = (height * ds_factor, width * ds_factor)
    log.info(f"downscaling IFS {height}x{width} by {ds_factor} -> {target_shape[0]}x{target_shape[1]}")

    IFS_data_radar_scale = np.zeros((n_ens, n_steps, *target_shape))
    for i in range(n_ens):
        for j in range(n_steps):
            # scattered nodata means out-of-domain cells, which carry no rain; whole-field
            # nodata is caught by _validate_ifs_HL_data above
            field = np.nan_to_num(precipitation[i, j].values, nan=0.0)
            if np.all(field == 0) or np.std(field) == 0:
                continue  # rainfarm fails on fields without variability; leave as zeros
            down = rainfarm.downscale(field, ds_factor=ds_factor, kernel_type='gaussian')
            # rainfarm can emit values a hair below zero, which turn into NaN in the log()
            # taken by the advection correction below.
            IFS_data_radar_scale[i, j] = np.clip(down, 0, None)

    # BACKWARD ADVECTION CORRECTION: interpolate the hourly fields to timestep_interval
    substeps = 60 // timestep_interval
    all_steps = [IFS_data_radar_scale[:, 0:1]]  # keep first slice as (n_ens, 1, ny, nx)
    for j in range(n_steps - 1):
        in_between = np.zeros((n_ens, substeps, *target_shape))
        for i in range(n_ens):
            in_between[i] = advection_correction_backward(
                IFS_data_radar_scale[i][j : j + 2], T=60, t=timestep_interval
            )
        all_steps.append(in_between)
    IFS_nlgrid_hres_advected = np.concatenate(all_steps, axis=1)

    # Write to netcdf so cdo can use the data
    ifs_file_advected = ifs_file_preprocessed.replace("hres_interp_nlgrid", "hres_advect_xr")
    if ifs_file_advected == ifs_file_preprocessed:
        ifs_file_advected = ifs_file_preprocessed.replace(".nc", "_hres_advect_xr.nc")

    log.info("starting: cdo_to_netcdf:")
    cdo_to_netcdf(
        date=date,
        destinE_data=IFS_data_raw,
        destinE_data_cut=precipitation,
        numerical_data=IFS_nlgrid_hres_advected,
        destineE_datafolder=os.path.dirname(ifs_file_preprocessed),
        filename=ifs_file_advected,
        freq=f"{timestep_interval}min",
        historical_destine=False,
    )

    # REGRID IFS DATA TO KNMI RADAR GRID
    log.debug("starting: cdo.remapnn:")
    if knmi_grid_file is None:
        knmi_grid_file = os.path.join(radar_path, "knmi_grid.txt")
    cdo.remapnn(
        knmi_grid_file,  # target grid
        input=str(
            ifs_file_advected
        ),  # source file #KW: removing pre-processed/ from the path
        output=str(
            ifs_file_preprocessed
        ),  # output file #KW: removing pre-processed/ from the path
    )

    return _ifs_slice_to_blend(
        xr.open_dataset(ifs_file_preprocessed), R_xr, timestep_interval, timesteps,
        ifs_file_preprocessed,
    )

