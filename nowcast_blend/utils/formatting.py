import os
from datetime import datetime, date
import numpy.typing as npt
import numpy as np
import xarray as xr
import pandas as pd

from nowcast_blend.preprocess.preprocess_radar import convert_input_to_xarray_dataset

import logging

log = logging.getLogger(__name__)


def convert_npy_to_nc_file(
    path_blend,
    path_nowcast,
    metadata_blend,
    metadata_nowcast,
    path_nwp=None,
    metadata_nwp=None,
):
    log.info(f"Converting {path_blend} to a netCDF file")

    # check formats are correct:
    # TODO: Check with researchers, timestep=10 seems wrong, I fixed it by inferring
    # Which in our case results into 1800s = 30min, is this what we want?
    #Comment by Joep: I just ran into this issue and indeed 10 is wrong, should be 1800. even cleaner to infer it like you did. nice fix
    timestamps = pd.to_datetime(metadata_blend["timestamps"])
    startdate = timestamps[0]
    startdate_dt = pd.to_datetime(startdate).to_pydatetime()
    if len(timestamps) > 1:
        timestep = int((timestamps[1] - timestamps[0]).total_seconds())
    else:
        timestep = 1800

    blended_forecast = convert_input_to_xarray_dataset(
        precip=np.load(path_blend),
        quality=None,
        metadata=metadata_blend,
        startdate=startdate_dt,  # metadata_blend["timestamps"][0],
        timestep=timestep,
    )
    if len(timestamps) == blended_forecast.sizes["time"]:
        blended_forecast = blended_forecast.assign_coords(time=timestamps)
    # radar_nowcast = convert_input_to_xarray_dataset(
    #    precip=np.load(path_nowcast),
    #    quality=None,
    #    metadata=metadata_nowcast,
    #    startdate=metadata_nowcast["timestamps"][0],
    #    timestep=10,
    # )

    # radar_nowcast.precip_intensity.attrs["transform"] = "No"
    blended_forecast.precip_intensity.attrs["transform"] = "No"

    blended_forecast.to_netcdf(path_blend[:-3] + "nc")
    # radar_nowcast.to_netcdf(path_nowcast[:-3] + 'nc')

    # if path_nwp!= None:
    #    nwp_forecast = convert_input_to_xarray_dataset(
    #        precip=np.load(path_nwp),
    #        quality=None,
    #        metadata=metadata_nwp,
    #        startdate=metadata_nwp["timestamps"][0],
    #        timestep=10,
    #    )
    #    nwp_forecast.precip_intensity.attrs["transform"] = "No"
    #    nwp_forecast.to_netcdf(blended_forecast[:-3] + '.nc')
