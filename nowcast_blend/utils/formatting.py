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

    # capture the reference time once, before we start mutating variables/attrs
    precip_var = blended_forecast.attrs.get("precip_var", list(blended_forecast.data_vars)[0])
    ref_time = blended_forecast[precip_var].time.values[0]

    if "time" in blended_forecast.dims:
        # The fields are rates (mm/h) over `timestep` seconds, so each one adds rate * timestep/3600 mm.
        # A plain cumsum of mm/h would overestimate the amount by 3600/timestep (2x for 30 min steps).
        precip_units = blended_forecast[precip_var].attrs.get("units")
        if precip_units == "mm/h":
            step_hours = timestep / 3600.0
        elif precip_units == "mm":
            step_hours = 1.0  # already an amount per time step
        else:
            raise ValueError(f"Cannot compute cumulative precipitation from units {precip_units!r}")
        cumulative = (blended_forecast[precip_var] * step_hours).cumsum(dim="time")
        cumulative.name = f"{precip_var}_cumulative"
        cumulative.attrs = {
            "long_name": "cumulative precipitation over the forecast period",
            "units": 'mm',
            "comment": f"Cumulative sum over the time dimension of {precip_var} [{precip_units}] times the time step ({timestep} s)",
            "grid_mapping": "polar_stereographic",
            "coordinates": "lat lon forecast_reference_time",
            "threshold": blended_forecast[precip_var].attrs.get("threshold", ""),
            "zerovalue": blended_forecast[precip_var].attrs.get("zerovalue", ""),
        }
        assert (cumulative.diff(dim="time") >= 0).all(), (
            "Cumulative precipitation: found decreasing values across time. "           
        )
        blended_forecast = blended_forecast.assign({cumulative.name: cumulative})
    
    # real coordinate variable — required by ADAGUC, replaces the old string-attribute approach
    blended_forecast = blended_forecast.assign_coords(forecast_reference_time=ref_time)
    blended_forecast["forecast_reference_time"].attrs = {
    "standard_name": "forecast_reference_time",
    "long_name": "time of model initialization",
}

    blended_forecast[precip_var].attrs["coordinates"] = "lat lon forecast_reference_time"

    blended_forecast.to_netcdf(
    path_blend[:-3] + "nc",
    encoding={
        "forecast_reference_time": {"units": "seconds since 1970-01-01 00:00:00 +00:00"}
    },
)

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
