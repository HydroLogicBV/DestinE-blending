from datetime import datetime, timedelta
import copy
import os
import re
import tempfile
from polytope.api import Client

from nowcast_blend.utils.logging import configure_polytope_logging

import logging

log = logging.getLogger(__name__)


def summarize_download_error(error):
    message = str(error)
    if "Date is too old" in message:
        return "Date is too old for the available DestinE datasource."
    if "user not authorized" in message:
        return "Credentials are valid, but this user is not authorized for the requested DestinE datasource."
    if "No matching datasource found" in message:
        return "No matching DestinE datasource found for this request."
    return error.__class__.__name__


configure_polytope_logging()


def create_polytope_client(log_level):
    client_kwargs = {
        "address": "polytope.lumi.apps.dte.destination-earth.eu",
        "log_level": log_level,
        "quiet": True,
    }
    user_key = os.environ.get("POLYTOPE_USER_KEY")
    if user_key:
        client_kwargs["user_key"] = user_key
    return Client(**client_kwargs)


def check_destine_available(date, param):
    configure_polytope_logging()
    time_range_acc = "/".join(f"{i}-{i+1}" for i in range(1))
    date_str = date.strftime("%Y%m%d")
    request = {
        "class": "d1",
        "expver": "0001",
        "grid": "0.05/0.05",
        "stream": "oper",
        "dataset": "extremes-dt",
        "date": date_str,
        "time": "0000",
        "type": "fc",
        "levtype": "sfc",
        "step": time_range_acc,
        "param": str(param),
    }
    log.info(f"Checking if data exists for date == {date}")
    client = create_polytope_client(log_level="WARNING")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            client.retrieve(
                "destination-earth",
                request,
                output_file=os.path.join(tmp_dir, "tmp_destine.grib"),
            )
        return True
    except Exception as e:
        log.warning(f"Data not available for {date}: {e}")
        return False


def run_download_destine(date, cfg, dirs):
    """Pipeline to download destinE data."""
    date_str_day = date.strftime("%Y%m%d")

    destine_file_original = (
        dirs.destine / f"DestinE_ExtremesDT_{date_str_day}_{cfg.settings.param}.grib"
    )
    destine_date = date_str_day

    if destine_file_original.exists():
        log.info(f"Existing original DestinE file found: {destine_file_original}")
    else:
        log.info("Downloading DestinE data")
        destine_file_original, destine_date = download_destine(
            date=date,
            historical=cfg.settings.historical_destine,
            destine_file_original=destine_file_original,
            param=cfg.settings.param,
        )
    return destine_file_original, destine_date


def download_destine(date, historical, destine_file_original, param):
    configure_polytope_logging()

    # Make sure the whole timerange is downloaded
    # time_range = "/".join(str(i) for i in range(97))
    # time_range_acc = "/".join(f"{i}-{i+1}" for i in range(24)) #KW: 24 hours is not enough when we have eg nowcast at 1pm for 12 hours
    time_range_acc = "/".join(f"{i}-{i+1}" for i in range(73))

    # KW: this isn't needed because I've already passed todays date as an argument to the function.
    # Use for operational download
    # date = datetime.now() - timedelta(days=1)

    # Uncomment for historical download
    # dates = np.arange(20250722, 20250732, 1)
    # for date_numb in dates:

    # date = datetime.strptime(str(date_numb),'%Y%m%d')

    date_str = date.strftime("%Y%m%d")
    yr = date.year
    mnth = date.month

    # Make sure local folder exists
    # destine_path_yearmonth = destine_path + '{}/{}/'.format(yr,str(mnth).zfill(2)) # KW: added zfill to make sure the month has 2 digits to match the format in other functions
    # for folder in [destine_path_yearmonth]:
    #    if not os.path.exists(folder):
    #        os.makedirs(folder)

    if historical == False:
        log.info(
            f"Downloading new forecast for {date_str} from DestinE Extremes DT when historical == False -- from polytope API"
        )
        client = create_polytope_client(log_level="INFO")
        configure_polytope_logging()

        # Optional cleanup: revoke previous requests
        try:
            client.revoke("all")
        except Exception as e:
            log.warning("Could not revoke previous Polytope requests: %s", e)
        request = {
            "class": "d1",
            "expver": "0001",
            "grid": "0.05/0.05",
            "stream": "oper",
            "dataset": "extremes-dt",
            "date": date_str,
            "time": "0000",
            "type": "fc",
            "levtype": "sfc",
            "step": time_range_acc,
            "param": str(param),
            "area": "56.4/-1/48.4/11.87",
        }
    else:
        # from ecmwfapi import ECMWFDataServer ## original script but didn't work
        # client = ECMWFDataServer()

        from ecmwfapi import ECMWFService

        server = ECMWFService("mars")

        # KW: why two requests here for to get extremesDT from ECMWF? Doesn't the second overwrite the first?
        request = {
            "area": "80/-20/20/30",
            "class": "rd",
            "dataset": "research",
            "date": "2023-10-11/to/2023-10-29",
            "expver": "i4ql",
            "grid": "0.05/0.05",
            "levtype": "sfc",
            "param": "tprate",
            "step": "0/1/2/3/4",
            "stream": "oper",
            "target": "output.grib",
            "time": "00:00:00",
            "type": "fc",
        }
        request = {
            "area": "80/-20/20/30",
            "class": "od",
            "date": "2023-10-01",
            "expver": 1,
            "levtype": "sfc",
            "number": "1/2/3",
            "param": str(param),
            "step": "0/1/2/3/4/5/6/7/8/9/10/11/12",
            "stream": "enfo",
            "time": "00:00:00",
            "type": "pf",
            "target": "output.grib",
        }
    # if 'feature' in request:
    #     extention = '.covjson'
    # else:
    #     extention = '.grib'

    #    The data will be saved in the current working directory
    # destine_file_date  = destine_path_yearmonth + f'DestinE_ExtremesDT_{date_str}_{param}{extention}'
    # destine_file_date_regrid_nc = destine_path_yearmonth + f'DestinE_ExtremesDT_{date_str}_{param}_regrid_nl.nc'

    # local_file_today = local_folder_today + 'DestinE_ExtremesDT_20231101_218.228-219.228-228.128.grib'

    # KW: original line was "files = client...". This fails when the extremesDT data isn't available yet
    # files = client.retrieve("destination-earth", request, output_file= local_file_today)
    req = copy.deepcopy(request)
    try:
        log.info(f"Trying to download data for {destine_file_original}...")
        files = client.retrieve(
            "destination-earth", request, output_file=destine_file_original
        )
        log.info(f"Success for {destine_file_original}")
        destine_date = request["date"]
        # return files, destine_file_original

    except Exception as e:
        reason = summarize_download_error(e)
        log.warning("DestinE download failed for %s: %s", request["date"], reason)
        if "Date is too old" in str(e) or "user not authorized" in str(e):
            raise RuntimeError(
                f"DestinE download failed for {request['date']}: {reason}"
            ) from e

        # fallback to previous day
        prev_date = (
            datetime.strptime(request["date"], "%Y%m%d") - timedelta(days=1)
        ).strftime("%Y%m%d")
        req["date"] = prev_date

        # fix lead times
        # req["step"] = make_time_range(48) #KW: 24 makes problems
        log.info(f"Trying previous day: {prev_date}, step == {req['step']}")
        # req["step"] = "/".join(f"{i}-{i+1}" for i in range(48))
        # log.info(f"Trying previous day: {prev_date}, step == {req['step']}")
        # log.info(f"steps = {req['step']}")

        log.info(f"Trying previous day: {prev_date}")
        destine_file_original = destine_file_original.with_name(
            re.sub(r"\d{8}", prev_date, destine_file_original.name)
        )
        if not os.path.exists(destine_file_original):
            log.info(
                f"prev_date file does not exist either so we download it: {destine_file_original}"
            )
            files = client.retrieve(
                "destination-earth", req, output_file=destine_file_original
            )  # filename is wrong
        else:
            log.info(f"prev_date file already exists so we just use that")
        destine_date = prev_date

    # server.execute(request, local_file_today)

    return destine_file_original, destine_date
