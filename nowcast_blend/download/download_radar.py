import os
import requests
import shutil
from pysteps import io

from datetime import datetime, timedelta

from nowcast_blend.utils.utils import round_to_5min

import logging

log = logging.getLogger(__name__)


def get_radar_product(gauge_adjusted):
    if gauge_adjusted:
        return {
            "url": "https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m/versions/1.0/files",
            "filename_pattern": "RAD_NL25_RAC_RT_%Y%m%d%H%M",
        }
    return {
        "url": "https://api.dataplatform.knmi.nl/open-data/datasets/radar_reflectivity_composites/versions/2.0/files",
        "filename_pattern": "RAD_NL25_PCP_NA_%Y%m%d%H%M",
    }


def round_to_5min(dt):
    minutes = dt.minute
    rounded = int(round(minutes / 5.0) * 5)
    diff = rounded - minutes
    return (dt + timedelta(minutes=diff)).replace(second=0, microsecond=0)


def download_radar_knmi(gauge_adjusted, last_hour, date, input_dir, api_key=None):
    radar_product = get_radar_product(gauge_adjusted)
    url = radar_product["url"]
    filename_pattern = radar_product["filename_pattern"]
    lastfile = last_hour.strftime(f"{filename_pattern}.h5")

    if not api_key:
        api_key = os.environ.get("KNMI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "KNMI_API_KEY is required to download missing radar files. "
            "Set it in the environment."
        )

    file_list_response = requests.get(
        url,
        headers={"Authorization": api_key},
        params={"startAfterFilename": lastfile, "maxKeys": 12},
    )
    file_list_response.raise_for_status()
    file_list = file_list_response.json().get("files") or []
    log.info(
        "KNMI radar API returned files: %s",
        [file_info.get("filename") for file_info in file_list],
    )

    if len(file_list) < 4:
        raise RuntimeError(
            f"KNMI radar API returned {len(file_list)} files after {lastfile}; "
            "need at least 4 radar files for DGMR. Check the run date, product availability, "
            "and KNMI_API_KEY."
        )

    # Download the last 3 available files
    for file_info in file_list[-4:]:
        fn = file_info["filename"]
        log.info(fn)

        yr = fn[16:20]
        mnth = fn[20:22]
        day = fn[22:24]
        hour = fn[24:26]
        minute = fn[26:28]

        local_folder_today = os.path.join(input_dir, yr, mnth, day)
        os.makedirs(local_folder_today, exist_ok=True)

        local_file = os.path.join(local_folder_today, fn)

        if not os.path.exists(local_file):

            get_file_response = requests.get(
                url + "/" + fn + "/url", headers={"Authorization": api_key}
            )
            get_file_response.raise_for_status()

            download_url = get_file_response.json().get("temporaryDownloadUrl")
            if not download_url:
                raise RuntimeError(
                    f"KNMI API did not return a temporaryDownloadUrl for {fn}"
                )

            dataset_file = requests.get(download_url, stream=True)
            dataset_file.raise_for_status()

            with open(local_file, "wb") as f:
                dataset_file.raw.decode_content = True
                shutil.copyfileobj(dataset_file.raw, f)
    fns = io.find_by_date(
        date, input_dir, "%Y/%m/%d", filename_pattern, "h5", 5, num_prev_files=3
    )
    assert (
        len(fns[0]) == 4
    ), f"fns does not contain enough radar images for DGMR (needs 4, contains {len(fns[0])})"
    return fns


def run_download_radar(date, gauge_adjusted, input_dir, api_key=None):
    # inset a date and time (in utc)
    last_hour = date + timedelta(hours=-1)
    date_5min = round_to_5min(date) - timedelta(
        minutes=5
    )  # round to 5 minutes, then substract 5 minutes so that DGMR is initialised on the hour exactly
    # TODO: date_5min = round_to_5min(date) #Currently running DGMR on 5 past the hour, but including last radar image -> gives 6hours +5 minutes which is needed for blending
    last_hour_5min = round_to_5min(last_hour) - timedelta(
        minutes=5
    )  # see reason above for not using this
    fn_pattern = get_radar_product(gauge_adjusted)["filename_pattern"]

    expected_dates = [date_5min - timedelta(minutes=5 * ii) for ii in range(3, -1, -1)]
    expected_files = [
        expected_date.strftime(f"{fn_pattern}.h5") for expected_date in expected_dates
    ]
    log.info("Expected radar files: %s", expected_files)

    # check if data exists, otherwise download
    fns = None
    try:
        fns = io.find_by_date(
            date_5min, input_dir, "%Y/%m/%d", fn_pattern, "h5", 5, num_prev_files=3
        )
        assert (
            len(fns[0]) == 4
        ), f"fns does not contain enough radar images for DGMR (needs 4, contains {len(fns[0])})"
        if None in fns:
            raise AssertionError("(Part of Radar files not found.")
        if None in fns[0]:
            raise AssertionError("(Part of Radar files not found.")
        if None in fns[1]:
            raise AssertionError("(Part of Radar files not found.")
        log.info(f"Existing radar files found: {expected_files}")
    except:
        fns = download_radar_knmi(
            gauge_adjusted, last_hour_5min, date_5min, input_dir, api_key=api_key
        )

    return fns
