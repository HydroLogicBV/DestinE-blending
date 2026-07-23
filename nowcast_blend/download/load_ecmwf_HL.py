import os
from datetime import timedelta

import requests

import logging
log = logging.getLogger(__name__)


def download_ecmwf_15day_HL(headers, date, output_zip=None, n_days=2):
    """Download the ECMWF 15-day ensemble (30/50/90 percentiles) from the HydroNet API.

    Parameters
    ----------
    headers : dict
        HTTP header (incl. bearer token) as returned by ``WIWBAuthClient.execute()``.
    date : datetime
        Run date. The forecast window starts at this day at 00:00 UTC and spans
        ``n_days`` days.
    output_zip : str, optional
        If given, the returned zip (a bundle of GeoTIFFs) is written to this path.
    n_days : int
        Length of the forecast window in days (default 1, matching the original request).

    Returns
    -------
    bytes
        The raw response content: a zip archive containing one GeoTIFF per
        (percentile, hourly-accumulation-step).
    """
    url = "https://hnapi.hydronet.com/api/data/get"

    start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=n_days)

    payload = {
        "Readers": [
            {
                "DataSourceCode": "Ecmwf.Ensemble.15day",
                "Settings": {
                    "StructureType": "EnsembleGrid",
                    "ModelRun": "Last",
                    "StartDate": start_date.strftime("%Y%m%d%H%M%S"),
                    "EndDate": end_date.strftime("%Y%m%d%H%M%S"),
                    "VariableCodes": ["tp"],
                    "CalculationType": "Default",
                    "Interval": {
                        "Type": "Hours",
                        "Value": 1
                    },
                    "Extent": {
                        "Xll": -1,
                        "Yll": 48.4,
                        "Xur": 11.87,
                        "Yur": 56.4,
                        "SpatialReference": {
                            "Epsg": 4326
                        }
                    }
                }
            }
        ],
        "Processors": [
            {
                "ProcessorCode": "Statistics",
                "Settings": {
                    "Percentiles": [30, 50, 90],
                    "CalculationType": "Percentile"
                }
            }
        ],
        "Exporter": {
            "DataFormatCode": "geotiff",
            "SpatialReference": {
                "Epsg": 4326
            }
        }
    }

    log.info(f"Requesting ECMWF 15-day HL ensemble for {start_date} -> {end_date}")
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    if output_zip is not None:
        os.makedirs(os.path.dirname(output_zip), exist_ok=True)
        with open(output_zip, "wb") as f:
            f.write(response.content)
        log.info(f"Saved ECMWF 15-day HL zip to {output_zip}")

    return response.content  # zip archive containing GeoTIFFs
