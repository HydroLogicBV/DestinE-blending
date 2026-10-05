from datetime import datetime, timedelta, timezone
import time
from zipfile import Path
import os
import  xarray as xr
import numpy as np

import pysteps
from pysteps.utils import transformation

from nowcast_blend.utils.utils import floor_to_30min, closest_ecmwf_available
from nowcast_blend.utils.logging import set_log_format
from nowcast_blend.utils.formatting import convert_npy_to_nc_file
from nowcast_blend.utils.paths import build_dirs
from nowcast_blend.download.download_radar import run_download_radar
from nowcast_blend.download.download_destine import run_download_destine
from nowcast_blend.download.refresh_token_HL import HLAuthClient
from nowcast_blend.download.load_ecmwf_HL import download_ecmwf_15day_HL
from nowcast_blend.preprocess.preprocess_radar import load_and_preprocess_radar
from nowcast_blend.preprocess.preprocess_destine import load_and_preprocess_destine
from nowcast_blend.preprocess.preprocess_ifs import pre_process_ifs_data, pre_process_ifs_HL_data
from nowcast_blend.nowcast.dgmr_orchestration import load_or_generate_dgmr_ensemble
from nowcast_blend.postprocess.ensemble_probabilities_destine import ensemble_probabilities_destine
from nowcast_blend.machine_learning.machine_learning import (
    run_machine_learning,
    build_machine_learning_custom_weights,
)
from nowcast_blend.blending.blending import blending_function
import hydra
from omegaconf import DictConfig
import logging
import warnings

log = logging.getLogger(__name__)


# TODO: Is this expected?
# Suppress: ypixelsize does not match y1, y2 and array shape, using ypixelsize for pixel size
warnings.filterwarnings(
    "ignore", category=UserWarning, module="nowcast_blend.preprocess.preprocess_radar"
)

def run_pipeline(cfg: DictConfig) -> None:
    logging.getLogger("nowcast_blend").setLevel(str(cfg.runtime.log_level).upper())

    log.debug(f"Using real time date: {cfg.settings.rt}")
    if cfg.settings.rt:
        date_orig = datetime.now(timezone.utc).replace(tzinfo=None)
        log.debug(f"UTC pipeline date: {date_orig}")
        date = floor_to_30min(date_orig)
    else:
        date = datetime(
            int(cfg.rundate.year),
            int(cfg.rundate.month),
            int(cfg.rundate.day),
            int(cfg.rundate.hour),
            int(cfg.rundate.minute),
        )
    date_str = date.strftime("%Y%m%d%H")
    log.info(f"Running blending code for {date}")
    start_time = time.time()

    dirs = build_dirs(cfg, date)
    dirs.ensure_runtime_dirs()

    file_prefix = (
        dirs.blended
        / f"Blended_forecast_{date_str}_step_min_{cfg.settings.timestep_interval}_len_{cfg.settings.timesteps}"
    )
    if cfg.settings.pysteps_nowcast:
        file_name = "_pysteps_nowcast"
        weights_suffix = "_weights"
    else:
        multi_extention = "_IFS" if cfg.settings.model_used != "ExtremesDT" else ""
        custom_weights_extention = (
            "_optimised_weights" if cfg.settings.custom_weights else ""
        )
        noise_extention = "_noise" if cfg.settings.noise else ""
        probmatching_extention = "_probmatch" if cfg.settings.probmatching else ""
        custom_extention = (
            f"_{cfg.settings.custom_extention}" if cfg.settings.custom_extention else ""
        )
        file_name = f"_ens_dgmr_{cfg.settings.n_ens_members_dgmr}_ens_{cfg.settings.n_ens_members}{custom_extention}"
        weights_suffix = f"{multi_extention}{noise_extention}{probmatching_extention}{custom_weights_extention}_weights"
    blended_file = file_prefix.with_name(f"{file_prefix.name}{file_name}.npy")
    blended_file_weights = file_prefix.with_name(
        f"{file_prefix.name}{file_name}{weights_suffix}.npy"
    )

    log.info("--------------------------------------------------------------------")
    log.info("1. Radar data - download and preprocess")
    log.info("--------------------------------------------------------------------")
    radar_files = run_download_radar(
        date=date,
        gauge_adjusted=cfg.settings.gauge_adjusted,
        input_dir=str(dirs.radar),
    )
    metadata_radar, R_xr = load_and_preprocess_radar(radar_files)

    log.info("--------------------------------------------------------------------")
    log.info("2a. DestinE data - download and preprocess")
    log.info("--------------------------------------------------------------------")
    Extremes_DT_downloaded = False
    if cfg.settings.model_used != 'IFS':
        try:
            destine_file, destine_date = run_download_destine(date, cfg, dirs)
            destine_nlgrid_blend = load_and_preprocess_destine(
                destine_file, destine_date, cfg, dirs, R_xr
            )
            Extremes_DT_downloaded = True
        except Exception as e:
            if cfg.settings.model_used == 'ExtremesDT':
                log.exception(f"DestinE data not available for date == {date} ({e}), aborting script")
                raise
            log.warning(f"DestinE data not available for date == {date} ({e}), continuing with IFS only")




    if cfg.settings.model_used != 'ExtremesDT':
        log.info("--------------------------------------------------------------------")
        log.info("2b. IFS data - download if needed and preprocess:")
        log.info("--------------------------------------------------------------------")
        ifs_init_time = closest_ecmwf_available(date)

        # The new IFS runs only contain data from 01:00 UTC onwards. For runs between
        # 00:00 and 01:00 UTC, today's IFS file has no data covering the run window yet,
        # so use the previous day's IFS run (a 15-day forecast that still covers today).
        if date.hour < 1:
            ifs_date = date - timedelta(days=1)
            log.info(
                f"Run hour {date.hour:02d} UTC is before 01 UTC; the new IFS runs only "
                f"contain data from 01 UTC onwards, so using the IFS run of {ifs_date:%Y-%m-%d}."
            )
        else:
            ifs_date = date
        ifs_date_str_day = ifs_date.strftime("%Y%m%d")
        ifs_dir_hl = build_dirs(cfg, ifs_date).ifs
        ifs_dir_hl.mkdir(parents=True, exist_ok=True)

        date_str = date.strftime('%Y%m%d%H')
        # ifs HydroNet (15-day ensemble, GeoTIFF) paths:
        ifs_file_preprocessed = dirs.ifs / f'IFS_{date_str}_init{ifs_init_time}_{cfg.settings.param}_hres_interp_nlgrid_{cfg.settings.timestep_interval}_{cfg.settings.timesteps}.nc'
        ifs_zip_HL = ifs_dir_hl / f"IFS_HL_15day_{ifs_date_str_day}_{cfg.settings.param}.zip"
        ifs_file_HL_nc = ifs_dir_hl / f"IFS_HL_15day_{ifs_date_str_day}_{cfg.settings.param}.nc"

        # HydroNet 15-day ensemble (30/50/90 percentiles, GeoTIFF) download + preprocess
        secret = os.environ.get("HYDRONET_API_TOKEN")
        client = os.environ.get("HYDRONET_API_CLIENT")
        auth_client = HLAuthClient(client_id=client, client_secret=secret)
        http_header = auth_client.execute()

        if not os.path.exists(ifs_zip_HL):
            log.info(f"downloading ifs HydroNet zip: {ifs_zip_HL}")
            download_ecmwf_15day_HL(http_header, ifs_date, output_zip=ifs_zip_HL)
        else:
            log.info(f"ifs HydroNet zip already downloaded: {ifs_zip_HL}")

        if not os.path.exists(ifs_file_HL_nc):
            log.info(f"preprocessing ifs HydroNet GeoTIFFs -> {ifs_file_HL_nc}")
            pre_process_ifs_HL_data(ifs_zip_HL, ifs_file_HL_nc)

        else:
            log.info(f"ifs HydroNet netcdf already exists: {ifs_file_HL_nc}")

        # downscale + advection-correct + regrid onto the KNMI radar grid
        IFS_nlgrid_blend = pre_process_ifs_data(
            ifs_file_HL_nc, ifs_file_preprocessed, cfg, date,
            cfg.settings.timestep_interval, cfg.settings.timesteps, str(dirs.radar), R_xr,
            knmi_grid_file=str(dirs.resources / "knmi_grid.txt"))

    log.info("--------------------------------------------------------------------")
    log.info("3. DGMR...    ")
    log.info("--------------------------------------------------------------------")
    # TODO no option to change the timlength yet for DGMR (now only works if it accounts to 6 hours)
    dgmr_file = (
        dirs.dgmr
        / f"DGMR_{date_str}_step_min_{cfg.settings.timestep_interval}_len_{cfg.settings.timesteps}_ens_{cfg.settings.n_ens_members_dgmr}.npy"
    )
    DGMR_det_long = load_or_generate_dgmr_ensemble(
        R_xr.precip_intensity.values, dgmr_file, cfg
    )

    # Select only the relevant time steps from the DGMR output
    if int(cfg.settings.timestep_interval) != 5:
        step = int(cfg.settings.timestep_interval) // 5
        DGMR_det = DGMR_det_long[:, ::step]
        del DGMR_det_long
        import gc
        gc.collect()
    else:
        DGMR_det = DGMR_det_long

    # check we have the right number of times
    assert len(DGMR_det[0]) == (
        int(cfg.settings.timesteps) + 1
    ), f"length of DGMR output is not the same as the timesteps value, len is {len(DGMR_det[0])}!"
    # TODO: do we need this?
    # not used currently, but here to check if times are correct
    # new_times_DGMR = pd.date_range(R_xr['time'][-1].values, pd.to_datetime(R_xr['time'][-1].values) + timedelta(minutes = 5 * int(cfg.settings.timesteps) * (int(cfg.settings.timestep_interval) / 5) ), freq=f"{cfg.settings.timestep_interval}min")

    log.info("--------------------------------------------------------------------")
    log.info("4. Organise metadata and data...    ")
    log.info("--------------------------------------------------------------------")
    if cfg.settings.model_used == 'ExtremesDT':
        # organise the metadata
        destine_nlgrid_blend_metadata = metadata_radar
        destine_nlgrid_blend_metadata['timestamps'] = destine_nlgrid_blend.time.values
        destine_nlgrid_blend_metadata['institution'] = destine_nlgrid_blend.institution
        destine_nlgrid_blend_metadata['unit'] = 'mm/h'
        destine_nlgrid_blend_metadata['threshold'] = float(0.1)
        metadata_radar['transform'] = None
        metadata_DGMR = metadata_radar
        # Log-transform the data
        metadata_radar['timestamps'] = destine_nlgrid_blend_metadata['timestamps']

    if cfg.settings.model_used == 'multi-model' or cfg.settings.model_used == 'IFS':
        # organise the metadata
        IFS_nlgrid_blend_metadata = metadata_radar
        IFS_nlgrid_blend_metadata['timestamps'] = IFS_nlgrid_blend.time.values
        IFS_nlgrid_blend_metadata['institution'] = "European Centre for Medium-Range Weather Forecasts"
        IFS_nlgrid_blend_metadata['unit'] = 'mm/h'
        IFS_nlgrid_blend_metadata['threshold'] = float(0.1)
        metadata_radar['transform'] = None
        metadata_DGMR = metadata_radar
        # Log-transform the data
        metadata_radar['timestamps'] = IFS_nlgrid_blend_metadata['timestamps']

    DGMR_det_db, metadata_radar_db = transformation.dB_transform(
        DGMR_det, metadata_radar, threshold=0.1, zerovalue=-15.0
    )
    if DGMR_det_db.ndim == 3:
        DGMR_det_db = DGMR_det_db[None, :]
    converter = pysteps.utils.get_method("mm/h")
    radar_precip, metadata_radar = converter(
        R_xr.precip_intensity.values, metadata_radar
    )

    #after this block, even if multi-model is True or 'IFS', the destine_nlgrid_blend_xxx variables names are used. 
    if cfg.settings.model_used == 'multi-model':
        if Extremes_DT_downloaded == True:
            # Stack DestinE (deterministic, one member) and the three IFS percentile members
            # along a shared 'ensemble' axis, giving the (n_models, time, y, x) the blending
            # expects. DestinE lands at index 0.
            ifs_blend = IFS_nlgrid_blend.transpose('ensemble', 'time', 'y', 'x')
            destine_ens = destine_nlgrid_blend.assign_coords(ensemble=51.0).expand_dims(dim='ensemble', axis=0)

            # concat would silently outer-join a mismatched time axis into NaN-padded models
            destine_times = destine_ens['time'].values
            ifs_times = ifs_blend['time'].values
            if not np.array_equal(destine_times, ifs_times):
                raise ValueError(
                    f"DestinE and IFS time axes differ, refusing to combine them. "
                    f"DestinE: {destine_times[0]} - {destine_times[-1]} (n={len(destine_times)}); "
                    f"IFS: {ifs_times[0]} - {ifs_times[-1]} (n={len(ifs_times)})"
                )

            IFS_ExtremesDT_blend = xr.concat([destine_ens, ifs_blend], dim='ensemble', join='exact')
            log.info(
                f"multi-model NWP stack: {dict(IFS_ExtremesDT_blend.sizes)}, "
                f"ensemble labels {IFS_ExtremesDT_blend.ensemble.values} (51=DestinE, rest=IFS percentiles)"
            )
            destine_nlgrid_blend_val, destine_nlgrid_blend_metadata = converter(IFS_ExtremesDT_blend.tp.values, IFS_nlgrid_blend_metadata)
        else:
            ifs_blend = IFS_nlgrid_blend.transpose('ensemble', 'time', 'y', 'x')
            destine_nlgrid_blend_val, destine_nlgrid_blend_metadata = converter(ifs_blend.tp.values, IFS_nlgrid_blend_metadata)
            # concat would silently outer-join a mismatched time axis into NaN-padded models
            

    elif cfg.settings.model_used == 'IFS':
        destine_nlgrid_blend_val, destine_nlgrid_blend_metadata = converter(
            IFS_nlgrid_blend.transpose('ensemble', 'time', 'y', 'x').tp.values, IFS_nlgrid_blend_metadata)
    else:
        destine_nlgrid_blend_val, destine_nlgrid_blend_metadata = converter(destine_nlgrid_blend.tp.values, destine_nlgrid_blend_metadata)


    # Threshold the data
    radar_precip[radar_precip < 0.1] = 0.0
    destine_nlgrid_blend_val[destine_nlgrid_blend_val < 0.1] = 0.0

    # transform the data to dB
    transformer = pysteps.utils.get_method("dB")
    radar_precip, radar_metadata = transformer(
        radar_precip, metadata_radar, threshold=0.1
    )
    nwp_precip, nwp_metadata = transformer(
        destine_nlgrid_blend_val, destine_nlgrid_blend_metadata, threshold=0.1
    )

    # r_nwp has to be four dimentional (n_models, time, y, x).
    # If we only use one model:
    if nwp_precip.ndim == 3:
        nwp_precip = nwp_precip[None, :]

    log.info("--------------------------------------------------------------------")
    log.info("5. Determining machine learning weights if enabled... ")
    log.info("--------------------------------------------------------------------")
    custom_weights = None
    probmatching = None
    if cfg.settings.use_machine_learning_weights:
        log.info("machine learning weights enabled, running!")
        cluster_weights = run_machine_learning(
            dirs.machine_learning,
            date_str,
            destine_nlgrid_blend_val,
            DGMR_det,
        )
        custom_weights = build_machine_learning_custom_weights(cluster_weights)
        probmatching = bool(cluster_weights["use_probmatching"])


    log.info("--------------------------------------------------------------------")
    log.info("6. Do the blending...")
    log.info("--------------------------------------------------------------------")
    blending_function(
        blended_file=str(blended_file),
        blended_file_weights=str(blended_file_weights),
        config=cfg,
        radar_precip=radar_precip,
        nwp_precip=nwp_precip,
        DGMR_det_db=DGMR_det_db,
        radar_metadata=radar_metadata,
        nwp_metadata=nwp_metadata,
        custom_weights=custom_weights,
        probmatching=probmatching,
    )

    log.info("--------------------------------------------------------------------")
    log.info("6. write to netcdf...")
    log.info("--------------------------------------------------------------------")
    convert_npy_to_nc_file(
        str(blended_file), dgmr_file, destine_nlgrid_blend_metadata, metadata_DGMR
    )

    log.info("--------------------------------------------------------------------")
    log.info("7. run zware-buien post-processing...")
    log.info("--------------------------------------------------------------------")

    from pathlib import Path

    # Get the path relative to the nowcast_blend package
    config_path = str(Path(__file__).parent.parent / "configs" / "zware-buien.yaml")

    blended_file_nc = str(blended_file)[:-4] + ".nc"
    blended_file_zware_buien = Path(str(blended_file)[:-4] + "_zware_buien.nc")
    ensemble_probabilities_destine(config_path, str(blended_file_nc), blended_file_zware_buien)


    log.info(f"{(time.time() - start_time)/60} minutes")


@hydra.main(version_base=None, config_path="../configs", config_name="nowcast-blend")
def main(cfg: DictConfig) -> None:
    set_log_format()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
