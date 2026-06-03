import numpy as np

from datetime import datetime, timezone
import time

import pysteps
from pysteps.utils import transformation

from nowcast_blend.utils.utils import floor_to_30min
from nowcast_blend.utils.logging import set_log_format
from nowcast_blend.utils.formatting import convert_npy_to_nc_file
from nowcast_blend.utils.paths import build_dirs
from nowcast_blend.download.download_radar import run_download_radar
from nowcast_blend.download.download_destine import run_download_destine
from nowcast_blend.preprocess.preprocess_radar import load_and_preprocess_radar
from nowcast_blend.preprocess.preprocess_destine import load_and_preprocess_destine
from nowcast_blend.nowcast.dgmr_for_blending import run_dgmr_ensemble
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
        multi_extention = "_IFS" if cfg.settings.multi_model else ""
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
    destine_file, destine_date = run_download_destine(date, cfg, dirs)
    destine_nlgrid_blend = load_and_preprocess_destine(
        destine_file, destine_date, cfg, dirs, R_xr
    )

    if cfg.settings.multi_model:
        log.info("--------------------------------------------------------------------")
        log.info("2b. IFS data - download if needed and preprocess:")
        log.info("--------------------------------------------------------------------")
        log.warning("IFS download and preprocessing to be added")
        # ifs_init_time = closest_ecmwf_available(date)

    log.info("--------------------------------------------------------------------")
    log.info("3. DGMR...    ")
    log.info("--------------------------------------------------------------------")
    # TODO no option to change the timlength yet for DGMR (now only works if it accounts to 6 hours)
    dgmr_file = (
        dirs.dgmr
        / f"DGMR_{date_str}_step_min_{cfg.settings.timestep_interval}_len_{cfg.settings.timesteps}_ens_{cfg.settings.n_ens_members_dgmr}.npy"
    )
    if not dgmr_file.exists():
        log.info(f"DGMR file is missing so we have to make the nowcast: {dgmr_file}")
        # if DGMR hasn't been run, then run it and save it
        DGMR_det_long = run_dgmr_ensemble(
            R_xr.precip_intensity.values,
            ens_members=int(cfg.settings.n_ens_members_dgmr),
            forecast_length=int(
                (int(cfg.settings.timestep_interval) * int(cfg.settings.timesteps)) / 90
            ),
            max_workers=cfg.runtime.dgmr_max_workers,
            random_seed=cfg.runtime.dgmr_random_seed,
        )
        np.save(dgmr_file, DGMR_det_long, allow_pickle=True)
    else:
        # if dgmr has been run, then just import it from file
        log.info(f"Importing DGMR file: {dgmr_file}")
        DGMR_det_long = np.load(dgmr_file, allow_pickle=True)

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
    # organise the metadata
    destine_nlgrid_blend_metadata = metadata_radar
    destine_nlgrid_blend_metadata["timestamps"] = destine_nlgrid_blend.time.values
    destine_nlgrid_blend_metadata["institution"] = destine_nlgrid_blend.institution
    destine_nlgrid_blend_metadata["unit"] = "mm/h"
    destine_nlgrid_blend_metadata["threshold"] = float(0.1)
    metadata_radar["transform"] = None
    metadata_DGMR = metadata_radar
    # Log-transform the data
    metadata_radar["timestamps"] = destine_nlgrid_blend_metadata["timestamps"]
    DGMR_det_db, metadata_radar_db = transformation.dB_transform(
        DGMR_det, metadata_radar, threshold=0.1, zerovalue=-15.0
    )
    if DGMR_det_db.ndim == 3:
        DGMR_det_db = DGMR_det_db[None, :]
    converter = pysteps.utils.get_method("mm/h")
    radar_precip, metadata_radar = converter(
        R_xr.precip_intensity.values, metadata_radar
    )
    if cfg.settings.multi_model == True:
        log.warning(f"Multi_model mode not implemented")
        # destine_nlgrid_blend_val, destine_nlgrid_blend_metadata = converter(IFS_ExtremesDT_blend.tp.values, destine_nlgrid_blend_metadata)
    else:
        destine_nlgrid_blend_val, destine_nlgrid_blend_metadata = converter(
            destine_nlgrid_blend.tp.values, destine_nlgrid_blend_metadata
        )

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

    log.info(f"{(time.time() - start_time)/60} minutes")


@hydra.main(version_base=None, config_path="../configs", config_name="nowcast-blend")
def main(cfg: DictConfig) -> None:
    set_log_format()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
