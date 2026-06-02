import os
import contextlib
import numpy as np
import pandas as pd
import time

import pysteps
from pysteps import blending, motion

from nowcast_blend.utils.logging import LoggerStream

import logging

log = logging.getLogger(__name__)


def run_pysteps_forecast(**kwargs):
    stdout = LoggerStream(log, prefix="pysteps: ")
    with contextlib.redirect_stdout(stdout):
        try:
            return blending.steps.forecast(**kwargs)
        finally:
            stdout.flush()


def blending_function(
    blended_file,
    blended_file_weights,
    config,
    radar_precip,
    nwp_precip,
    DGMR_det_db,
    radar_metadata,
    nwp_metadata,
    custom_weights=None,
    probmatching=None,
):
    ###############################################################################
    # For the initial time step (t=0), the NWP rainfall forecast is not that different
    # from the observed radar rainfall, but it misses some of the locations and
    # shapes of the observed rainfall fields. Therefore, the NWP rainfall forecast will
    # initially get a low weight in the blending process.
    #
    # Determine the velocity fields
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # local_folder_today_blend =  destineE_datafolder + 'blended_forecast/{}/{}/'.format(yr,mnth)
    # path_blend = local_folder_today_blend + f'/Blended_forecast_{date_str}_step_min_{timestep_interval}_len_{timesteps}_ens_dgmr_{n_ens_members_dgmr}_ens_{n_ens_members}.npy'

    # for folder in [local_folder_today_blend]:
    #    if not os.path.exists(folder):
    #        os.makedirs(folder)

    # if config['SETTINGS']['multi_model']:
    #    multi_extention = '_IFS'
    # else:
    #    multi_extention = ''

    # if custom_weights == None:
    #    custom_weights_extention = ''
    # else:
    #    custom_weights_extention = '_optimised_weights'

    # if noise:
    #    noise_extention = '_noise'
    # else:
    #    noise_extention = ''

    # if probmatching:
    #    probmatching_extention = '_probmatch'
    # else:
    #    probmatching_extention = ''

    converter = pysteps.utils.get_method("mm/h")

    if config.settings.use_machine_learning_weights:
        noise_method = "nonparametric"
    elif config.settings.noise:
        noise_method = "nonparametric"
    else:
        noise_method = None

    if config.settings.use_machine_learning_weights:
        if probmatching:
            prob_match = "cdf"
        else:
            prob_match = None
    elif config.settings.probmatching:
        prob_match = "cdf"
    else:
        prob_match = None

    random_seed = config.runtime.pysteps_random_seed
    if random_seed is not None:
        random_seed = int(random_seed)

    # if custom_extention == None:
    #    custom_extention = ''

    # if machine_learning_weights == True:
    #    custom_extention = 'machine_learning' + custom_extention

    # if pysteps_nowcast:
    #    path_blend = local_folder_today_blend + f'/Blended_forecast_{date_str}_step_min_{timestep_interval}_len_{timesteps}_pysteps_nowcast.npy'
    #    path_blend_weights = local_folder_today_blend + f'/Blended_forecast_{date_str}_step_min_{timestep_interval}_len_{timesteps}_pysteps_nowcast_weights.npy'
    # else:
    #    path_blend = local_folder_today_blend + f'/Blended_forecast_{date_str}_step_min_{timestep_interval}_len_{timesteps}_ens_dgmr_{n_ens_members_dgmr}_ens_{n_ens_members}{multi_extention}{noise_extention}{probmatching_extention}{custom_weights_extention}{custom_extention}.npy'
    #    path_blend_weights = local_folder_today_blend + f'/Blended_forecast_{date_str}_step_min_{timestep_interval}_len_{timesteps}_ens_dgmr_{n_ens_members_dgmr}_ens_{n_ens_members}{multi_extention}{noise_extention}{probmatching_extention}{custom_weights_extention}{custom_extention}_weights.npy'

    # Calculate the machine learning weights here

    if not os.path.exists(blended_file) or config.settings.re_do_blending:
        oflow_method = motion.get_method("lucaskanade")

        # First for the radar images
        velocity_radar = oflow_method(radar_precip)

        # Then for the NWP forecast
        velocity_nwp = []
        # Loop through the models
        for n_model in range(nwp_precip.shape[0]):
            # Loop through the timesteps. We need two images to construct a motion
            # field, so we can start from timestep 1. Timestep 0 will be the same
            # as timestep 1.
            _v_nwp_ = []
            for t in range(1, nwp_precip.shape[1]):
                v_nwp_ = oflow_method(nwp_precip[n_model, t - 1 : t + 1, :])
                _v_nwp_.append(v_nwp_)
                v_nwp_ = None
            # Add the velocity field at time step 1 to time step 0.
            _v_nwp_ = np.insert(_v_nwp_, 0, _v_nwp_[0], axis=0)
            velocity_nwp.append(_v_nwp_)

        velocity_nwp = np.stack(velocity_nwp)

        ################################################################################
        # The blended forecast
        # --------------------
        # timestep_interval = 5
        mask = ~np.isfinite(velocity_nwp)
        n_replaced = np.sum(mask)

        if n_replaced > 0:
            log.info(
                f"Replacing {n_replaced} non-finite values in precip_cascades with 0."
            )
        # try:
        if config.settings.pysteps_nowcast:
            precip_forecast_stacked = run_pysteps_forecast(
                precip=radar_precip[1:],
                # precip_nowcast=DGMR_det_db,
                # nowcasting_method="external_nowcast",
                mask_method=None,
                precip_models=nwp_precip,
                velocity=velocity_radar,
                velocity_models=velocity_nwp,
                timesteps=config.settings.timesteps,
                timestep=config.settings.timestep_interval,
                issuetime=pd.to_datetime(radar_metadata["timestamps"][-1]),
                n_ens_members=config.settings.n_ens_members,
                # resample_distribution=False,
                precip_thr=radar_metadata["threshold"],
                kmperpixel=radar_metadata["xpixelsize"] / 1000.0,
                # noise_stddev_adj=noise, #major difference from pysteps paper
                noise_method=noise_method,  # major difference from pysteps paper
                weights_method=config.settings.weights_method,
                custom_weights=config.settings.custom_weights,
                return_weights=config.settings.return_weights,
                probmatching_method=prob_match,  ##major difference from pysteps paper, used before: "cdf"
                seed=random_seed,
                # domain= 'spectral',
                vel_pert_method=None,
            )
        elif config.settings.use_machine_learning_weights:
            precip_forecast_stacked = run_pysteps_forecast(
                precip=radar_precip[1:],
                precip_nowcast=DGMR_det_db,
                nowcasting_method="external_nowcast",
                mask_method=None,
                precip_models=nwp_precip,
                velocity=velocity_radar,
                velocity_models=velocity_nwp,
                timesteps=config.settings.timesteps,
                timestep=config.settings.timestep_interval,
                issuetime=pd.to_datetime(radar_metadata["timestamps"][-1]),
                n_ens_members=config.settings.n_ens_members,
                fft_method="pyfftw",
                # resample_distribution=False,
                precip_thr=radar_metadata["threshold"],
                kmperpixel=radar_metadata["xpixelsize"] / 1000.0,
                # noise_stddev_adj=noise, #major difference from pysteps paper
                noise_method=noise_method,  # major difference from pysteps paper
                weights_method=config.settings.weights_method,
                custom_weights=custom_weights,
                return_weights=config.settings.return_weights,
                probmatching_method=prob_match,  ##major difference from pysteps paper, used before: "cdf"
                seed=random_seed,
                # domain= 'spectral',
                vel_pert_method=None,
            )

        else:
            precip_forecast_stacked = run_pysteps_forecast(
                precip=radar_precip[1:],
                precip_nowcast=DGMR_det_db,
                nowcasting_method="external_nowcast",
                mask_method=None,
                precip_models=nwp_precip,
                velocity=velocity_radar,
                velocity_models=velocity_nwp,
                timesteps=config.settings.timesteps,
                timestep=config.settings.timestep_interval,
                issuetime=pd.to_datetime(radar_metadata["timestamps"][-1]),
                n_ens_members=config.settings.n_ens_members,
                fft_method="pyfftw",
                # resample_distribution=False,
                precip_thr=radar_metadata["threshold"],
                kmperpixel=radar_metadata["xpixelsize"] / 1000.0,
                # noise_stddev_adj=noise, #major difference from pysteps paper
                noise_method=noise_method,  # major difference from pysteps paper
                weights_method=config.settings.weights_method,
                custom_weights=config.settings.custom_weights,
                return_weights=config.settings.return_weights,
                probmatching_method=prob_match,  ##major difference from pysteps paper, used before: "cdf"
                seed=random_seed,
                # domain= 'spectral',
                vel_pert_method=None,
            )

        # except: #If the blending fails, it is likely due to a an error with x= non-finite number in Gamma determination. Use climatological weights instead.
        #     log.info('Error in blending with weights method:', weights_method, ' - switching to custom climatological weights')
        #     GAMMA = np.array([
        #     [0.99805, 0.9933],
        #     [0.9925,  0.9752],
        #     [0.9776, 0.923],
        #     [0.9297,  0.750],
        #     [0.796,   0.367],
        #     [0.482,   0.069],
        # ])
        #     regr_pars = np.array(
        #         [
        #             [130.0, 165.0, 120.0, 55.0, 50.0, 15.0],
        #             [155.0, 220.0, 200.0, 75.0, 10e4, 10e4],
        #         ]
        #     )
        #     clim_cor_values = np.array([0.848, 0.537, 0.237, 0.065, 0.02, 0.0044])
        #     custom_weights = {
        #         "GAMMA": GAMMA,
        #         "regr_pars": regr_pars,
        #         "clim_cor_values": clim_cor_values,
        #     }
        #     precip_forecast_stacked = blending.steps.forecast(
        #         precip=radar_precip,
        #         precip_nowcast=DGMR_det_db,
        #         nowcasting_method="external_nowcast",
        #         mask_method=None,
        #         precip_models=nwp_precip,
        #         velocity=velocity_radar,
        #         velocity_models=velocity_nwp,
        #         timesteps=timesteps,
        #         timestep=timestep_interval,
        #         issuetime=pd.to_datetime(radar_metadata['timestamps'][-1]),
        #         n_ens_members=n_ens_members,
        #         # resample_distribution=False,
        #         precip_thr=radar_metadata["threshold"],
        #         kmperpixel=radar_metadata["xpixelsize"] / 1000.0,
        #         # noise_stddev_adj=None,
        #         # noise_method=None,
        #         weights_method = 'custom',
        #         custom_weights = custom_weights,
        #         return_weights = return_weights,
        #         probmatching_method="cdf",
        #         vel_pert_method=None,
        #     )

        if config.settings.return_weights:
            precip_forecast_stacked, weights = precip_forecast_stacked
            np.save(blended_file_weights, weights, allow_pickle=True)

        precip_forecast_mm, _ = converter(precip_forecast_stacked, radar_metadata)
        np.save(blended_file, precip_forecast_mm, allow_pickle=True)

    else:
        log.info(f"Found existing blended output {blended_file}")
        precip_forecast_mm = np.load(blended_file, allow_pickle=True)
        if config.settings.return_weights:
            try:
                weights = np.load(blended_file_weights, allow_pickle=True)
            except:
                weights = []

    # converter = pysteps.utils.get_method("mm/h")
    radar_precip_mm, _ = converter(DGMR_det_db, radar_metadata)
    nwp_precip_mm, _ = converter(nwp_precip, nwp_metadata)
    # log.info((time.time() - start_time), "minutes")
    if config.settings.return_weights:
        return precip_forecast_mm, radar_precip_mm, nwp_precip_mm, weights
    else:
        return precip_forecast_mm, radar_precip_mm, nwp_precip_mm
