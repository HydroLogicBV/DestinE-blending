#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGMR Nowcasting Module

This module runs the DGMR nowcasting model for the Netherlands using KNMI radar data.
It can be imported into other Python scripts or run as a standalone script.

Main function:
    run_dgmr() -> np.ndarray
        Returns DGMR_det (the deterministic nowcast)

Author: Joep (refactored as module by ChatGPT)
"""

import os, certifi
import sys
import time as timing
import shutil
import requests
import numpy as np
from functools import lru_cache
from datetime import datetime, timedelta
from base64 import decodebytes

from pysteps import io
from pysteps.utils import conversion

os.environ["CURL_CA_BUNDLE"] = certifi.where()

import tensorflow as tf
import tensorflow_hub


import logging

log = logging.getLogger(__name__)

# Model location
REMOTE_TFHUB_BASE_PATH = "gs://dm-nowcasting-example-data/tfhub_snapshots"


# DGMR settings
NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18
# TODO: These are hardcoded, will they change? If not, no need to pass them around
DGMR_INPUT_HEIGHT = 1536
DGMR_INPUT_WIDTH = 1280


def get_model_path(input_height, input_width):
    """Return configured local DGMR model path or the remote TF-Hub snapshot."""
    local_model_path = os.environ.get("DGMR_MODEL_PATH")
    if local_model_path:
        return local_model_path
    return os.path.join(REMOTE_TFHUB_BASE_PATH, f"{input_height}x{input_width}")


def load_module(input_height, input_width):
    """Load DGMR from local SavedModel path if configured, otherwise TF-Hub/GCS."""
    model_path = get_model_path(input_height, input_width)
    log.info("Loading DGMR model from %s", model_path)
    hub_module = tensorflow_hub.load(model_path)
    return hub_module.signatures["default"]


@lru_cache(maxsize=1)
def get_module():
    """Load the DGMR model on first use and reuse it within this Python process."""
    return load_module(DGMR_INPUT_HEIGHT, DGMR_INPUT_WIDTH)


def predict(
    module,
    input_frames,
    num_samples=1,
    include_input_frames_in_result=False,
    random_seed=None,
):
    """Run DGMR model prediction."""
    input_frames = tf.math.maximum(input_frames, 0.0)
    input_frames = tf.expand_dims(input_frames, 0)
    input_frames = tf.tile(input_frames, multiples=[num_samples, 1, 1, 1, 1])

    _, input_signature = module.structured_input_signature
    z_size = input_signature["z"].shape[1]
    if random_seed is None:
        z_samples = tf.random.normal(shape=(num_samples, z_size))
    else:
        z_samples = tf.random.stateless_normal(
            shape=(num_samples, z_size),
            seed=tf.constant(random_seed, dtype=tf.int32),
        )

    inputs = {
        "z": z_samples,
        "labels$onehot": tf.ones(shape=(num_samples, 1)),
        "labels$cond_frames": input_frames,
    }
    samples = module(**inputs)["default"]

    if not include_input_frames_in_result:
        samples = samples[:, NUM_INPUT_FRAMES:, ...]

    return tf.math.maximum(samples, 0.0)


def run_dgmr(R, module=None, runtimes=4, random_seed=None):
    # Load the model for size 1536 by 1280

    """Run the DGMR pipeline and return deterministic forecast (DGMR_det)."""
    if module is None:
        module = get_module()

    t1 = timing.time()

    # --- Prepare DGMR input ---
    input_DGMR = R[-4:]
    paddings = tf.constant([[0, 0], [385, 386], [290, 290]])
    input_DGMR = tf.pad(input_DGMR, paddings, "CONSTANT")
    input_DGMR = np.reshape(np.float32(input_DGMR), (4, 1536, 1280, 1))
    input_DGMR[np.isinf(input_DGMR)] = 0.0

    log.info("Running DGMR rollout 1/%s...", runtimes)
    prediction_1 = predict(
        module,
        input_DGMR,
        include_input_frames_in_result=True,
        random_seed=None if random_seed is None else [random_seed, 0],
    )
    prediction_1 = np.reshape(prediction_1, (22, 1536, 1280, 1))[3:]
    extended_predictions = prediction_1.copy()

    for i in range(runtimes - 1):
        log.info("Running DGMR rollout %s/%s...", i + 2, runtimes)
        prediction = predict(
            module,
            extended_predictions[-4:],
            include_input_frames_in_result=False,
            random_seed=None if random_seed is None else [random_seed, i + 1],
        )
        prediction = np.reshape(prediction, (18, 1536, 1280, 1))
        extended_predictions = np.concatenate((extended_predictions, prediction))
        # copy the last DGMR image, so that there is enough images for the blending
        if i == runtimes - 1:
            extended_predictions = np.concatenate(
                (extended_predictions, extended_predictions[-1])
            )

    # prediction_2 = predict(module, prediction_1[-4:], include_input_frames_in_result=False)
    # prediction_2 = np.reshape(prediction_2, (18, 1536, 1280, 1))

    # prediction_3 = predict(module, prediction_2[-4:], include_input_frames_in_result=False)
    # prediction_3 = np.reshape(prediction_3, (18, 1536, 1280, 1))

    # prediction_4 = predict(module, prediction_3[-4:], include_input_frames_in_result=False)
    # prediction_4 = np.reshape(prediction_4, (18, 1536, 1280, 1))

    # extended_predictions = np.concatenate((prediction_1, prediction_2, prediction_3,prediction_4))
    DGMR_det = np.reshape(
        extended_predictions[:, 385:1150, 290:990, :],
        (len(extended_predictions), 765, 700),
    )

    t2 = timing.time()
    log.info(
        f"DGMR run completed in {int((t2 - t1) / 60)} min {int((t2 - t1) % 60)} sec"
    )

    return DGMR_det


import time as timing
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_dgmr_ensemble(
    R,
    ens_members,
    module=None,
    forecast_length=4,
    max_workers=None,
    random_seed=None,
):
    """Run DGMR ensemble.

    DGMR is memory-heavy, so the safe default is one worker. Increase the
    configured worker count only when the container has enough memory.
    """
    log.info(f"Launching DGMR ensemble with {ens_members} members...")
    if module is None:
        module = get_module()
    if random_seed is not None:
        random_seed = int(random_seed)
        log.info("Using DGMR random_seed=%s", random_seed)

    results = [None] * ens_members
    t_start = timing.time()

    if max_workers is None:
        max_workers = 1
    max_workers = int(max_workers)
    max_workers = max(1, min(max_workers, ens_members))
    if max_workers == 1:
        for i in range(ens_members):
            log.info("Running DGMR ensemble member %s/%s...", i + 1, ens_members)
            member_seed = None if random_seed is None else random_seed + i
            results[i] = run_dgmr(R, module, forecast_length, random_seed=member_seed)
            log.info("Ensemble member %s/%s finished.", i + 1, ens_members)
    else:
        log.info("Running DGMR ensemble with %s parallel workers.", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_dgmr,
                    R,
                    module,
                    forecast_length,
                    random_seed=None if random_seed is None else random_seed + i,
                ): i
                for i in range(ens_members)
            }
            for completed, f in enumerate(as_completed(futures), 1):
                i = futures[f]
                result = f.result()
                results[i] = result
                log.info(f"Ensemble member {completed}/{ens_members} finished.")

    # Stack ensemble results: shape = (ens_members, time, y, x)
    DGMR_ens = np.stack(results, axis=0)

    t_end = timing.time()
    log.info(
        f"DGMR ensemble completed in {int((t_end - t_start) / 60)} min {int((t_end - t_start) % 60)} sec"
    )

    return DGMR_ens


if __name__ == "__main__":
    output = run_dgmr(R)
    log.info("DGMR_det shape:", output.shape)
