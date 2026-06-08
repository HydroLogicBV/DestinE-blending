"""Operational DGMR generation, file reuse, and subprocess entrypoint helpers."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from nowcast_blend.nowcast.dgmr_for_blending import run_dgmr_ensemble
from nowcast_blend.utils.logging import set_log_format

log = logging.getLogger(__name__)


def dgmr_forecast_length(cfg):
    """Return the number of 90-minute DGMR rollouts needed for the configured horizon."""
    return int((int(cfg.settings.timestep_interval) * int(cfg.settings.timesteps)) / 90)


def run_dgmr_ensemble_subprocess(radar_precip, dgmr_file, cfg):
    """Generate DGMR in a separate process so TensorFlow memory is released."""
    dgmr_input_file = dgmr_file.with_suffix(".input.npy")
    np.save(dgmr_input_file, radar_precip, allow_pickle=False)

    cmd = [
        sys.executable,
        "-m",
        "nowcast_blend.nowcast.dgmr_orchestration",
        "--input",
        str(dgmr_input_file),
        "--output",
        str(dgmr_file),
        "--ens-members",
        str(int(cfg.settings.n_ens_members_dgmr)),
        "--forecast-length",
        str(dgmr_forecast_length(cfg)),
        "--random-seed",
        str(int(cfg.runtime.dgmr_random_seed)),
    ]
    if cfg.runtime.dgmr_max_workers is not None:
        cmd.extend(["--max-workers", str(int(cfg.runtime.dgmr_max_workers))])

    try:
        subprocess.run(cmd, check=True)
    finally:
        dgmr_input_file.unlink(missing_ok=True)


def load_or_generate_dgmr_ensemble(radar_precip, dgmr_file, cfg):
    """Load an existing DGMR ensemble or generate it using the configured mode."""
    if dgmr_file.exists():
        log.info(f"Importing DGMR file: {dgmr_file}")
        return np.load(dgmr_file, allow_pickle=True)

    log.info(f"DGMR file is missing so we have to make the nowcast: {dgmr_file}")
    if cfg.runtime.dgmr_subprocess:
        run_dgmr_ensemble_subprocess(radar_precip, dgmr_file, cfg)
        return np.load(dgmr_file, allow_pickle=True)

    log.info("Generating DGMR ensemble in main process")
    dgmr = run_dgmr_ensemble(
        radar_precip,
        ens_members=int(cfg.settings.n_ens_members_dgmr),
        forecast_length=dgmr_forecast_length(cfg),
        max_workers=cfg.runtime.dgmr_max_workers,
        random_seed=cfg.runtime.dgmr_random_seed,
    )
    np.save(dgmr_file, dgmr, allow_pickle=True)
    return dgmr


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DGMR ensemble to a file.")
    parser.add_argument("--input", required=True, help="Input radar precipitation .npy")
    parser.add_argument("--output", required=True, help="Output DGMR ensemble .npy")
    parser.add_argument("--ens-members", type=int, required=True)
    parser.add_argument("--forecast-length", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    set_log_format()

    args = parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    radar_precip = np.load(input_file, allow_pickle=False)
    log.info("Generating DGMR ensemble in subprocess: %s", output_file)
    dgmr = run_dgmr_ensemble(
        radar_precip,
        ens_members=args.ens_members,
        forecast_length=args.forecast_length,
        max_workers=args.max_workers,
        random_seed=args.random_seed,
    )
    np.save(output_file, dgmr, allow_pickle=True)
    log.info("DGMR ensemble written: %s", output_file)


if __name__ == "__main__":
    main()
