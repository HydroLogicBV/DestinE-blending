from dataclasses import dataclass
from pathlib import Path

RADAR_DIR = "knmi_radar"
RADAR_GAUGE_ADJ_DIR = "knmi_radar_gauge_adj"
DESTINE_DIR = "ExtremesDT"
IFS_DIR = "IFS"
DGMR_DIR = "DGMR"
BLENDED_DIR = "blended"
RESOURCES_DIR = "resources"


@dataclass(frozen=True)
class Dirs:
    """Directory layout for one nowcast-blend run."""

    data: Path
    input: Path
    output: Path
    destine: Path
    ifs: Path
    dgmr: Path
    radar: Path
    blended: Path
    resources: Path
    machine_learning: Path

    def ensure_runtime_dirs(self) -> None:
        for path in (self.destine, self.ifs, self.dgmr, self.radar, self.blended):
            path.mkdir(parents=True, exist_ok=True)


def build_dirs(cfg, date) -> Dirs:
    data = Path(str(cfg.paths.data)).expanduser()
    input_dir = data / str(cfg.paths.input_dir)
    output_dir = data / str(cfg.paths.output_dir)
    resources_dir = Path(RESOURCES_DIR)
    radar_dir = RADAR_GAUGE_ADJ_DIR if cfg.settings.gauge_adjusted else RADAR_DIR
    date_suffix = Path(f"{date:%Y}") / f"{date:%m}"

    return Dirs(
        data=data,
        input=input_dir,
        output=output_dir,
        destine=input_dir / DESTINE_DIR / date_suffix,
        ifs=input_dir / IFS_DIR / date_suffix,
        dgmr=input_dir / DGMR_DIR / date_suffix,
        radar=input_dir / radar_dir,
        blended=output_dir / BLENDED_DIR / date_suffix,
        resources=resources_dir,
        machine_learning=resources_dir / "machine_learning",
    )
