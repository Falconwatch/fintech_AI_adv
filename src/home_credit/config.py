from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path
    data_dir: Path
    artifacts_dir: Path
    random_seed: int
    holdout_size: float
    training: dict[str, Any]


def load_config(config_path: str | Path) -> ProjectConfig:
    config_file = Path(config_path).resolve()
    root_dir = config_file.parents[1]

    with config_file.open("r", encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh)

    project_cfg = raw_config["project"]
    split_cfg = raw_config["split"]

    return ProjectConfig(
        root_dir=root_dir,
        data_dir=root_dir / project_cfg["data_dir"],
        artifacts_dir=root_dir / project_cfg["artifacts_dir"],
        random_seed=int(project_cfg["random_seed"]),
        holdout_size=float(split_cfg["holdout_size"]),
        training=raw_config["training"],
    )
