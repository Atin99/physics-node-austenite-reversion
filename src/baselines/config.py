from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Project42Config:
    root: Path = Path(__file__).resolve().parents[2]

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def synthetic_dir(self) -> Path:
        return self.root / "data" / "synthetic"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


def ensure_dirs(cfg: Project42Config) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.synthetic_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
