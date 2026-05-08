from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import RunTelemetry


def save_telemetry(telemetry: RunTelemetry, cfg: Config) -> None:
    log_path = Path(cfg.telemetry_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(telemetry)) + "\n")
