from __future__ import annotations

import json
from pathlib import Path

from .contracts import RunState


class RunStore:
    def __init__(self, root: Path, run_id: str) -> None:
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def save(self, state: RunState) -> None:
        temp_path = self.state_path.with_suffix(".json.tmp")
        temp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        temp_path.replace(self.state_path)

    def write_report(self, report: str) -> Path:
        report_path = self.run_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path

    def load(self) -> RunState:
        return RunState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

