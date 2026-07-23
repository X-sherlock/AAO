from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_quality_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target
