from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_strict_json(path: Path, payload: Any) -> None:
    """Write standards-compliant JSON and reject NaN/Infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
