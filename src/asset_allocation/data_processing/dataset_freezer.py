from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from asset_allocation.data_download.real_ohlcv import provider_from_config
from asset_allocation.data_validation import validate_ohlcv_frame, write_quality_report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(directory: Path, names: list[str]) -> Path:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in sorted(names)]
    path = directory / "checksums.sha256"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_checksums(directory: str | Path) -> dict[str, bool]:
    root = Path(directory)
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"checksum manifest is missing: {manifest}")
    results: dict[str, bool] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        target = root / name
        results[name] = target.is_file() and sha256_file(target) == expected
    if not results or not all(results.values()):
        failed = [name for name, passed in results.items() if not passed]
        raise ValueError("checksum verification failed: " + ", ".join(failed))
    return results


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "pyarrow", "yfinance", "PyYAML"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def freeze_raw_dataset(raw: dict[str, Any], project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    dataset_id = str(raw["dataset"]["id"])
    dataset_version = str(raw["dataset"].get("version", dataset_id))
    output_root = Path(raw["dataset"].get("raw_root", "data/raw"))
    if not output_root.is_absolute():
        output_root = root / output_root
    output_dir = output_root / dataset_id
    if output_dir.exists():
        raise FileExistsError(
            f"frozen dataset directory already exists and cannot be overwritten: {output_dir}"
        )
    universe = raw["assets"]
    assets = tuple(item["asset_code"] if isinstance(item, dict) else item for item in universe)
    date_range = raw.get("date_range", {})
    start = date_range.get("start")
    end = date_range.get("end")
    provider = provider_from_config(raw, root)
    parameters = dict(raw.get("provider", {}).get("parameters", {}))
    frame = provider.fetch(assets, start, end, **parameters)
    validation = raw.get("validation", {})
    quality = validate_ohlcv_frame(
        frame,
        assets,
        requested_start_date=start,
        requested_end_date=end,
        extreme_return_threshold=float(validation.get("extreme_return_threshold", 0.35)),
        extreme_return_action=str(validation.get("extreme_return_action", "flag")),
        max_consecutive_missing=int(validation.get("max_consecutive_missing", 5)),
    )
    output_dir.mkdir(parents=True)
    ohlcv_path = output_dir / "ohlcv.parquet"
    frame.to_parquet(ohlcv_path, index=False)
    metadata_rows: list[dict[str, Any]] = []
    metadata_by_code = {
        item["asset_code"]: item for item in universe if isinstance(item, dict)
    }
    for asset in assets:
        subset = frame[frame["asset_code"] == asset].sort_values("date")
        info = metadata_by_code.get(asset, {})
        metadata_rows.append(
            {
                "asset_code": asset,
                "asset_name": info.get("asset_name", asset),
                "asset_class": info.get("asset_class", "unspecified"),
                "currency": str(subset["currency"].iloc[0]),
                "provider": provider.name,
                "requested_start_date": start,
                "requested_end_date": end,
                "actual_start_date": subset["date"].min().date().isoformat(),
                "actual_end_date": subset["date"].max().date().isoformat(),
                "row_count": int(len(subset)),
                "first_valid_date": subset["date"].min().date().isoformat(),
                "last_valid_date": subset["date"].max().date().isoformat(),
            }
        )
    pd.DataFrame(metadata_rows).to_csv(
        output_dir / "asset_metadata.csv", index=False, encoding="utf-8"
    )
    actual_common_start = max(row["actual_start_date"] for row in metadata_rows)
    actual_common_end = min(row["actual_end_date"] for row in metadata_rows)
    manifest = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider.name,
        "provider_parameters": parameters,
        "asset_universe": list(assets),
        "requested_date_range": [start, end],
        "actual_common_date_range": [actual_common_start, actual_common_end],
        "raw_file_paths": ["ohlcv.parquet", "asset_metadata.csv"],
        "row_counts": {
            "total": int(len(frame)),
            "by_asset": {
                asset: int((frame["asset_code"] == asset).sum()) for asset in assets
            },
        },
        "missing_value_counts": quality["missing_value_counts"],
        "duplicate_counts": quality["duplicate_count"],
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
        "git_commit": _git_commit(root),
        "immutability": "directory must never be overwritten",
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_quality_report(quality, output_dir / "quality_report.json")
    write_checksums(
        output_dir,
        [
            "ohlcv.parquet",
            "asset_metadata.csv",
            "dataset_manifest.json",
            "quality_report.json",
        ],
    )
    return output_dir
