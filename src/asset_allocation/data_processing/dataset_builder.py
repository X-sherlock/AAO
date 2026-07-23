from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asset_allocation.data_download import frame_to_ohlcv
from asset_allocation.data_processing.calendar_alignment import (
    holding_period_returns,
    weekly_rebalance_indices,
)
from asset_allocation.data_processing.dataset_freezer import (
    verify_checksums,
    write_checksums,
)
from asset_allocation.data_processing.split_generator import generate_splits
from asset_allocation.data_validation import fit_train_only_scaler
from asset_allocation.feature_engineering import build_features


@dataclass(frozen=True)
class ProcessedMarketDataset:
    dataset_id: str
    dates: np.ndarray
    assets: tuple[str, ...]
    feature_names: tuple[str, ...]
    features: np.ndarray
    returns: np.ndarray
    sample_covariance: np.ndarray
    shrinkage_covariance: np.ndarray
    correlation: np.ndarray
    liquidity_proxy: np.ndarray
    split_manifest: dict[str, Any]

    def covariance(self, estimator: str = "shrinkage") -> np.ndarray:
        if estimator == "sample":
            return self.sample_covariance
        if estimator == "shrinkage":
            return self.shrinkage_covariance
        raise ValueError("risk estimator must be 'sample' or 'shrinkage'")


def _inclusive_positions(
    dates: pd.DatetimeIndex, start: str, end: str
) -> tuple[int, int]:
    left = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    right = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    if not 0 <= left <= right < len(dates):
        raise ValueError(f"split range is outside processed data: {start}..{end}")
    return left, right


def build_processed_dataset(raw: dict[str, Any], project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    dataset_id = str(raw["dataset"]["id"])
    raw_path = Path(raw["dataset"].get("raw_path", f"data/raw/{dataset_id}"))
    processed_path = Path(
        raw["dataset"].get("processed_path", f"data/processed/{dataset_id}")
    )
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    if not processed_path.is_absolute():
        processed_path = root / processed_path
    if processed_path.exists():
        raise FileExistsError(
            f"processed dataset already exists and cannot be overwritten: {processed_path}"
        )
    verify_checksums(raw_path)
    manifest = json.loads(
        (raw_path / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["dataset_id"] != dataset_id:
        raise ValueError("raw manifest dataset_id does not match build config")
    assets = tuple(manifest["asset_universe"])
    frame = pd.read_parquet(raw_path / "ohlcv.parquet")
    data = frame_to_ohlcv(frame, assets=assets)
    feature_config = raw.get("features", {})
    windows = tuple(int(value) for value in feature_config.get("windows", [20, 60, 120]))
    features = build_features(
        data,
        windows=windows,
        ewma_decay=float(feature_config.get("ewma_decay", 0.94)),
    )
    frequency = str(raw.get("rebalance", {}).get("frequency", "weekly"))
    if frequency != "weekly":
        raise ValueError("first real-data version supports weekly rebalancing only")
    all_decisions = weekly_rebalance_indices(data.dates)
    decision_indices = all_decisions[all_decisions >= features.earliest_valid_index]
    values = features.values[decision_indices]
    if not np.all(np.isfinite(values)):
        raise ValueError("processed decision features contain non-finite values")
    dates = data.dates[decision_indices]
    returns = holding_period_returns(np.asarray(data.adjusted_close), decision_indices)
    sample_covariance = features.sample_covariance[decision_indices]
    shrinkage_covariance = features.shrinkage_covariance[decision_indices]
    correlation = features.correlation[decision_indices]
    liquidity_feature_indices = [
        features.names.index(f"volume_percentile_{asset}") for asset in assets
    ]
    liquidity = 1.0 - values[:, liquidity_feature_indices]
    split_configs = raw.get("splits")
    if split_configs is None:
        split_configs = [raw.get("split", {"mode": "fixed", "split_id": "fixed_v1"})]
    records = []
    for split_config in split_configs:
        records.extend(generate_splits(dates, split_config))
    split_manifest = {
        "dataset_id": dataset_id,
        "date_semantics": "inclusive boundaries; strictly ordered train then validation then test",
        "standardization": "one scaler per split fitted on training rows only",
        "splits": [record.as_dict() for record in records],
    }
    processed_path.mkdir(parents=True)
    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    pd.DataFrame(values, columns=features.names).assign(
        date=date_index
    ).loc[:, ["date", *features.names]].to_parquet(
        processed_path / "features.parquet", index=False
    )
    pd.DataFrame(returns, columns=assets).assign(date=date_index).loc[
        :, ["date", *assets]
    ].to_parquet(processed_path / "returns.parquet", index=False)
    pd.DataFrame(
        {
            "date": date_index,
            "source_daily_index": decision_indices,
            "decision_semantics": "state at close; action held for next period",
        }
    ).to_parquet(processed_path / "rebalance_calendar.parquet", index=False)
    np.savez_compressed(
        processed_path / "risk_matrices.npz",
        sample_covariance=sample_covariance,
        shrinkage_covariance=shrinkage_covariance,
        correlation=correlation,
        liquidity_proxy=liquidity,
    )
    (processed_path / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    feature_metadata = {
        "dataset_id": dataset_id,
        "assets": list(assets),
        "feature_names": list(features.names),
        "windows_daily_observations": list(windows),
        "earliest_valid_daily_index": features.earliest_valid_index,
        "raw_union_date_count": int(frame["date"].nunique()),
        "aligned_common_date_count": int(len(data.dates)),
        "alignment_dates_lost": int(frame["date"].nunique() - len(data.dates)),
        "decision_date_count": int(len(dates)),
        "return_semantics": "adj_close[t] / adj_close[t-1] - 1 between decision dates",
        "ohlc_semantics": "unadjusted provider OHLC retained verbatim",
        "adjusted_close_semantics": "provider adjusted close used for continuous returns",
        "volume_semantics": "provider raw volume; never imputed with zero",
        "dollar_volume_semantics": "close * volume; approximate liquidity proxy",
        "risk_estimators": ["sample", "shrinkage"],
        "offline_market_features": [
            "OHLCV", "asset returns", "volatility", "covariance", "correlation",
            "liquidity proxies", "credit pressure", "asset drawdown", "decision dates"
        ],
        "online_portfolio_state": [
            "current holdings", "pre-trade drift weights", "portfolio wealth",
            "portfolio peak", "portfolio drawdown", "realized turnover",
            "realized transaction cost", "projected final weights"
        ],
        "causality": "feature row t uses daily observations with dates <= decision date t",
    }
    (processed_path / "feature_metadata.json").write_text(
        json.dumps(feature_metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scaler_root = processed_path / "scaler_metadata"
    scaler_root.mkdir()
    for record in records:
        train_start, train_end = _inclusive_positions(
            date_index, record.train_start, record.train_end
        )
        scaler = fit_train_only_scaler(values, train_start, train_end)
        (scaler_root / f"{record.split_id}.json").write_text(
            json.dumps(scaler.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    names = [
        "features.parquet",
        "returns.parquet",
        "rebalance_calendar.parquet",
        "risk_matrices.npz",
        "split_manifest.json",
        "feature_metadata.json",
    ] + [
        f"scaler_metadata/{record.split_id}.json" for record in records
    ]
    write_checksums(processed_path, names)
    return processed_path


def load_processed_dataset(path: str | Path) -> ProcessedMarketDataset:
    root = Path(path)
    verify_checksums(root)
    metadata = json.loads((root / "feature_metadata.json").read_text(encoding="utf-8"))
    split_manifest = json.loads(
        (root / "split_manifest.json").read_text(encoding="utf-8")
    )
    feature_frame = pd.read_parquet(root / "features.parquet")
    return_frame = pd.read_parquet(root / "returns.parquet")
    risk = np.load(root / "risk_matrices.npz")
    feature_names = tuple(metadata["feature_names"])
    assets = tuple(metadata["assets"])
    if not feature_frame["date"].equals(return_frame["date"]):
        raise ValueError("feature and return calendars differ")
    return ProcessedMarketDataset(
        dataset_id=str(metadata["dataset_id"]),
        dates=feature_frame["date"].to_numpy(dtype="datetime64[D]"),
        assets=assets,
        feature_names=feature_names,
        features=feature_frame.loc[:, feature_names].to_numpy(dtype=np.float64),
        returns=return_frame.loc[:, assets].to_numpy(dtype=np.float64),
        sample_covariance=risk["sample_covariance"],
        shrinkage_covariance=risk["shrinkage_covariance"],
        correlation=risk["correlation"],
        liquidity_proxy=risk["liquidity_proxy"],
        split_manifest=split_manifest,
    )
