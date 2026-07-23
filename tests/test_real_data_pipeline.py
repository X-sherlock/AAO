from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from asset_allocation.config import CostConfig, ExperimentConfig, ProjectionConfig
from asset_allocation.data_download import (
    LocalFileProvider,
    SyntheticProvider,
    YFinanceProvider,
    frame_to_ohlcv,
    make_synthetic_ohlcv,
    ohlcv_to_frame,
)
from asset_allocation.data_processing import (
    build_processed_dataset,
    freeze_raw_dataset,
    generate_splits,
    load_processed_dataset,
    verify_checksums,
)
from asset_allocation.data_validation import (
    assert_causal_perturbation,
    fit_train_only_scaler,
    validate_ohlcv_frame,
)
from asset_allocation.exceptions import DataValidationError
from asset_allocation.feature_engineering import build_features
from asset_allocation.portfolio_environment import PortfolioEnvironment


ASSETS = ("SPY", "IEF", "HYG", "GLD")


def canonical_frame(periods: int = 220) -> pd.DataFrame:
    return ohlcv_to_frame(make_synthetic_ohlcv(ASSETS, periods=periods, seed=17))


def raw_config(csv_path: Path, dataset_id: str = "test_real_v1") -> dict:
    return {
        "dataset": {
            "id": dataset_id,
            "version": "v1",
            "raw_root": "data/raw",
            "raw_path": f"data/raw/{dataset_id}",
            "processed_path": f"data/processed/{dataset_id}",
        },
        "provider": {"name": "local_csv", "path": str(csv_path)},
        "date_range": {"start": None, "end": None},
        "assets": [
            {
                "asset_code": asset,
                "asset_name": asset,
                "asset_class": "fixture",
            }
            for asset in ASSETS
        ],
        "validation": {
            "extreme_return_threshold": 0.9,
            "extreme_return_action": "flag",
            "max_consecutive_missing": 2,
        },
        "features": {"windows": [5, 10, 20], "ewma_decay": 0.94},
        "rebalance": {"frequency": "weekly"},
        "split": {
            "mode": "fixed",
            "split_id": "fixed_v1",
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
        },
    }


def build_fixture(tmp_path: Path):
    csv_path = tmp_path / "source.csv"
    canonical_frame().to_csv(csv_path, index=False)
    config = raw_config(csv_path)
    raw_path = freeze_raw_dataset(config, tmp_path)
    processed_path = build_processed_dataset(config, tmp_path)
    return config, raw_path, processed_path


def test_real_provider_schema(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=3)
    provider_frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
            "Adj Close": [10.4, 11.4, 12.4],
            "Volume": [100, 110, 120],
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda *args, **kwargs: provider_frame.copy()),
    )
    result = YFinanceProvider().fetch(("SPY",), "2020-01-01", "2020-02-01")
    assert tuple(result.columns) == (
        "date", "asset_code", "open", "high", "low", "close",
        "adjusted_close", "volume", "currency", "source"
    )
    assert set(result["asset_code"]) == {"SPY"}


def test_local_file_provider_schema(tmp_path):
    path = tmp_path / "ohlcv.csv"
    canonical_frame(40).to_csv(path, index=False)
    result = LocalFileProvider(path).fetch(ASSETS, None, None)
    assert len(result) == 160
    assert set(result["asset_code"]) == set(ASSETS)


def test_dataset_freeze_checksum(tmp_path):
    _, raw_path, _ = build_fixture(tmp_path)
    assert all(verify_checksums(raw_path).values())
    assert {
        "ohlcv.parquet", "asset_metadata.csv", "dataset_manifest.json",
        "quality_report.json", "checksums.sha256"
    }.issubset({path.name for path in raw_path.iterdir()})


def test_no_duplicate_keys_and_ohlc_consistency():
    report = validate_ohlcv_frame(canonical_frame(40), ASSETS)
    assert report["duplicate_count"] == 0
    assert report["ohlc_high_errors"] == report["ohlc_low_errors"] == 0


def test_no_future_leakage():
    data = make_synthetic_ohlcv(ASSETS, periods=80, seed=9)
    original = build_features(data, windows=(5, 10))
    adjusted = np.asarray(data.adjusted_close).copy()
    adjusted[41:] *= 4
    changed = build_features(
        replace(data, adjusted_close=adjusted), windows=(5, 10)
    )
    assert_causal_perturbation(original.values, changed.values, 40)
    assert not np.allclose(original.values[41], changed.values[41])


def test_train_only_scaler():
    values = np.arange(60, dtype=float).reshape(20, 3)
    scaler = fit_train_only_scaler(values, 0, 9)
    changed = values.copy()
    changed[10:] *= 1000
    changed_scaler = fit_train_only_scaler(changed, 0, 9)
    np.testing.assert_allclose(scaler.mean, changed_scaler.mean)
    np.testing.assert_allclose(scaler.scale, changed_scaler.scale)
    changed_train = values.copy()
    changed_train[:10] += 5
    assert not np.allclose(
        scaler.mean, fit_train_only_scaler(changed_train, 0, 9).mean
    )


def test_fixed_split_no_overlap():
    dates = pd.date_range("2020-01-01", periods=100)
    split = generate_splits(
        dates,
        {"mode": "fixed", "split_id": "x", "train_fraction": 0.6, "validation_fraction": 0.2},
    )[0]
    assert split.train_end < split.validation_start
    assert split.validation_end < split.test_start


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_walk_forward_split_no_overlap(mode):
    dates = pd.date_range("2020-01-01", periods=60)
    splits = generate_splits(
        dates,
        {
            "mode": mode,
            "train_periods": 20,
            "validation_periods": 10,
            "test_periods": 10,
            "step_periods": 10,
        },
    )
    assert len(splits) >= 2
    assert all(
        item.train_end < item.validation_start
        and item.validation_end < item.test_start
        for item in splits
    )


def test_real_and_synthetic_feature_schema_compatible(tmp_path):
    _, _, processed_path = build_fixture(tmp_path)
    real = load_processed_dataset(processed_path)
    synthetic = build_features(
        make_synthetic_ohlcv(ASSETS, periods=220, seed=20),
        windows=(5, 10, 20),
    )
    assert real.feature_names == synthetic.names


def _small_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=1,
        assets=ASSETS,
        base_weights=np.full(4, 0.25),
        alpha_max=0.1,
        weight_min=np.zeros(4),
        weight_max=np.ones(4),
        max_deviation=np.ones(4),
        turnover_max=2.0,
        risk_max_daily=1.0,
        feature_windows=(5, 10, 20),
        ewma_decay=0.94,
        cost=CostConfig(1.0, 0.0, 0.0),
        projection=ProjectionConfig(1e-9, 1000),
    )


def test_environment_runs_on_real_dataset(tmp_path):
    _, _, processed_path = build_fixture(tmp_path)
    data = load_processed_dataset(processed_path)
    scaler = fit_train_only_scaler(data.features, 0, 5)
    env = PortfolioEnvironment(
        scaler.transform(data.features),
        data.returns,
        data.shrinkage_covariance,
        data.liquidity_proxy,
        _small_config(),
        start_index=0,
        end_index=2,
    )
    trace = []
    while True:
        transition = env.step(np.zeros(4), 0.0)
        trace.append(transition.wealth)
        if transition.done:
            break
    assert len(trace) == 3
    assert all(np.isfinite(trace))


def test_missing_asset_fails_explicitly(tmp_path):
    path = tmp_path / "missing.csv"
    canonical_frame(40).query("asset_code != 'GLD'").to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="missing required assets"):
        LocalFileProvider(path).fetch(ASSETS, None, None)


def test_provider_failure_does_not_fallback_to_synthetic(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame()),
    )
    with pytest.raises(RuntimeError, match="no rows"):
        YFinanceProvider().fetch(("SPY",), None, None)


def test_synthetic_provider_uses_same_protocol():
    result = SyntheticProvider().fetch(ASSETS, None, None, periods=40, seed=1)
    restored = frame_to_ohlcv(result, ASSETS)
    assert restored.shape == (40, 4)
