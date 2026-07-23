from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OHLCVData:
    dates: np.ndarray
    assets: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    adjusted_close: np.ndarray | None = None
    currencies: tuple[str, ...] | None = None
    sources: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.adjusted_close is None:
            object.__setattr__(self, "adjusted_close", np.asarray(self.close).copy())
        if self.currencies is None:
            object.__setattr__(
                self, "currencies", tuple("USD" for _ in self.assets)
            )
        if self.sources is None:
            object.__setattr__(
                self, "sources", tuple("synthetic" for _ in self.assets)
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape


def make_synthetic_ohlcv(
    assets: tuple[str, ...],
    periods: int = 260,
    seed: int = 20260723,
) -> OHLCVData:
    """Create a deterministic fixture. It is not research or backtest data."""
    if periods < 30:
        raise ValueError("periods must be at least 30")
    rng = np.random.default_rng(seed)
    n = len(assets)
    factor = rng.normal(0.00015, 0.007, size=periods)
    defensive = rng.normal(0.00008, 0.003, size=periods)
    loadings = np.linspace(1.2, -0.3, n)
    log_returns = (
        0.0001
        + factor[:, None] * loadings[None, :]
        + defensive[:, None] * (1.0 - np.abs(loadings))[None, :]
        + rng.normal(0.0, 0.004, size=(periods, n))
    )
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    overnight = rng.normal(0.0, 0.001, size=(periods, n))
    open_ = close * np.exp(overnight)
    spread = np.abs(rng.normal(0.004, 0.0015, size=(periods, n)))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=14.0, sigma=0.35, size=(periods, n))
    start = date(2020, 1, 2)
    dates = np.array(
        [np.datetime64(start + timedelta(days=i)) for i in range(periods)],
        dtype="datetime64[D]",
    )
    return OHLCVData(dates, assets, open_, high, low, close, volume)


class SyntheticProvider:
    """Adapter that exposes deterministic fixtures through the provider protocol."""

    name = "synthetic"

    def fetch(
        self,
        assets: tuple[str, ...],
        start_date: str | None = None,
        end_date: str | None = None,
        **parameters: Any,
    ):
        from asset_allocation.data_download.base import ohlcv_to_frame

        data = make_synthetic_ohlcv(
            assets,
            periods=int(parameters.get("periods", 260)),
            seed=int(parameters.get("seed", 20260723)),
        )
        return ohlcv_to_frame(data, source=self.name)


def freeze_fixture(data: OHLCVData, csv_path: Path, metadata_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "asset", "open", "high", "low", "close", "volume"])
        for t, dt in enumerate(data.dates):
            for i, asset in enumerate(data.assets):
                writer.writerow(
                    [
                        str(dt),
                        asset,
                        f"{data.open[t, i]:.10f}",
                        f"{data.high[t, i]:.10f}",
                        f"{data.low[t, i]:.10f}",
                        f"{data.close[t, i]:.10f}",
                        f"{data.volume[t, i]:.4f}",
                    ]
                )
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = {
        "synthetic_fixture_only": True,
        "prohibited_interpretation": "not real market data and not a performance result",
        "generator": "asset_allocation.data_download.synthetic.make_synthetic_ohlcv",
        "assets": list(data.assets),
        "date_range": [str(data.dates[0]), str(data.dates[-1])],
        "rows": int(data.shape[0] * data.shape[1]),
        "adjustment": "not applicable",
        "timezone": "fixture calendar without exchange semantics",
        "sha256": digest,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
