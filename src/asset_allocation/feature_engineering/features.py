from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asset_allocation.data_download.synthetic import OHLCVData


@dataclass(frozen=True)
class FeatureSet:
    values: np.ndarray
    names: tuple[str, ...]
    returns: np.ndarray
    sample_covariance: np.ndarray
    shrinkage_covariance: np.ndarray
    correlation: np.ndarray
    earliest_valid_index: int

    @property
    def covariance(self) -> np.ndarray:
        """Default risk estimate used by the feasibility environment."""
        return self.shrinkage_covariance


def simple_returns(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    result = np.full_like(close, np.nan)
    result[1:] = close[1:] / close[:-1] - 1.0
    return result


def _ewma_std(values: np.ndarray, decay: float) -> np.ndarray:
    length = len(values)
    weights = decay ** np.arange(length - 1, -1, -1, dtype=np.float64)
    weights /= weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    variance = np.sum(((values - mean) ** 2) * weights[:, None], axis=0)
    return np.sqrt(np.maximum(variance, 0.0))


def _sample_covariance(values: np.ndarray) -> np.ndarray:
    n = values.shape[1]
    if len(values) < 2:
        return np.eye(n) * 1e-8
    covariance = np.cov(values, rowvar=False, ddof=1)
    return np.atleast_2d(covariance) + np.eye(n) * 1e-10


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-16))
    correlation = covariance / np.outer(scale, scale)
    return np.clip(correlation, -1.0, 1.0)


def build_features(
    data: OHLCVData,
    windows: tuple[int, ...] = (20, 60, 120),
    ewma_decay: float = 0.94,
) -> FeatureSet:
    """Build causal features: row t reads only OHLCV rows <= t."""
    adjusted_close = np.asarray(data.adjusted_close, dtype=np.float64)
    returns = simple_returns(adjusted_close)
    t_count, n_assets = data.shape
    largest = max(windows)
    names: list[str] = []
    names.extend(f"historical_return_{asset}" for asset in data.assets)
    for window in windows:
        names.extend(f"vol_ewma_{window}_{asset}" for asset in data.assets)
        names.extend(f"downside_vol_{window}_{asset}" for asset in data.assets)
    names.extend(
        [
            "average_correlation",
            "equity_treasury_correlation",
            "equity_gold_correlation",
            "credit_treasury_correlation",
            "correlation_max_eigenvalue",
        ]
    )
    names.extend(f"volume_percentile_{asset}" for asset in data.assets)
    names.extend(f"dollar_volume_{asset}" for asset in data.assets)
    names.extend(f"amihud_{asset}" for asset in data.assets)
    names.extend(f"amihud_change_{asset}" for asset in data.assets)
    names.extend(
        [
            "credit_pressure_hyg_ief",
            "credit_pressure_lqd_hyg",
            "credit_relative_drawdown",
        ]
    )
    names.extend(f"asset_drawdown_{asset}" for asset in data.assets)
    values = np.full((t_count, len(names)), np.nan, dtype=np.float64)
    sample_covariance = np.full(
        (t_count, n_assets, n_assets), np.nan, dtype=np.float64
    )
    shrinkage_covariance = np.full_like(sample_covariance, np.nan)
    correlation = np.full_like(sample_covariance, np.nan)
    asset_index = {asset: i for i, asset in enumerate(data.assets)}

    def pair(correlation: np.ndarray, left: str, right: str) -> float:
        if left not in asset_index or right not in asset_index:
            return 0.0
        return float(correlation[asset_index[left], asset_index[right]])

    for t in range(largest, t_count):
        row: list[float] = list(returns[t])
        for window in windows:
            history = returns[t - window + 1 : t + 1]
            row.extend(_ewma_std(history, ewma_decay))
            downside = np.minimum(history, 0.0)
            row.extend(np.sqrt(np.mean(downside**2, axis=0)))
        risk_history = returns[t - largest + 1 : t + 1]
        sample_cov = _sample_covariance(risk_history)
        diagonal = np.diag(np.diag(sample_cov))
        shrinkage_cov = 0.9 * sample_cov + 0.1 * diagonal
        sample_covariance[t] = sample_cov
        shrinkage_covariance[t] = shrinkage_cov
        corr = _correlation_from_covariance(shrinkage_cov)
        correlation[t] = corr
        off_diagonal = corr[~np.eye(n_assets, dtype=bool)]
        row.extend(
            [
                float(off_diagonal.mean()),
                pair(corr, "SPY", "IEF"),
                pair(corr, "SPY", "GLD"),
                pair(corr, "HYG", "IEF"),
                float(np.linalg.eigvalsh(corr)[-1]),
            ]
        )
        volume_history = data.volume[t - largest + 1 : t + 1]
        row.extend(np.mean(volume_history <= data.volume[t], axis=0))
        dollar_volume = data.close[t] * data.volume[t]
        row.extend(dollar_volume)
        amihud = np.abs(returns[t]) / np.maximum(dollar_volume, 1e-12)
        previous_dollar_volume = data.close[t - 1] * data.volume[t - 1]
        previous_amihud = np.abs(returns[t - 1]) / np.maximum(
            previous_dollar_volume, 1e-12
        )
        row.extend(amihud)
        row.extend(amihud - previous_amihud)
        if "HYG" in asset_index and "IEF" in asset_index:
            hy = asset_index["HYG"]
            tr = asset_index["IEF"]
            hyg_period = float(np.prod(1.0 + risk_history[:, hy]) - 1.0)
            ief_period = float(np.prod(1.0 + risk_history[:, tr]) - 1.0)
            pressure_hyg_ief = ief_period - hyg_period
            if "LQD" in asset_index:
                lqd = asset_index["LQD"]
                lqd_period = float(np.prod(1.0 + risk_history[:, lqd]) - 1.0)
                pressure_lqd_hyg = lqd_period - hyg_period
            else:
                pressure_lqd_hyg = 0.0
            relative = risk_history[:, hy] - risk_history[:, tr]
            relative_wealth = np.cumprod(1.0 + relative)
            row.extend(
                [
                    pressure_hyg_ief,
                    pressure_lqd_hyg,
                    float(1.0 - relative_wealth[-1] / relative_wealth.max()),
                ]
            )
        else:
            row.extend([0.0, 0.0, 0.0])
        running_peak = np.max(adjusted_close[: t + 1], axis=0)
        row.extend(1.0 - adjusted_close[t] / running_peak)
        values[t] = np.asarray(row)
    return FeatureSet(
        values,
        tuple(names),
        returns,
        sample_covariance,
        shrinkage_covariance,
        correlation,
        largest,
    )
