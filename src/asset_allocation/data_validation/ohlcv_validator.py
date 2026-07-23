from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from asset_allocation.data_download.base import normalize_ohlcv_frame
from asset_allocation.exceptions import DataValidationError


def validate_ohlcv_frame(
    frame: pd.DataFrame,
    expected_assets: tuple[str, ...],
    requested_start_date: str | None = None,
    requested_end_date: str | None = None,
    extreme_return_threshold: float = 0.35,
    extreme_return_action: str = "flag",
    max_consecutive_missing: int = 5,
) -> dict[str, Any]:
    data = normalize_ohlcv_frame(frame)
    failures: list[str] = []
    warnings: list[str] = []
    duplicate_count = int(data.duplicated(["date", "asset_code"]).sum())
    if duplicate_count:
        failures.append(f"duplicate date+asset_code keys: {duplicate_count}")
    missing_assets = sorted(set(expected_assets) - set(data["asset_code"]))
    if missing_assets:
        failures.append("missing required assets: " + ", ".join(missing_assets))
    missing_counts = {
        column: int(data[column].isna().sum()) for column in data.columns
    }
    required_missing = sum(missing_counts[column] for column in (
        "date", "asset_code", "open", "high", "low", "close",
        "adjusted_close", "volume", "currency", "source"
    ))
    if required_missing:
        failures.append(f"required fields contain {required_missing} missing values")
    for column in ("open", "high", "low", "close", "adjusted_close"):
        count = int((data[column] <= 0).sum())
        if count:
            failures.append(f"{column} has {count} non-positive values")
    negative_volume = int((data["volume"] < 0).sum())
    if negative_volume:
        failures.append(f"volume has {negative_volume} negative values")
    high_errors = int(
        (
            data["high"]
            < data[["open", "close", "low"]].max(axis=1)
        ).sum()
    )
    low_errors = int(
        (
            data["low"]
            > data[["open", "close", "high"]].min(axis=1)
        ).sum()
    )
    if high_errors:
        failures.append(f"high consistency errors: {high_errors}")
    if low_errors:
        failures.append(f"low consistency errors: {low_errors}")
    ranges: dict[str, dict[str, Any]] = {}
    extreme_rows: list[dict[str, Any]] = []
    for asset in expected_assets:
        subset = data[data["asset_code"] == asset].sort_values("date")
        if subset.empty:
            continue
        if not subset["date"].is_monotonic_increasing:
            failures.append(f"dates are not increasing for {asset}")
        returns = subset["adjusted_close"].pct_change()
        extreme = subset.loc[returns.abs() > extreme_return_threshold, ["date", "adjusted_close"]]
        for index, row in extreme.iterrows():
            extreme_rows.append(
                {
                    "asset_code": asset,
                    "date": row["date"].date().isoformat(),
                    "adjusted_close": float(row["adjusted_close"]),
                    "return": float(returns.loc[index]),
                    "disposition": extreme_return_action,
                }
            )
        ranges[asset] = {
            "actual_start_date": subset["date"].min().date().isoformat(),
            "actual_end_date": subset["date"].max().date().isoformat(),
            "row_count": int(len(subset)),
        }
        if requested_start_date:
            requested_start = pd.Timestamp(requested_start_date)
            if subset["date"].min() > requested_start + pd.Timedelta(days=7):
                failures.append(
                    f"{asset} starts after requested range: "
                    f"{subset['date'].min().date()} > {requested_start.date()}"
                )
        if requested_end_date:
            requested_end = pd.Timestamp(requested_end_date)
            if subset["date"].max() < requested_end - pd.Timedelta(days=7):
                failures.append(
                    f"{asset} ends before requested range: "
                    f"{subset['date'].max().date()} < {requested_end.date()}"
                )
    if extreme_rows:
        message = f"found {len(extreme_rows)} adjusted-close returns above threshold"
        if extreme_return_action == "fail":
            failures.append(message)
        else:
            warnings.append(message)
    # Missing calendar rows are reported, never filled. Weekends are excluded by union calendar.
    union_dates = pd.Index(sorted(data["date"].unique()))
    missing_runs: dict[str, int] = {}
    for asset in expected_assets:
        present = set(data.loc[data["asset_code"] == asset, "date"])
        flags = [date not in present for date in union_dates]
        longest = current = 0
        for flag in flags:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        missing_runs[asset] = longest
        if longest > max_consecutive_missing:
            failures.append(
                f"{asset} has {longest} consecutive missing common-calendar rows"
            )
        elif longest:
            warnings.append(f"{asset} has an isolated missing run of {longest} rows")
    report = {
        "status": "failed" if failures else "passed",
        "row_count": int(len(data)),
        "asset_count": int(data["asset_code"].nunique()),
        "duplicate_count": duplicate_count,
        "missing_value_counts": missing_counts,
        "ohlc_high_errors": high_errors,
        "ohlc_low_errors": low_errors,
        "negative_volume_count": negative_volume,
        "requested_date_range": [requested_start_date, requested_end_date],
        "asset_ranges": ranges,
        "longest_missing_run_by_asset": missing_runs,
        "extreme_return_threshold": extreme_return_threshold,
        "extreme_returns": extreme_rows,
        "warnings": sorted(set(warnings)),
        "failures": failures,
    }
    if failures:
        raise DataValidationError("; ".join(failures))
    return report
