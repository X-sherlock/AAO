from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asset_allocation.data_processing import load_processed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a frozen real-data QA report")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    dataset_id = config["dataset"]["id"]
    raw_path = PROJECT_ROOT / config["dataset"].get(
        "raw_path", f"data/raw/{dataset_id}"
    )
    processed_path = PROJECT_ROOT / config["dataset"].get(
        "processed_path", f"data/processed/{dataset_id}"
    )
    raw = pd.read_parquet(raw_path / "ohlcv.parquet")
    processed = load_processed_dataset(processed_path)
    prices = raw.pivot(index="date", columns="asset_code", values="adjusted_close")
    volumes = raw.pivot(index="date", columns="asset_code", values="volume")
    closes = raw.pivot(index="date", columns="asset_code", values="close")
    daily_returns = prices.pct_change()
    weekly_returns = pd.DataFrame(
        processed.returns, index=pd.to_datetime(processed.dates), columns=processed.assets
    )
    normalized_prices = prices / prices.iloc[0]
    drawdowns = 1.0 - prices / prices.cummax()
    average_correlation = pd.Series(
        [
            matrix[~np.eye(len(processed.assets), dtype=bool)].mean()
            for matrix in processed.correlation
        ],
        index=pd.to_datetime(processed.dates),
    )
    spy, ief = processed.assets.index("SPY"), processed.assets.index("IEF")
    stock_bond = pd.Series(
        processed.correlation[:, spy, ief], index=pd.to_datetime(processed.dates)
    )
    feature_map = {
        name: index for index, name in enumerate(processed.feature_names)
    }
    credit = pd.Series(
        processed.features[:, feature_map["credit_pressure_hyg_ief"]],
        index=pd.to_datetime(processed.dates),
    )
    ranges = (
        raw.groupby("asset_code")["date"]
        .agg(["min", "max", "count"])
        .astype({"count": int})
    )
    split_rows = processed.split_manifest["splits"]
    report = {
        "dataset_id": dataset_id,
        "interpretation": "data readiness report; not a strategy performance result",
        "asset_ranges": {
            asset: {
                "start": row["min"].date().isoformat(),
                "end": row["max"].date().isoformat(),
                "rows": int(row["count"]),
            }
            for asset, row in ranges.iterrows()
        },
        "raw_missing_values": {
            key: int(value) for key, value in raw.isna().sum().items()
        },
        "daily_return_distribution": daily_returns.describe().to_dict(),
        "weekly_return_distribution": weekly_returns.describe().to_dict(),
        "processed_sample_count": int(len(processed.dates)),
        "feature_missing_ratio": float(np.isnan(processed.features).mean()),
        "splits": split_rows,
    }
    output_dir = PROJECT_ROOT / config.get(
        "report", {}
    ).get("output_dir", f"reports/{dataset_id}/data_quality")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            'report plots require: python -m pip install -e ".[report]"'
        ) from exc
    figure, axes = plt.subplots(4, 3, figsize=(18, 18), constrained_layout=True)
    normalized_prices.plot(ax=axes[0, 0], legend=False, title="Adjusted price / NAV curves")
    daily_returns.plot.hist(
        ax=axes[0, 1], bins=80, alpha=0.25, legend=False, title="Daily return distribution"
    )
    weekly_returns.plot.hist(
        ax=axes[0, 2], bins=50, alpha=0.25, legend=False, title="Weekly return distribution"
    )
    daily_returns.rolling(60).std().plot(
        ax=axes[1, 0], legend=False, title="60-day rolling volatility"
    )
    average_correlation.plot(ax=axes[1, 1], title="Average correlation")
    stock_bond.plot(ax=axes[1, 2], title="SPY-IEF correlation")
    (closes * volumes).apply(np.log1p).plot(
        ax=axes[2, 0], legend=False, title="log(1 + dollar volume proxy)"
    )
    credit.plot(ax=axes[2, 1], title="HYG vs IEF credit pressure")
    drawdowns.plot(ax=axes[2, 2], legend=False, title="Asset drawdowns")
    axes[3, 0].axis("off")
    table = axes[3, 0].table(
        cellText=[
            [asset, row["min"].date(), row["max"].date(), int(row["count"])]
            for asset, row in ranges.iterrows()
        ],
        colLabels=["Asset", "First", "Last", "Rows"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    axes[3, 1].bar(
        ["raw missing", "feature missing"],
        [int(raw.isna().sum().sum()), int(np.isnan(processed.features).sum())],
    )
    axes[3, 1].set_title("Missing values")
    axes[3, 2].plot(pd.to_datetime(processed.dates), np.ones(len(processed.dates)))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    first_split = split_rows[0]
    for color, label in zip(colors, ("train", "validation", "test")):
        axes[3, 2].axvspan(
            pd.Timestamp(first_split[f"{label}_start"]),
            pd.Timestamp(first_split[f"{label}_end"]),
            alpha=0.3,
            color=color,
            label=label,
        )
    axes[3, 2].set_title(f"Split intervals: {first_split['split_id']}")
    axes[3, 2].legend()
    figure.suptitle(f"{dataset_id} frozen-data readiness report", fontsize=16)
    figure.savefig(output_dir / "dataset_report.png", dpi=150)
    plt.close(figure)
    print(output_dir)


if __name__ == "__main__":
    main()
