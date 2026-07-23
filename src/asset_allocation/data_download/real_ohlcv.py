from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from asset_allocation.data_download.base import (
    MarketDataProvider,
    normalize_ohlcv_frame,
)
from asset_allocation.data_download.local_file import LocalFileProvider
from asset_allocation.exceptions import OptionalDependencyError


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def fetch(
        self,
        assets: tuple[str, ...],
        start_date: str | None,
        end_date: str | None,
        **parameters: Any,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise OptionalDependencyError(
                'online download requires: python -m pip install -e ".[download]"'
            ) from exc
        rows: list[pd.DataFrame] = []
        for asset in assets:
            try:
                raw = yf.download(
                    asset,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    timeout=int(parameters.get("timeout", 30)),
                    multi_level_index=False,
                )
            except Exception as exc:
                raise RuntimeError(f"yfinance download failed for {asset}: {exc}") from exc
            if raw is None or raw.empty:
                raise RuntimeError(f"yfinance returned no rows for required asset {asset}")
            raw = raw.reset_index()
            raw["asset_code"] = asset
            raw["currency"] = str(parameters.get("currency", "USD"))
            raw["source"] = self.name
            rows.append(raw)
        return normalize_ohlcv_frame(pd.concat(rows, ignore_index=True))


def provider_from_config(raw: dict[str, Any], project_root: Path) -> MarketDataProvider:
    provider = raw.get("provider", {})
    name = provider.get("name", "yfinance")
    if name == "yfinance":
        return YFinanceProvider()
    if name in {"local_csv", "local_parquet", "local_file"}:
        path = Path(provider["path"])
        if not path.is_absolute():
            path = project_root / path
        return LocalFileProvider(path)
    raise ValueError(f"unsupported provider: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/import and freeze real OHLCV")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    from asset_allocation.data_processing.dataset_freezer import freeze_raw_dataset

    output = freeze_raw_dataset(raw, config_path.parents[2])
    print(output)


if __name__ == "__main__":
    main()
