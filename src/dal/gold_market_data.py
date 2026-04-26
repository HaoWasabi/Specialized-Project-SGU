from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS: dict[str, str] = {
    "Gold": "GC=F",
    "DXY": "DX-Y.NYB",
    "Oil": "CL=F",
    "S&P500": "^GSPC",
    "Bond_10Y": "^TNX",
}


class GoldMarketDataProvider:
    def __init__(self, tickers: Mapping[str, str] | None = None) -> None:
        self.tickers = dict(tickers or DEFAULT_TICKERS)
        self.csv_path = Path(__file__).resolve().parents[1] / "gia_vang_benchmark.csv"

    def fetch_benchmark(self, start_date: str) -> pd.DataFrame:
        raw = yf.download(
            list(self.tickers.values()),
            start=start_date,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError("Không tải được dữ liệu benchmark từ yfinance.")

        close_frame = self._extract_field_frame(raw, "Close")
        volume_frame = self._extract_field_frame(raw, "Volume")

        required_columns = {
            "Gold_Close": self.tickers["Gold"],
            "DXY": self.tickers["DXY"],
            "Oil": self.tickers["Oil"],
            "SP500": self.tickers["S&P500"],
            "Bond_Yield": self.tickers["Bond_10Y"],
            "Gold_Volume": self.tickers["Gold"],
        }

        frame = pd.DataFrame({"Date": raw.index})
        for output_name, ticker in required_columns.items():
            source_frame = volume_frame if output_name == "Gold_Volume" else close_frame
            if ticker not in source_frame.columns:
                raise KeyError(f"Thiếu cột {ticker} trong dữ liệu tải về từ yfinance.")
            frame[output_name] = source_frame[ticker].to_numpy()

        benchmark_df = frame.ffill().dropna().reset_index(drop=True)
        benchmark_df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
        return benchmark_df

    @staticmethod
    def _extract_field_frame(raw: pd.DataFrame, field: str) -> pd.DataFrame:
        field_frame = raw[field]
        if isinstance(field_frame, pd.Series):
            field_frame = field_frame.to_frame()
        return field_frame