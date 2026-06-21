from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "value_traded",
}


def load_ohlcv_csv(file_path: str) -> pd.DataFrame:
    """Load and standardize daily OHLCV stock data."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = pd.read_csv(path)

    data.columns = [
        column.strip().lower()
        for column in data.columns
    ]

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data = (
        data
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    return data


if __name__ == "__main__":
    sample_path = "sample_data/sample_ohlcv.csv"

    dataframe = load_ohlcv_csv(sample_path)

    print(dataframe.head())
    print()
    dataframe.info()
    print()
    print(f"Rows loaded: {len(dataframe)}")
    print(f"Tickers: {dataframe['ticker'].nunique()}")
    print(
        f"Date range: {dataframe['date'].min()} "
        f"to {dataframe['date'].max()}"
    )