from pathlib import Path
import pandas as pd
from .data_loader import load_ohlcv_csv

STOCK_SOURCE_PATH = "sample_data/sample_ohlcv.csv"
BENCHMARK_SOURCE_PATH = "sample_data/sample_vn30.csv"
OUTPUT_PATH = "data/processed/labels.parquet"

MAIN_TARGET = "forward_relative_return_5d"


FORWARD_HORIZON_SHORT = 1
FORWARD_HORIZON_MEDIUM = 5
FORWARD_HORIZON_LONG = 10

def build_forward_labels(data: pd.DataFrame,
) -> pd.DataFrame:
    """Create forward-looking return labels."""

    labels = (
        data
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
        .copy()
    )

    future_adjusted_close_1d = (
        labels
        .groupby("ticker")["adjusted_close"]
        .shift(-FORWARD_HORIZON_SHORT)
    )

    labels["forward_return_1d"] = (
        future_adjusted_close_1d/labels["adjusted_close"] -1
    )

    future_adjusted_close_5d = (
        labels
        .groupby("ticker")["adjusted_close"]
        .shift(-FORWARD_HORIZON_MEDIUM)
    )
    
    labels["forward_return_5d"] = (
        future_adjusted_close_5d/labels["adjusted_close"] -1
    )
    
    future_adjusted_close_10d = (
        labels
        .groupby("ticker")["adjusted_close"]
        .shift(-FORWARD_HORIZON_LONG)
    )
    
    labels["forward_return_10d"] = (
        future_adjusted_close_10d/labels["adjusted_close"] -1
    )

    return labels

def build_vn30_forward_returns(benchmark_data : pd.DataFrame,) -> pd.DataFrame:
    """Create forward returns for the VN30 benchmark."""

    benchmark = (
        benchmark_data
        .sort_values("date")
        .reset_index(drop = True)
        .copy()
    )

    future_vn30_close_1d = (
        benchmark["adjusted_close"]
        .shift(-FORWARD_HORIZON_SHORT)
    )
    
    benchmark["vn30_forward_return_1d"] = (
        future_vn30_close_1d/benchmark["adjusted_close"] -1
    )

    future_vn30_close_5d = (
        benchmark["adjusted_close"]
        .shift(-FORWARD_HORIZON_MEDIUM)
    )
    
    benchmark["vn30_forward_return_5d"] = (
        future_vn30_close_5d/benchmark["adjusted_close"] -1
    )
    
    future_vn30_close_10d = (
        benchmark["adjusted_close"]
        .shift(-FORWARD_HORIZON_LONG)
    )
    
    benchmark["vn30_forward_return_10d"] = (
        future_vn30_close_10d/benchmark["adjusted_close"] -1
    )
    
    return benchmark
    
def add_relative_forward_labels(stock_labels: pd.DataFrame, benchmark_labels: pd.DataFrame,) -> pd.DataFrame: 
    """Add stock returns relative to the VN30 benchmark"""

    benchmark_columns = [
        "date",
        "vn30_forward_return_1d",
        "vn30_forward_return_5d",
        "vn30_forward_return_10d",
    ]

    combined = stock_labels.merge(
        benchmark_labels[benchmark_columns],
        on = "date",
        how = "left",
        validate = "many_to_one",
    )

    combined["forward_relative_return_1d"] = (
        combined["forward_return_1d"] - combined["vn30_forward_return_1d"]
    )

    combined["forward_relative_return_5d"] = (
        combined["forward_return_5d"] - combined["vn30_forward_return_5d"]
    )

    combined["forward_relative_return_10d"] = (
        combined["forward_return_10d"] - combined["vn30_forward_return_10d"]
    )

    return combined


def main() -> None: 
    """Build, save, and verify the forward-label dataset"""

    stock_data = load_ohlcv_csv(STOCK_SOURCE_PATH)
    benchmark_data = pd.read_csv(
        BENCHMARK_SOURCE_PATH,
        parse_dates = ["date"],
    )

    stock_labels = build_forward_labels(stock_data)
    
    benchmark_labels = build_vn30_forward_returns(benchmark_data)

    labels = add_relative_forward_labels( 
        stock_labels,
        benchmark_labels,
    )

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents = True, 
        exist_ok= True,
    )

    labels.to_parquet(output_path, index = False,) 

    reloaded_labels = pd.read_parquet(output_path)

    print("Label generation completed.")
    print(f"Rows: {len(reloaded_labels)}")
    print(
        f"Tickers: "
        f"{reloaded_labels['ticker'].nunique()}"
    )
    print(f"Main target: {MAIN_TARGET}")
    print(
        f"Valid main-target rows: "
        f"{reloaded_labels[MAIN_TARGET].notna().sum()}"
    )
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    