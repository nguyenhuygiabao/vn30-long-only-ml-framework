# Methodology

## 1. Project Objective

This project builds a VN30 long-only machine learning framework for ranking large-cap Vietnamese equities and testing whether predictive signals can be converted into a constrained portfolio after transaction costs and Vietnam-specific execution frictions.

The objective is not to claim a live trading strategy but to build a research-grade pipeline that connects data ingestion, feature engineering, supervised learning, walk-forward validation, portfolio construction, transaction-cost-aware backtesting, and diagnostic robustness tests.

The framework focuses on cross-sectional stock selection inside the VN30 universe. For each trading date, the model estimates which stocks are likely to outperform the VN30 benchmark over a forward horizon. The portfolio then selects and weights stocks subject to long-only constraints, turnover controls, and issuer-group concentration limits.

## 2. Investment Universe

The investment universe is the current VN30 constituent list used in this project. It contains 30 Vietnamese large-cap equities across banks, real estate, consumer, energy, industrials, brokerage, and other major sectors.

The project uses a static VN30 universe file stored in:

`config/vn30_universe.csv`

This file contains ticker-level metadata, including issuer-group labels. Issuer grouping is used to control concentration risk. For example, related companies in the Vingroup group are capped together rather than treated as fully independent exposures.

A static universe is simple and transparent, but it has one limitation: it does not fully account for historical VN30 membership changes. Therefore, the results should be interpreted as a research test on the selected VN30 universe rather than a fully point-in-time index membership simulation.

## 3. Market Data

The main market data source is daily OHLCV data downloaded with `vnstock`. The raw data is stored locally in:

`data/raw/vnstock/vn30_ohlcv.csv`

The dataset contains daily open, high, low, close, volume, and value-related information for the VN30 tickers. The current dataset covers the period from 2020-01-02 to 2026-06-30 and contains 46,863 rows across 30 tickers.

The project uses daily end-of-day data. It does not use intraday order-book data, tick data, or market-by-order data. Therefore, the framework is designed for medium-frequency research rather than high-frequency trading.

## 4. Label Construction

The project creates forward relative-return labels. For each stock and each date, the stock's future return is compared with the VN30 benchmark's future return over the same horizon.

The main label family is:

- 1-day forward relative return
- 5-day forward relative return
- 10-day forward relative return

The original baseline design used the 5-day label:

`forward_relative_return_5d`

This label asks whether a stock outperforms the VN30 benchmark over the next five trading days. Later horizon testing compared 1-day, 5-day, and 10-day labels. The horizon experiment found that the 10-day horizon produced the strongest diagnostic performance in this dataset, while the 5-day horizon remains a useful baseline.

Labels are constructed using future returns, but all model features are constructed using information available before the prediction date.

## 5. Feature Engineering

The feature pipeline combines several groups of features.

Momentum and return features measure recent price behavior, including simple returns, log returns, rolling returns, and medium-term momentum.

Risk features measure recent volatility and drawdown. These help the model identify stocks with unstable or stressed price behavior.

Liquidity features measure trading activity, value traded, dollar-volume ranks, abnormal volume, and illiquidity. These features are important in Vietnam because liquidity conditions affect whether a model signal can realistically be traded.

Price-limit features capture Vietnam-specific market microstructure. The framework estimates ceiling and floor prices, distance to price limits, whether stocks hit price limits, whether stocks closed at price limits, and whether price-limit pressure is broad across the universe.

Herding features measure market-wide crowding behavior. These include return dispersion, percent of stocks moving up or down together, rolling average pairwise correlation, market-direction agreement, and herding index features.

The combined feature dataset is saved locally as:

`data/processed/features_combined.parquet`

Generated parquet files are not committed to Git because they are reproducible outputs rather than source files.

## 6. Prediction Models

The project tests several model families:

- Linear models
- Tree-based regression models
- Classification models
- Baseline portfolios

The main predictive model used in later portfolio testing is gradient boosting. Gradient boosting was selected because it produced stronger rank-ordering performance than the linear models in the walk-forward tests.

The model is evaluated mainly as a ranking model, not only as a point forecast model. This is appropriate because the portfolio construction step uses predicted returns to rank stocks and select the top candidates.

## 7. Walk-Forward Validation

The project uses walk-forward validation instead of random train-test splitting.

This matters because financial data is time-ordered. Random splitting would leak future market regimes into the training set and produce overly optimistic results.

Each walk-forward window trains on past data, validates on a later period, and tests on a still-later period. This structure better resembles the actual research workflow of training a model using only information available before the test period.

The project also uses purging logic around train, validation, and test boundaries to reduce leakage from overlapping forward-return labels.

## 8. Portfolio Construction

Predicted returns are converted into portfolio weights using a constrained long-only optimizer.

The core portfolio rules are:

- Long-only positions
- Top-ranked stock selection
- Maximum single-stock weight
- Maximum turnover
- Issuer-group concentration cap
- Optional herding-aware risk mode

The normal optimizer selects stocks based on model predictions while respecting portfolio constraints.

The herding-aware optimizer reduces exposure during high-herding market states. This risk-control mode is designed to reduce crowding and concentration when many stocks are moving together.

Issuer-group caps are especially important in the VN30 universe because related companies can appear as separate tickers while sharing common business exposure.

## 9. Transaction Costs and Execution Frictions

The backtester includes transaction costs:

- Commission cost
- Slippage cost
- Liquidity penalty

The framework also tests price-limit-aware execution. In Vietnam, stocks can hit daily ceiling or floor prices. A simple backtest may assume trades can always be executed, but this can be unrealistic.

The price-limit-aware execution mode applies rules such as:

- Blocking buys when a stock closes at the ceiling price
- Blocking sells when a stock closes at the floor price
- Recomputing feasible trades after execution restrictions

This makes the backtest more conservative and more connected to Vietnam’s actual trading environment.

## 10. Robustness Tests

The project includes two major robustness tests.

First, feature ablation testing removes one feature group at a time. This checks whether model performance depends too heavily on one feature family. The ablation results showed that volume/liquidity, price-limit, and risk features are especially important. Herding features contributed less to raw prediction performance, but herding still remains useful as a portfolio-level risk-control concept.

Second, forecast horizon testing compares 1-day, 5-day, and 10-day labels. The 1-day horizon performed poorly and appeared noisy. The 5-day horizon performed strongly. The 10-day horizon performed best across rank IC, hit rate, diagnostic Sharpe, drawdown, turnover, and final cumulative after-cost active return.

## 11. Bias Controls

Several controls are used to reduce common backtest biases.

The model uses walk-forward validation instead of random splits.

Feature construction is past-only. Rolling risk features use lagged returns so that same-day or future information is not accidentally included.

Forward labels are kept separate from model features.

Transaction costs are included.

Liquidity penalties are included.

Price-limit-aware execution is tested separately from normal execution.

Generated data files are not committed as source code. The repository commits source files, report tables, and final figures, while large reproducible parquet outputs remain local.

## 12. Main Limitations

The framework still has important limitations.

The VN30 universe is treated as a static universe, so the test does not fully model historical index membership changes.

The backtest uses overlapping forward returns. Therefore, cumulative active returns and diagnostic Sharpe values should be interpreted as research diagnostics, not as final live-trading performance statistics.

The data is daily OHLCV data, so the project does not model intraday queue priority, order-book depth, hidden liquidity, or high-frequency execution.

The model does not include fundamental data, news, analyst forecasts, ownership flows, or macroeconomic variables.

The project does not claim to overcome information asymmetry in the Vietnamese stock market. Instead, it builds a disciplined public-data framework that accounts for several Vietnam-specific frictions.

## 13. Summary

The methodology is designed to be realistic enough for research while remaining transparent and reproducible. The project does not only train a model; it connects prediction, validation, constrained optimization, transaction costs, execution frictions, feature ablation, forecast-horizon testing, and final visualization.

The main research conclusion is that medium-horizon stock selection inside the VN30 universe is more promising than 1-day prediction. In this experiment, the 10-day forecast horizon is the strongest tested horizon, while the 5-day horizon remains the original baseline benchmark.
