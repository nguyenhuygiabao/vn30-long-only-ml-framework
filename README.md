# VN30 Long-Only Machine Learning Forecasting Framework

## Project Overview

This project develops a long-only machine learning research framework for selecting and allocating capital among VN30 stocks.

The framework is designed around features of the Vietnamese equity market, including:

* the absence of generally accessible short selling
* the HOSE daily price-limit mechanism
* liquidity and transaction-cost constraints
* possible herd-like market behavior
* difficulty executing trades when stocks are locked at ceiling or floor prices

The project is intended as a quantitative research and portfolio-construction framework. It is not presented as a guaranteed profitable trading system.

## Research Question

Can price, volume, volatility, liquidity, price-limit behavior, and market-wide herding proxies help rank VN30 stocks by their future returns relative to the VN30 index?

## Investment Universe

The intended investment universe is the historical membership of the VN30 index.

The initial development dataset contains synthetic observations for three tickers:

* FPT
* HPG
* VNM

The synthetic dataset is used only to test the data pipeline. It is not used to draw investment conclusions.

Historical VN30 membership will be required later to reduce survivorship bias.

## Prediction Target

The planned primary target is:

```text
forward_relative_return_5d
```

This is defined as:

```text
stock's next five-trading-day return
minus
VN30 index's next five-trading-day return
```

The framework predicts relative returns rather than raw stock prices.

## Market Constraints

### Long-Only Constraint

Portfolio weights cannot be negative.

The framework will not assume ordinary short selling is available.

### Position Constraints

The planned portfolio rules include:

* maximum weight per stock
* maximum portfolio turnover
* optional cash allocation
* liquidity-based position limits

### Price-Limit Constraint

The initial configuration assumes a normal HOSE daily price limit of 7 percent.

The future backtester will not automatically assume that:

* a stock locked at its ceiling price can be purchased
* a stock locked at its floor price can be sold

Official reference prices and exchange rounding rules should be used when suitable data becomes available.

### Herding Constraint

Herding cannot be directly observed from daily OHLCV data.

The project will therefore use market-wide herding proxies such as:

* cross-sectional return dispersion
* average pairwise return correlation
* market breadth
* percentage of VN30 stocks hitting ceiling or floor prices
* abnormal market-wide volume
* concentration of traded value

These variables will be described as proxies, not direct proof of investor herding.

## Planned Data Inputs

The intended input data includes:

* date
* ticker
* open price
* high price
* low price
* close price
* adjusted close price
* trading volume
* traded value
* VN30 index data
* historical VN30 membership
* corporate-action information where available

## Planned Feature Groups

The framework is expected to include:

* momentum
* short-term reversal
* realized volatility
* volume shocks
* liquidity
* relative strength
* market beta
* price-limit behavior
* herding proxies

## Planned Models

The first models will be deliberately simple:

* Ridge regression
* Elastic Net
* logistic regression
* Random Forest
* Gradient Boosting

More complicated models will only be considered after the simple models and baselines have been evaluated correctly.

## Validation Method

The framework will use time-ordered walk-forward validation.

Random train-test splitting will not be used because it can create look-ahead bias in financial time-series research.

Model evaluation will include:

* Spearman rank information coefficient
* top-stock hit rate
* top-minus-bottom return spread
* portfolio Sharpe ratio
* maximum drawdown
* turnover
* transaction-cost impact

## Baseline Strategies

Machine learning results will be compared with:

* equal-weight VN30
* momentum ranking
* reversal ranking
* low-volatility selection
* liquidity-adjusted momentum

A machine learning model will not be considered useful merely because it has good in-sample fit.

## Expected Model Output

After receiving properly formatted VN30 market data, the completed framework is expected to produce:

1. a ranking of eligible VN30 stocks by predicted forward relative return
2. predicted return or probability scores
3. long-only portfolio weights
4. buy, hold, or avoid research signals
5. ceiling and floor execution warnings
6. a market herding-risk indicator
7. estimated portfolio risk and expected return
8. a backtest report
9. comparisons with simple baseline strategies

## Current Project Status

Completed:

* repository structure
* Python package installation
* project configuration
* synthetic OHLCV sample dataset
* initial CSV data loader
* required-column validation
* date and ticker standardization

In progress:

* financial data-quality checks
* automated data-quality reporting

Not yet implemented:

* feature engineering
* prediction labels
* baseline strategies
* machine learning models
* walk-forward validation
* portfolio optimization
* transaction-cost modeling
* price-limit-aware execution
* herding-regime controls

## Known Limitations

The project currently has the following limitations:

* sample data is synthetic
* historical VN30 membership has not yet been added
* corporate actions are not yet verified
* official ceiling and floor prices are not yet included
* order-book data is unavailable
* herding variables will only be proxies
* transaction costs and market impact are not yet calibrated

## Repository Structure

```text
configs/       Project configuration
data/          Raw, processed, and external data
notebooks/     Research and visualization notebooks
reports/       Generated research reports
sample_data/   Small reproducible sample dataset
src/           Reusable Python source code
tests/         Automated tests
```

## Current Sample Input Schema

The current loader expects the following columns:

```text
date
ticker
open
high
low
close
adjusted_close
volume
value_traded
```

## Run the Current Data Loader

From the project root:

```powershell
python .\src\data_loader.py
```

Expected summary:

```text
Rows loaded: 30
Tickers: 3
Date range: 2024-01-02 to 2024-01-15
```

## Disclaimer

This project is for educational and quantitative research purposes. It does not provide investment advice or guarantee future trading performance.
