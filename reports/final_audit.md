# Final Audit

## 1. Audit Purpose

This file records the final reproducibility and report audit for the VN30 long-only machine learning framework.

The goal of the audit is to confirm that the repository contains the final source code, report tables, figures, methodology write-up, final results write-up, and manual checks needed to support the project conclusions.

## 2. Repository State

The latest completed project commits before this audit were:

- `c4428e2 Add final results report`
- `512069e Add project methodology`
- `a4fe94c Add result visualization figures`
- `3631e0c Add forecast horizon testing`
- `d22aa80 Add feature ablation testing`
- `3dc4715 Add herding-aware risk control`
- `94ae647 Add price-limit-aware backtest comparison`
- `3b20c5e Add backtester with transaction costs`

The tracked repository was clean before the final audit commit.

Generated local data files remain untracked by design. These include raw data files and reproducible parquet outputs in:

- `data/raw/`
- `data/processed/`

These files are excluded from the committed research artifact because they are generated outputs rather than source code or final report artifacts.

## 3. Compilation Audit

All Python files were compiled with:

`Get-ChildItem -Recurse -Filter *.py | ForEach-Object { py -m py_compile $_.FullName }`

The command completed without Python compilation errors.

## 4. Final Script Rerun

The final report-producing scripts were rerun:

- `py -m src.horizon_tests`
- `py -m src.visualize_results`

The horizon script regenerated:

`reports/tables/horizon_results.csv`

The visualization script regenerated the final figure set in:

`reports/figures/`

## 5. Horizon Test Verification

The final horizon manual check was rerun:

`py .\tests\manual_check_horizon_tests.py`

The check confirmed:

- Prediction horizons: `[1, 5, 10]`
- Result horizons: `[1, 5, 10]`
- One result row per horizon
- Required result columns present
- Hit rates valid
- Sharpe values valid
- Drawdowns valid
- Turnover valid
- Best IC horizon: `10`
- Best Sharpe horizon: `10`
- Best final return horizon: `10`
- `10d is best on core metrics: True`

The final horizon result supports the conclusion that the 10-day forecast horizon is the strongest tested horizon in this experiment.

## 6. Visualization Verification

The final visualization manual check was rerun:

`py .\tests\manual_check_visualize_results.py`

The check confirmed that all expected figures exist and are non-empty.

The verified final figures are:

- `cumulative_after_cost_active_return.png`
- `active_drawdown.png`
- `portfolio_turnover.png`
- `top_gradient_boosting_feature_importance.png`
- `ablation_diagnostic_sharpe.png`
- `horizon_diagnostic_sharpe.png`
- `rolling_diagnostic_sharpe.png`
- `horizon_rank_ic.png`

## 7. Final Report Artifacts

The final report folder contains:

- `reports/methodology.md`
- `reports/final_results.md`
- `reports/model_report.md`
- `reports/data_quality_report.md`
- `reports/tables/ablation_results.csv`
- `reports/tables/horizon_results.csv`
- `reports/figures/`

The main final interpretation is documented in:

`reports/final_results.md`

The project methodology is documented in:

`reports/methodology.md`

## 8. Final Audit Conclusion

The project passes the final audit.

The repository contains a reproducible VN30 long-only machine learning research framework with:

- Data ingestion scripts
- Feature engineering pipeline
- Label construction
- Walk-forward validation
- Linear, tree-based, classification, and baseline model testing
- Risk model
- Constrained long-only optimizer
- Transaction-cost-aware backtester
- Price-limit-aware execution rules
- Herding-aware risk-control mode
- Feature ablation testing
- Forecast horizon testing
- Final result figures
- Methodology write-up
- Final results write-up
- Manual verification scripts

The project should be presented as a research framework rather than a deployable live trading strategy. The strongest tested forecast horizon is 10 days, while the original 5-day horizon remains a strong baseline.

