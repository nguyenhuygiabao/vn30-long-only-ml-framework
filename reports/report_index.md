# Report Index

This page links the main outputs of the VN30 Alpha Research Framework.

## Start here

- [HTML Dashboard](dashboard.html): visual dashboard with stock ranking, portfolio weights, figures, and metric glossary.
- [Project Context](../PROJECT_CONTEXT.md): reusable handoff prompt for continuing the project in a new coding session.
- [Final Results](Final Results.md): main findings, backtest results, horizon tests, ablation tests, figures, limitations, and conclusion.
- [Methodology](methodology.md): data, labels, features, models, validation, optimizer, backtester, transaction costs, and bias controls.
- [Final Audit](Final Audit.md): final repo state, script reruns, verification checks, and artifact audit.

## Supporting reports

- [Model Report](Model Report.md): model evaluation details.
- [Data Quality Report](data_quality_report.md): data coverage and quality checks.

## Result tables

- [Ablation Results](tables/Ablation Results.csv): feature-ablation results.
- [Horizon Results](tables/Horizon Results.csv): forecast-horizon results.

## Main figures

- [Top Gradient Boosting Feature Importance](figures/Top Gradient Boosting Feature Importance.png)
- [Ablation Diagnostic Sharpe](figures/Ablation Diagnostic Sharpe.png)
- [Horizon Diagnostic Sharpe](figures/Horizon Diagnostic Sharpe.png)
- [Horizon Rank IC](figures/Horizon Rank Ic.png)

## Current conclusion

The project should be presented as a transparent research framework, not a live trading strategy.

Current evidence suggests that the 10-day forward relative-return label is the strongest tested horizon, while the 5-day label remains a strong baseline. The full feature set performs best overall. Price-limit, risk, volume, and liquidity features matter more than herding features for raw predictive performance, while herding remains useful as portfolio-level risk control.

## Local viewing commands

- code .\reports\Final Results.md
- code .\reports\methodology.md
- code .\reports\Final Audit.md
- code .\reports\Report Index.md
- explorer .\reports\figures
- explorer .\reports\tables

## Latest dashboard additions

The dashboard now includes the following robustness tables:

- `reports/tables/Benchmark Results.csv`: ML strategy comparison against equal-weight and simple rule-based baselines.
- `reports/tables/Concentration Summary.csv`: latest concentration metrics including max single-name weight, HHI, effective position count, and top issuer-group exposure.
- `reports/tables/Issuer Group Exposure Latest.csv`: latest issuer-group exposure by optimization mode.
- `reports/tables/Latest Rank Diagnostic.csv`: latest predicted rank versus realized rank by ticker.
- `reports/tables/Horizon Sample Disclosure.csv`: overlapping-window disclosure and approximate non-overlapping period counts.

The dashboard page at `reports/dashboard.html` displays these diagnostics directly.
