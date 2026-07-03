from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = REPORTS_DIR / "dashboard.html"


FIGURE_ORDER = [
    "cumulative_after_cost_active_return.png",
    "active_drawdown.png",
    "portfolio_turnover.png",
    "rolling_diagnostic_sharpe.png",
    "horizon_diagnostic_sharpe.png",
    "horizon_rank_ic.png",
    "ablation_diagnostic_sharpe.png",
    "top_gradient_boosting_feature_importance.png",
]


FIGURE_NOTES = {
    "cumulative_after_cost_active_return.png": {
        "meaning": "This is the main performance path. It shows whether the strategy grows after estimated trading costs.",
        "read": "Focus on the overall direction and the gap between lines. If the lines rise steadily, the backtest is strong. If one line is consistently above another, that version performs better.",
        "limit": "If the lines are very close, the graph is saying the different execution or optimization modes produce similar outcomes. We should later adjust this figure to make the comparison easier.",
    },
    "active_drawdown.png": {
        "meaning": "This shows how much performance falls from a previous high point.",
        "read": "The closer the line stays to zero, the less painful the strategy is. Deep downward spikes show periods when the strategy gave back previous gains.",
        "limit": "This graph is currently dense. The most useful part is the deepest drop and whether the strategy recovers quickly.",
    },
    "portfolio_turnover.png": {
        "meaning": "This shows how much the portfolio changes between rebalancing dates.",
        "read": "Higher turnover means more buying and selling. More trading usually means more costs and lower practical realism.",
        "limit": "If the lines are too close, the graph is not doing enough visual work. We should later change it into a clearer summary or rescale it.",
    },
    "rolling_diagnostic_sharpe.png": {
        "meaning": "This shows whether the strategy stays useful across time instead of only working in one lucky period.",
        "read": "Higher values mean better return compared with volatility in that rolling window. Long weak stretches are more concerning than short dips.",
        "limit": "Rolling metrics can be noisy, so the broad trend matters more than each small movement.",
    },
    "horizon_diagnostic_sharpe.png": {
        "meaning": "This compares which prediction horizon gives the best risk-adjusted backtest result.",
        "read": "The strongest horizon is the one with the highest value. In this project, the longer horizon is currently stronger than the 1-day horizon.",
        "limit": "This does not mean the best horizon will always stay best in future data.",
    },
    "horizon_rank_ic.png": {
        "meaning": "This checks whether the model ranks stocks in the right order for each forecast horizon.",
        "read": "Higher Rank IC means the model is better at putting future winners above future losers.",
        "limit": "A high ranking score supports the signal, but it still needs portfolio and cost testing.",
    },
    "ablation_diagnostic_sharpe.png": {
        "meaning": "This checks whether removing feature groups weakens performance.",
        "read": "If the full feature set performs best, the removed feature groups likely contain useful information.",
        "limit": "Small differences should not be over-interpreted. The key question is whether the ranking changes meaningfully.",
    },
    "top_gradient_boosting_feature_importance.png": {
        "meaning": "This shows which input variables the Gradient Boosting model uses most.",
        "read": "Features near the top have more influence on the model's predictions.",
        "limit": "Feature importance explains model usage, not guaranteed economic causality.",
    },
}


DISPLAY_NAMES = {
    "rank": "Rank",
    "ticker": "Ticker",
    "signal_date": "Signal date",
    "model": "Model",
    "model_score": "Model score",
    "weight": "Portfolio weight",
    "issuer_group": "Issuer group",
    "optimization_mode": "Optimization mode",
    "realized_forward_return": "Realized forward return",
    "forecast_horizon_days": "Forecast horizon days",
    "horizon_label": "Target label",
    "ablation_name": "Feature set tested",
    "evaluated_dates": "Evaluated dates",
    "feature_count": "Feature count",
    "average_rank_ic": "Average Rank IC",
    "average_top_5_hit_rate": "Average top-5 hit rate",
    "average_top_5_actual_return": "Average top-5 realized return",
    "average_selected_count": "Average selected stocks",
    "average_after_cost_return": "Average after-cost return",
    "return_volatility": "Return volatility",
    "diagnostic_sharpe": "Diagnostic Sharpe",
    "max_active_drawdown": "Max active drawdown",
    "average_turnover": "Average turnover",
    "maximum_turnover": "Maximum turnover",
    "final_cumulative_after_cost_active_return": "Final cumulative after-cost active return",
    "predicted_return": "Predicted return",
    "actual_return": "Realized forward return",
    "portfolio_turnover": "Portfolio turnover",
    "high_herding_day": "High-herding day",
}


PERCENT_COLUMNS = {
    "model_score",
    "weight",
    "realized_forward_return",
    "predicted_return",
    "actual_return",
    "average_top_5_hit_rate",
    "average_top_5_actual_return",
    "average_after_cost_return",
    "return_volatility",
    "max_active_drawdown",
    "average_turnover",
    "maximum_turnover",
    "portfolio_turnover",
}


REPORT_LINKS = [
    ("README", "../README.md"),
    ("Project Context", "../PROJECT_CONTEXT.md"),
    ("Report Index", "report_index.md"),
    ("Final Results", "final_results.md"),
    ("Methodology", "methodology.md"),
    ("Final Audit", "final_audit.md"),
    ("Model Report", "model_report.md"),
    ("Data Quality Report", "data_quality_report.md"),
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def format_number(value: object) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)):
        value = float(value)

        if abs(value) >= 1000:
            return f"{value:,.2f}"

        return f"{value:.6f}".rstrip("0").rstrip(".")

    return str(value)


def format_percent(value: object) -> str:
    if pd.isna(value):
        return ""

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()

    for column in formatted.columns:
        if column in PERCENT_COLUMNS:
            formatted[column] = formatted[column].map(format_percent)
        elif pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(format_number)

    formatted = formatted.rename(columns=DISPLAY_NAMES)
    return formatted


def table_html(title: str, df: pd.DataFrame, note: str = "") -> str:
    if df.empty:
        body = '<p class="muted">Table not found or empty.</p>'
    else:
        body = format_dataframe(df).to_html(
            index=False,
            escape=True,
            classes="data-table",
            border=0,
        )

    note_html = f'<p class="muted">{escape(note)}</p>' if note else ""

    return f"""
    <section class="panel">
      <h2>{escape(title)}</h2>
      {note_html}
      <div class="table-wrap">
        {body}
      </div>
    </section>
    """


def card_html(title: str, value: str, subtitle: str = "") -> str:
    subtitle_html = f'<div class="card-subtitle">{escape(subtitle)}</div>' if subtitle else ""

    return f"""
    <div class="card">
      <div class="card-title">{escape(title)}</div>
      <div class="card-value">{escape(value)}</div>
      {subtitle_html}
    </div>
    """


def best_row_value(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
) -> tuple[str, str]:
    if df.empty or label_column not in df.columns or value_column not in df.columns:
        return "N/A", "N/A"

    values = pd.to_numeric(df[value_column], errors="coerce")

    if values.dropna().empty:
        return "N/A", "N/A"

    index = values.idxmax()
    return str(df.loc[index, label_column]), format_number(values.loc[index])


def latest_stock_ranking() -> tuple[pd.DataFrame, str]:
    prediction_path = DATA_DIR / "tree_model_predictions.parquet"
    weight_path = DATA_DIR / "optimized_weights.parquet"

    if not prediction_path.exists():
        return pd.DataFrame(), "N/A"

    predictions = pd.read_parquet(prediction_path)

    if predictions.empty:
        return pd.DataFrame(), "N/A"

    if "model_name" in predictions.columns:
        gradient_boosting = predictions[predictions["model_name"].eq("gradient_boosting")]

        if not gradient_boosting.empty:
            predictions = gradient_boosting

    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    latest_date = predictions["date"].max()

    latest = (
        predictions[predictions["date"].eq(latest_date)]
        .sort_values("predicted_return", ascending=False)
        .reset_index(drop=True)
    )

    latest["rank"] = latest.index + 1

    ranking = latest[
        ["rank", "ticker", "predicted_return", "actual_return", "model_name"]
    ].rename(
        columns={
            "predicted_return": "model_score",
            "actual_return": "realized_forward_return",
            "model_name": "model",
        }
    )

    ranking["signal_date"] = latest_date.date().isoformat()

    if weight_path.exists():
        weights = pd.read_parquet(weight_path)

        if not weights.empty:
            weights = weights.copy()
            weights["date"] = pd.to_datetime(weights["date"])
            latest_weights = weights[weights["date"].eq(latest_date)].copy()

            if "optimization_mode" in latest_weights.columns:
                herding_aware = latest_weights[
                    latest_weights["optimization_mode"].eq("herding_aware")
                ]

                if not herding_aware.empty:
                    latest_weights = herding_aware

            weight_columns = [
                column
                for column in ["ticker", "weight", "issuer_group", "optimization_mode"]
                if column in latest_weights.columns
            ]

            ranking = ranking.merge(
                latest_weights[weight_columns],
                on="ticker",
                how="left",
            )

    if "weight" in ranking.columns:
        ranking["weight"] = ranking["weight"].fillna(0.0)

    columns = [
        column
        for column in [
            "rank",
            "ticker",
            "signal_date",
            "model",
            "model_score",
            "weight",
            "issuer_group",
            "optimization_mode",
            "realized_forward_return",
        ]
        if column in ranking.columns
    ]

    return ranking[columns].head(15), latest_date.date().isoformat()


def latest_portfolio_weights() -> tuple[pd.DataFrame, str]:
    weight_path = DATA_DIR / "optimized_weights.parquet"

    if not weight_path.exists():
        return pd.DataFrame(), "N/A"

    weights = pd.read_parquet(weight_path)

    if weights.empty:
        return pd.DataFrame(), "N/A"

    weights = weights.copy()
    weights["date"] = pd.to_datetime(weights["date"])
    latest_date = weights["date"].max()
    latest = weights[weights["date"].eq(latest_date)].copy()

    if "optimization_mode" in latest.columns:
        herding_aware = latest[latest["optimization_mode"].eq("herding_aware")]

        if not herding_aware.empty:
            latest = herding_aware

    latest = latest.sort_values("weight", ascending=False)

    columns = [
        column
        for column in [
            "ticker",
            "weight",
            "issuer_group",
            "predicted_return",
            "actual_return",
            "optimization_mode",
            "portfolio_turnover",
            "high_herding_day",
        ]
        if column in latest.columns
    ]

    return latest[columns], latest_date.date().isoformat()


def figure_html(filename: str) -> str:
    path = FIGURES_DIR / filename

    if not path.exists() or path.stat().st_size == 0:
        return ""

    title = filename.replace("_", " ").replace(".png", "").title()
    note = FIGURE_NOTES.get(
        filename,
        {
            "meaning": "This is one of the original research figures generated by the project.",
            "read": "Use the graph to inspect one part of model, portfolio, or backtest behavior.",
            "limit": "If the graph is visually crowded, it should be improved in the original visualization script.",
        },
    )

    return f"""
    <figure class="figure-card">
      <img src="figures/{escape(filename)}" alt="{escape(title)}">
      <figcaption>{escape(title)}</figcaption>
      <div class="figure-note">
        <p><strong>What this means:</strong> {escape(note["meaning"])}</p>
        <p><strong>How to read it:</strong> {escape(note["read"])}</p>
        <p><strong>Current limitation:</strong> {escape(note["limit"])}</p>
      </div>
    </figure>
    """


def glossary_html() -> str:
    rows = [
        (
            "Rank",
            "The position of a stock in the latest model ranking. Rank 1 means the model gives that stock the strongest score on the latest signal date.",
            "Lower rank number is better.",
        ),
        (
            "Ticker",
            "The stock code, such as FPT, VCB, VIC, or HPG.",
            "Used to identify each VN30 stock.",
        ),
        (
            "Signal date",
            "The date when the model ranking or portfolio signal was generated from the processed dataset.",
            "This is not automatically today's live trading date.",
        ),
        (
            "Model score / predicted return",
            "The model's estimated relative-return signal for a stock. It is best understood as a score for ranking stocks, not as a guaranteed future return.",
            "Higher is better.",
        ),
        (
            "Realized forward return",
            "The return that actually happened after the signal date over the target horizon. This is used to evaluate the model after the fact.",
            "This is not known when the model makes the prediction.",
        ),
        (
            "Portfolio weight",
            "The percentage of the research portfolio allocated to a stock. For example, 20% means one-fifth of the portfolio goes into that stock.",
            "Higher means the optimizer selected more of that stock.",
        ),
        (
            "Issuer group",
            "A grouping used to identify related companies, such as Vingroup names. This helps detect concentration risk.",
            "Useful for checking whether the portfolio is too concentrated.",
        ),
        (
            "Optimization mode",
            "The portfolio construction rule used after model ranking. For example, herding-aware mode tries to account for crowding or related-stock behavior.",
            "Different modes test different portfolio assumptions.",
        ),
        (
            "Forecast horizon",
            "How far ahead the model tries to predict. In this project, the tested horizons are 1 day, 5 days, and 10 days.",
            "The best horizon is the one with stronger ranking and backtest evidence.",
        ),
        (
            "Target label",
            "The exact future-return variable the model is trained to predict, such as forward_relative_return_5d or forward_relative_return_10d.",
            "It defines the prediction task.",
        ),
        (
            "Evaluated dates",
            "The number of dates included in the backtest or evaluation.",
            "More evaluated dates usually means more evidence.",
        ),
        (
            "Feature count",
            "The number of input variables used by the model.",
            "More features is not always better; ablation tests check whether features add value.",
        ),
        (
            "Rank IC",
            "Rank Information Coefficient. It measures whether the model ranked future winners above future losers. It focuses on ordering, not exact return prediction.",
            "Higher is better. Near zero means weak ranking ability. Negative means the ranking may be wrong-way.",
        ),
        (
            "Average Rank IC",
            "The average Rank IC across all evaluated dates.",
            "Higher means the ranking signal was more consistently useful.",
        ),
        (
            "Top-5 hit rate",
            "How often the model's top-ranked stocks ended up being among the stronger realized performers.",
            "Higher is better.",
        ),
        (
            "Average top-5 realized return",
            "The average future return of the stocks selected near the top of the ranking.",
            "Higher means the model's preferred stocks performed better afterward.",
        ),
        (
            "Average selected stocks",
            "The average number of stocks selected into the portfolio.",
            "This shows how concentrated or diversified the strategy is.",
        ),
        (
            "Average after-cost return",
            "The average return after estimated commission, slippage, and liquidity costs.",
            "Higher is better. This matters more than before-cost return.",
        ),
        (
            "Return volatility",
            "How much the strategy's returns fluctuate. Higher volatility means returns are less stable.",
            "Lower is usually more comfortable, but it must be compared with return.",
        ),
        (
            "Diagnostic Sharpe",
            "A simple risk-adjusted performance measure. It compares average return with return volatility.",
            "Higher is better. Negative means poor risk-adjusted performance.",
        ),
        (
            "Max active drawdown",
            "The worst fall from a previous peak in the active-return path.",
            "Closer to zero is better. Large negative values mean painful losing periods.",
        ),
        (
            "Average turnover",
            "The average amount of portfolio change between rebalancing dates.",
            "Lower is easier to trade. Very high turnover creates more trading costs.",
        ),
        (
            "Maximum turnover",
            "The largest single-period portfolio change in the backtest.",
            "High maximum turnover can signal unrealistic trading pressure.",
        ),
        (
            "Final cumulative after-cost active return",
            "The final compounded value of the strategy's active return after estimated trading costs.",
            "Higher is better, but it must be judged together with drawdown and turnover.",
        ),
        (
            "Feature ablation",
            "A test where one group of features is removed to see whether model performance gets worse.",
            "If removing a feature group hurts performance, that group likely adds useful information.",
        ),
        (
            "Full feature set",
            "The model version that uses all available feature groups.",
            "It should ideally outperform reduced versions.",
        ),
        (
            "Without herding",
            "A feature-ablation version where herding-related features are removed.",
            "If performance falls, herding features are useful.",
        ),
        (
            "Without price limit",
            "A feature-ablation version where Vietnam price-limit features are removed.",
            "If performance falls, price-limit features are useful.",
        ),
        (
            "Without risk",
            "A feature-ablation version where risk-related features are removed.",
            "If performance falls, risk features are useful.",
        ),
        (
            "Without volume/liquidity",
            "A feature-ablation version where volume and liquidity features are removed.",
            "If performance falls, liquidity features are useful.",
        ),
        (
            "Active return",
            "The strategy's return compared with the benchmark or active-return baseline used in the project.",
            "Positive active return means the strategy beats the baseline.",
        ),
        (
            "After-cost active return",
            "Active return after estimated trading costs.",
            "This is more realistic than before-cost active return.",
        ),
        (
            "Drawdown",
            "The fall from a previous high point.",
            "Large drawdowns mean the strategy can lose a lot before recovering.",
        ),
        (
            "Turnover",
            "How much the portfolio changes from one rebalance to the next.",
            "More turnover usually means more trading cost and lower live-trading realism.",
        ),
    ]

    body = "\n".join(
        f"""
        <tr>
          <td>{escape(term)}</td>
          <td>{escape(meaning)}</td>
          <td>{escape(direction)}</td>
        </tr>
        """
        for term, meaning, direction in rows
    )

    return f"""
    <section class="panel">
      <h2>Metric Glossary: What The Measured Data Means</h2>
      <p class="muted">
        This section explains the table columns and graph concepts in plain language.
        Read this if Rank IC, Sharpe, drawdown, turnover, or model score is unclear.
      </p>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Metric / concept</th>
              <th>Meaning</th>
              <th>How to interpret it</th>
            </tr>
          </thead>
          <tbody>
            {body}
          </tbody>
        </table>
      </div>
    </section>
    """


def report_links_html() -> str:
    links = []

    for label, href in REPORT_LINKS:
        target = (REPORTS_DIR / href).resolve()
        status = "available" if target.exists() else "missing"
        links.append(
            f'<a class="report-link {status}" href="{escape(href)}">{escape(label)}</a>'
        )

    return "\n".join(links)


def build_html() -> str:
    horizon_df = read_csv(TABLES_DIR / "horizon_results.csv")
    ablation_df = read_csv(TABLES_DIR / "ablation_results.csv")
    ranking_df, ranking_date = latest_stock_ranking()
    portfolio_df, portfolio_date = latest_portfolio_weights()

    best_horizon, best_horizon_sharpe = best_row_value(
        horizon_df,
        "forecast_horizon_days",
        "diagnostic_sharpe",
    )

    best_rank_ic_horizon, best_rank_ic = best_row_value(
        horizon_df,
        "forecast_horizon_days",
        "average_rank_ic",
    )

    best_feature_set, best_feature_sharpe = best_row_value(
        ablation_df,
        "ablation_name",
        "diagnostic_sharpe",
    )

    figure_cards = "\n".join(figure_html(filename) for filename in FIGURE_ORDER)

    figure_count = sum(
        1
        for filename in FIGURE_ORDER
        if (FIGURES_DIR / filename).exists() and (FIGURES_DIR / filename).stat().st_size > 0
    )

    cards = "\n".join(
        [
            card_html("Best horizon", best_horizon, f"Diagnostic Sharpe {best_horizon_sharpe}"),
            card_html("Best ranking quality", best_rank_ic_horizon, f"Rank IC {best_rank_ic}"),
            card_html("Best feature set", best_feature_set, f"Diagnostic Sharpe {best_feature_sharpe}"),
            card_html("Latest signal date", ranking_date, "latest processed model ranking"),
            card_html("Figures", f"{figure_count} / {len(FIGURE_ORDER)}", "original research figures"),
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VN30 Long-Only ML Research Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-soft: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --line: #374151;
      --accent: #38bdf8;
      --bad: #ef4444;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}

    header {{
      padding: 40px 48px 24px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, #0f172a 0%, #111827 60%, #1e293b 100%);
    }}

    header h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}

    header p {{
      margin: 0;
      color: var(--muted);
      max-width: 900px;
    }}

    main {{
      padding: 32px 48px 56px;
      max-width: 1500px;
      margin: 0 auto;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(5, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}

    .card, .panel, .figure-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
    }}

    .card {{
      padding: 20px;
    }}

    .card-title {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}

    .card-value {{
      font-size: 30px;
      font-weight: 700;
      color: var(--accent);
    }}

    .card-subtitle {{
      color: var(--muted);
      margin-top: 6px;
      font-size: 14px;
    }}

    .panel, .figure-card {{
      padding: 24px;
      margin-bottom: 24px;
    }}

    .panel h2 {{
      margin: 0 0 14px;
      font-size: 22px;
    }}

    .muted {{
      color: var(--muted);
    }}

    .table-wrap {{
      overflow-x: auto;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}

    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      white-space: nowrap;
    }}

    .data-table th {{
      color: var(--text);
      background: var(--panel-soft);
    }}

    .figure-card img {{
      display: block;
      width: 100%;
      max-height: 780px;
      object-fit: contain;
      border-radius: 12px;
      background: white;
    }}

    .figure-card figcaption {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }}

    .figure-note {{
      margin-top: 16px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      color: var(--muted);
      font-size: 15px;
    }}

    .figure-note p {{
      margin: 0 0 10px;
    }}

    .figure-note strong {{
      color: var(--text);
    }}

    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .report-link {{
      display: inline-block;
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--text);
      text-decoration: none;
      background: var(--panel-soft);
    }}

    .report-link:hover {{
      border-color: var(--accent);
    }}

    .report-link.missing {{
      color: var(--bad);
    }}

    footer {{
      color: var(--muted);
      padding-top: 16px;
      font-size: 13px;
    }}

    @media (max-width: 1100px) {{
      .cards {{
        grid-template-columns: repeat(2, minmax(160px, 1fr));
      }}
    }}

    @media (max-width: 560px) {{
      header, main {{
        padding-left: 20px;
        padding-right: 20px;
      }}

      .cards {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>VN30 Long-Only ML Research Dashboard</h1>
    <p>
      Static dashboard generated from the original project reports, tables, and figures.
      This page explains the latest model ranking, portfolio weights, backtest evidence,
      and current limitations. It is a research dashboard, not a live trading system.
    </p>
  </header>

  <main>
    <section class="cards">
      {cards}
    </section>

    <section class="panel">
      <h2>What This Dashboard Provides</h2>
      <p class="muted">
        This dashboard summarizes the model's latest processed VN30 ranking, the research portfolio selected from that ranking,
        the backtest evidence, and the limitations of the current framework.
      </p>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Provides</th>
              <th>Does not provide</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Latest processed VN30 stock ranking</td>
              <td>Live broker-ready trading orders</td>
            </tr>
            <tr>
              <td>Long-only research portfolio weights</td>
              <td>Guaranteed return or risk-free recommendation</td>
            </tr>
            <tr>
              <td>Backtest and robustness evidence</td>
              <td>Point-in-time VN30 membership guarantee</td>
            </tr>
            <tr>
              <td>Research signal based on local processed data</td>
              <td>Personalized investment advice</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    {table_html(
        "Latest VN30 Stock Ranking",
        ranking_df,
        "Stocks are ordered by the latest processed model score. Higher-ranked names are the stocks the model preferred on that signal date.",
    )}

    {table_html(
        "Latest Optimized Portfolio Weights",
        portfolio_df,
        f"Latest selected research portfolio on {portfolio_date}. The preferred display uses herding-aware optimization when available.",
    )}

    {table_html(
        "Forecast Horizon Results",
        horizon_df,
        "Compares whether the model works better with 1-day, 5-day, or 10-day prediction targets.",
    )}

    {table_html(
        "Feature Ablation Results",
        ablation_df,
        "Compares the full feature set with versions where feature groups are removed.",
    )}

    <section class="panel">
      <h2>Figures</h2>
      <p class="muted">
        These are the original research figures from reports/figures. The annotations explain how to read each graph.
        If a graph is visually crowded, the next step is to improve the original visualization script rather than creating duplicate graphs.
      </p>
    </section>

    {figure_cards}

    {glossary_html()}

    <section class="panel">
      <h2>What The Dashboard Means</h2>
      <p>
        The dashboard means that the project can generate a latest processed VN30 stock ranking,
        convert that ranking into a constrained long-only research portfolio, and evaluate whether the signal was historically useful.
      </p>
      <p>
        It does not mean the system is trade-ready. The current output is a research signal, not a broker-ready order list,
        not a guaranteed return forecast, and not personalized investment advice.
      </p>
    </section>

    <section class="panel">
      <h2>Report Links</h2>
      <p class="muted">Open the source reports for methodology, audit notes, and detailed results.</p>
      <div class="links">
        {report_links_html()}
      </div>
    </section>

    <footer>
      Generated locally from repository artifacts. Raw and processed data are intentionally kept out of Git.
    </footer>
  </main>
</body>
</html>
"""


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html()
    OUTPUT_PATH.write_text(html, encoding="utf-8", newline="\n")

    print()
    print("HTML dashboard generated:")
    print(f"  {OUTPUT_PATH}")
    print()
    print("Open with:")
    print("  start .\\reports\\dashboard.html")
    print()


if __name__ == "__main__":
    main()