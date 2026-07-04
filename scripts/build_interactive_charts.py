from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "interactive"

BACKTEST_RETURNS_PATH = DATA_DIR / "backtest_returns.parquet"


SCENARIO_ORDER = [
    ("herding_aware", "normal"),
    ("herding_aware", "price_limit_aware"),
    ("normal", "normal"),
    ("normal", "price_limit_aware"),
]

SCENARIO_LABELS = {
    ("herding_aware", "normal"): "Herding-aware / Normal",
    ("herding_aware", "price_limit_aware"): "Herding-aware / Price-limit aware",
    ("normal", "normal"): "Normal / Normal",
    ("normal", "price_limit_aware"): "Normal / Price-limit aware",
}

SCENARIO_COLORS = {
    ("herding_aware", "normal"): "#60a5fa",
    ("herding_aware", "price_limit_aware"): "#fbbf24",
    ("normal", "normal"): "#34d399",
    ("normal", "price_limit_aware"): "#f472b6",
}


def read_backtest_returns() -> pd.DataFrame:
    if not BACKTEST_RETURNS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BACKTEST_RETURNS_PATH}")

    data = pd.read_parquet(BACKTEST_RETURNS_PATH)
    data["date"] = pd.to_datetime(data["date"])

    return data.sort_values("date")


def scenario_group(
    data: pd.DataFrame,
    scenario_key: tuple[str, str],
) -> pd.DataFrame:
    optimization_mode, execution_mode = scenario_key

    group = data[
        data["optimization_mode"].eq(optimization_mode)
        & data["execution_mode"].eq(execution_mode)
    ].copy()

    return group.sort_values("date")


def base_layout(
    title: str,
    yaxis_title: str,
    data: pd.DataFrame,
    tickformat: str | None = None,
) -> dict:
    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(years=2)

    return {
        "title": {
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "color": "#eaf2ff",
                "size": 22,
            },
        },
        "template": "plotly_dark",
        "paper_bgcolor": "#050a14",
        "plot_bgcolor": "#050a14",
        "font": {
            "color": "#e5f2ff",
            "family": "Arial, Helvetica, sans-serif",
        },
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#0b1220",
            "bordercolor": "rgba(125, 211, 252, 0.45)",
            "font": {"color": "#e5f2ff"},
        },
        "height": 570,
        "margin": {
            "l": 78,
            "r": 380,
            "t": 58,
            "b": 32,
        },
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 0.70,
            "xanchor": "left",
            "x": 1.015,
            "font": {
                "color": "#e5f2ff",
                "size": 12,
            },
            "bgcolor": "rgba(5, 10, 20, 0)",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        "xaxis": {
            "title": {
                "text": "Date",
                "font": {"color": "#e5f2ff"},
            },
            "range": [start_date, end_date],
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.12)",
            "zeroline": True,
            "zerolinecolor": "rgba(255, 255, 255, 0.22)",
            "linecolor": "rgba(255, 255, 255, 0.30)",
            "tickfont": {"color": "#d8e6ff"},
            "rangeselector": {
                "x": 1.02,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
                "bgcolor": "#0b1220",
                "activecolor": "#1e40af",
                "bordercolor": "rgba(125, 211, 252, 0.35)",
                "font": {
                    "color": "#f8fafc",
                    "size": 12,
                },
                "buttons": [
                    {
                        "count": 3,
                        "label": "3m",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 6,
                        "label": "6m",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 1,
                        "label": "1y",
                        "step": "year",
                        "stepmode": "backward",
                    },
                    {
                        "count": 2,
                        "label": "2y",
                        "step": "year",
                        "stepmode": "backward",
                    },
                    {
                        "step": "all",
                        "label": "All",
                    },
                ],
            },
            "rangeslider": {
                "visible": True,
                "thickness": 0.045,
                "bgcolor": "#0b1220",
                "bordercolor": "rgba(255, 255, 255, 0.18)",
                "borderwidth": 1,
            },
        },
        "yaxis": {
            "title": {
                "text": yaxis_title,
                "font": {"color": "#e5f2ff"},
            },
            "tickformat": tickformat,
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.14)",
            "zeroline": True,
            "zerolinecolor": "rgba(255, 255, 255, 0.28)",
            "linecolor": "rgba(255, 255, 255, 0.30)",
            "tickfont": {"color": "#d8e6ff"},
        },
    }


def add_trace(
    fig: go.Figure,
    x: pd.Series | pd.Index,
    y: pd.Series,
    scenario_key: tuple[str, str],
    hover_label: str,
    hover_percent: bool = False,
) -> None:
    label = SCENARIO_LABELS[scenario_key]
    hover_value = "%{y:.2%}" if hover_percent else "%{y:.4f}"

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line={
                "color": SCENARIO_COLORS[scenario_key],
                "width": 1.7,
            },
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                f"{label}<br>"
                f"{hover_label}: {hover_value}"
                "<extra></extra>"
            ),
        )
    )


def range_button_style_override() -> str:
    return """
<style>
  .rangeselector .button rect {
    fill: #0b1220 !important;
    stroke: rgba(125, 211, 252, 0.45) !important;
  }

  .rangeselector .button text {
    fill: #f8fafc !important;
    font-weight: 700 !important;
  }

  .rangeselector .button.active rect {
    fill: #1e40af !important;
    stroke: #60a5fa !important;
  }

  .rangeselector .button.active text {
    fill: #ffffff !important;
    font-weight: 900 !important;
  }
</style>

<script>
  function applyRangeSelectorStyle() {
    const buttons = document.querySelectorAll(".rangeselector .button");

    buttons.forEach((button) => {
      const rect = button.querySelector("rect");
      const text = button.querySelector("text");
      const isActive = button.classList.contains("active");

      if (rect) {
        rect.style.fill = isActive ? "#1e40af" : "#0b1220";
        rect.style.stroke = isActive ? "#60a5fa" : "rgba(125, 211, 252, 0.45)";
      }

      if (text) {
        text.style.fill = "#ffffff";
        text.style.fontWeight = isActive ? "900" : "700";
      }
    });
  }

  window.addEventListener("load", () => {
    applyRangeSelectorStyle();

    const observer = new MutationObserver(() => {
      applyRangeSelectorStyle();
    });

    observer.observe(document.body, {
      attributes: true,
      childList: true,
      subtree: true,
    });

    document.body.addEventListener("click", () => {
      setTimeout(applyRangeSelectorStyle, 30);
      setTimeout(applyRangeSelectorStyle, 120);
    });
  });
</script>
"""


def write_chart(fig: go.Figure, output_path: Path) -> Path:
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    html = html.replace("</head>", range_button_style_override() + "\n</head>")
    output_path.write_text(html, encoding="utf-8", newline="\n")

    return output_path


def build_cumulative_return(data: pd.DataFrame) -> Path:
    fig = go.Figure()

    for scenario_key in SCENARIO_ORDER:
        group = scenario_group(data, scenario_key)

        if group.empty:
            continue

        add_trace(
            fig,
            group["date"],
            group["cumulative_after_cost_active_return"],
            scenario_key,
            "Cumulative return",
            hover_percent=False,
        )

    fig.update_layout(
        **base_layout(
            "Interactive Cumulative After-Cost Active Return",
            "Cumulative after-cost active return",
            data,
        )
    )

    return write_chart(fig, OUTPUT_DIR / "interactive_cumulative_return.html")


def build_drawdown(data: pd.DataFrame) -> Path:
    fig = go.Figure()

    for scenario_key in SCENARIO_ORDER:
        group = scenario_group(data, scenario_key)

        if group.empty:
            continue

        curve = group["cumulative_after_cost_active_return"]
        group["active_drawdown"] = curve - curve.cummax()

        weekly_drawdown = (
            group.set_index("date")["active_drawdown"]
            .resample("W-FRI")
            .min()
            .dropna()
        )

        add_trace(
            fig,
            weekly_drawdown.index,
            weekly_drawdown,
            scenario_key,
            "Weekly drawdown",
            hover_percent=True,
        )

    fig.update_layout(
        **base_layout(
            "Interactive Weekly Active Drawdown",
            "Weekly active drawdown",
            data,
            tickformat=".0%",
        )
    )

    return write_chart(fig, OUTPUT_DIR / "interactive_active_drawdown.html")


def build_turnover(data: pd.DataFrame) -> Path:
    fig = go.Figure()

    for scenario_key in SCENARIO_ORDER:
        group = scenario_group(data, scenario_key)

        if group.empty:
            continue

        weekly_turnover = (
            group.set_index("date")["portfolio_turnover"]
            .resample("W-FRI")
            .mean()
            .dropna()
        )

        add_trace(
            fig,
            weekly_turnover.index,
            weekly_turnover,
            scenario_key,
            "Weekly turnover",
            hover_percent=True,
        )

    fig.update_layout(
        **base_layout(
            "Interactive Weekly Portfolio Turnover",
            "Weekly portfolio turnover",
            data,
            tickformat=".0%",
        )
    )

    return write_chart(fig, OUTPUT_DIR / "interactive_portfolio_turnover.html")


def build_rolling_sharpe(data: pd.DataFrame) -> Path:
    fig = go.Figure()

    rolling_window = 60
    annualization = 252 ** 0.5

    for scenario_key in SCENARIO_ORDER:
        group = scenario_group(data, scenario_key)

        if group.empty:
            continue

        rolling_mean = group["after_cost_return"].rolling(rolling_window).mean()
        rolling_std = group["after_cost_return"].rolling(rolling_window).std()

        group["rolling_sharpe"] = (rolling_mean / rolling_std) * annualization
        group = group.dropna(subset=["rolling_sharpe"])

        if group.empty:
            continue

        add_trace(
            fig,
            group["date"],
            group["rolling_sharpe"],
            scenario_key,
            "Rolling 60-day Sharpe",
            hover_percent=False,
        )

    fig.update_layout(
        **base_layout(
            "Interactive Rolling 60-Day Diagnostic Sharpe",
            "Rolling 60-day diagnostic Sharpe",
            data,
        )
    )

    return write_chart(fig, OUTPUT_DIR / "interactive_rolling_diagnostic_sharpe.html")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = read_backtest_returns()

    output_paths = [
        build_cumulative_return(data),
        build_drawdown(data),
        build_turnover(data),
        build_rolling_sharpe(data),
    ]

    print()
    print("Interactive charts generated:")
    for path in output_paths:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
