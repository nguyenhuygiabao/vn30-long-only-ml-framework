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
    ("herding_aware", "normal"): "#0072B2",
    ("herding_aware", "price_limit_aware"): "#E69F00",
    ("normal", "normal"): "#009E73",
    ("normal", "price_limit_aware"): "#CC79A7",
}


def read_backtest_returns() -> pd.DataFrame:
    if not BACKTEST_RETURNS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BACKTEST_RETURNS_PATH}")

    data = pd.read_parquet(BACKTEST_RETURNS_PATH)
    data["date"] = pd.to_datetime(data["date"])

    return data


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


def date_menu(data: pd.DataFrame) -> dict:
    min_date = data["date"].min()
    end_date = data["date"].max()

    def window(
        months: int | None = None,
        years: int | None = None,
    ) -> list[str]:
        if months is not None:
            start_date = end_date - pd.DateOffset(months=months)
        elif years is not None:
            start_date = end_date - pd.DateOffset(years=years)
        else:
            start_date = min_date

        return [
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ]

    return {
        "type": "buttons",
        "direction": "right",
        "showactive": True,
        "x": 1.015,
        "xanchor": "left",
        "y": 0.83,
        "yanchor": "top",
        "pad": {"r": 0, "t": 0},
        "buttons": [
            {"label": "3m", "method": "relayout", "args": [{"xaxis.range": window(months=3)}]},
            {"label": "6m", "method": "relayout", "args": [{"xaxis.range": window(months=6)}]},
            {"label": "1y", "method": "relayout", "args": [{"xaxis.range": window(years=1)}]},
            {"label": "2y", "method": "relayout", "args": [{"xaxis.range": window(years=2)}]},
            {"label": "All", "method": "relayout", "args": [{"xaxis.range": window()}]},
        ],
    }


def base_layout(
    title: str,
    yaxis_title: str,
    data: pd.DataFrame,
    tickformat: str | None = None,
) -> dict:
    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(years=2)

    return {
        "title": {"text": title, "x": 0.02, "xanchor": "left"},
        "template": "plotly_white",
        "hovermode": "x unified",
        "height": 560,
        "margin": {"l": 78, "r": 380, "t": 58, "b": 26},
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 0.70,
            "xanchor": "left",
            "x": 1.015,
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        "updatemenus": [date_menu(data)],
        "xaxis": {
            "title": "Date",
            "range": [start_date, end_date],
            "rangeslider": {"visible": True, "thickness": 0.045},
        },
        "yaxis": {
            "title": yaxis_title,
            "tickformat": tickformat,
            "zeroline": True,
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
                "width": 1.5,
            },
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                f"{label}<br>"
                f"{hover_label}: {hover_value}"
                "<extra></extra>"
            ),
        )
    )


def write_chart(fig: go.Figure, output_path: Path) -> Path:
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )

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
