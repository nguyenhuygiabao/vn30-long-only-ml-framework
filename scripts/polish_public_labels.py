from pathlib import Path
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
INTERACTIVE = REPORTS / "interactive"
TABLES = REPORTS / "tables"


DISPLAY_LABELS = {
    # Ablation labels
    "all_features": "All features",
    "without_volume_liquidity": "Excluding volume/liquidity",
    "without_risk": "Excluding risk features",
    "without_herding": "Excluding herding features",
    "without_price_limit": "Excluding price-limit features",

    # ML scenario labels
    "ml_normal_normal": "ML: standard",
    "ml_normal_price_limit_aware": "ML: price-limit aware",
    "ml_herding_aware_normal": "ML: herding-aware",
    "ml_herding_aware_price_limit_aware": "ML: herding + price-limit aware",

    # Baseline labels
    "top5_momentum": "Top-5 momentum",
    "top5_reversal": "Top-5 reversal",
    "low_volatility_top10": "Low-volatility top 10",
    "equal_weight_all": "Equal-weight universe",

    # Common dashboard/card wording
    "Best Feature Set": "Best Ablation Variant",
    "BEST FEATURE SET": "BEST ABLATION VARIANT",
    "Best ML vs Naive Baseline": "ML vs Best Naive Baseline",
    "BEST ML VS NAIVE BASELINE": "ML VS BEST NAIVE BASELINE",
    "Interactive Weekly Active Drawdown": "Interactive Active-Return Drawdown",
    "Weekly active drawdown": "Active-return drawdown",
    "weekly active drawdown": "active-return drawdown",
}


TOKEN_REPLACEMENTS = {
    "rank ic": "Rank IC",
    "sharpe": "Sharpe",
    "ml": "ML",
    "vn30": "VN30",
    "hhi": "HHI",
    "adv": "ADV",
    "ohlcv": "OHLCV",
    "rsi": "RSI",
    "macd": "MACD",
    "atr": "ATR",
}


def pretty_label(value):
    if value is None:
        return ""

    text = str(value)
    if text in DISPLAY_LABELS:
        return DISPLAY_LABELS[text]

    text = text.replace("_", " ").replace("-", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.title()

    words = []
    for word in text.split():
        lower = word.lower()
        words.append(TOKEN_REPLACEMENTS.get(lower, word))

    return " ".join(words)


def replace_public_text_files():
    paths = []
    for pattern in ["*.html", "*.md"]:
        paths.extend(REPORTS.rglob(pattern))
    for extra in [ROOT / "README.md"]:
        if extra.exists():
            paths.append(extra)

    changed = []

    for path in sorted(set(paths)):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")

        original = text

        for raw, clean in DISPLAY_LABELS.items():
            text = text.replace(raw, clean)

        # Clean common snake_case labels that may appear outside the explicit mapping.
        # This intentionally avoids source code and CSV files.
        snake_candidates = set(re.findall(r"\b[a-z]+(?:_[a-z0-9]+){1,}\b", text))
        for candidate in sorted(snake_candidates, key=len, reverse=True):
            if candidate.startswith(("http_", "https_", "data_", "plotly_", "font_", "grid_")):
                continue
            if candidate in {"diagnostic_sharpe", "average_rank_ic", "average_top_5_hit_rate"}:
                continue
            text = text.replace(candidate, pretty_label(candidate))

        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")
            changed.append(path.relative_to(ROOT))

    return changed


def set_public_plot_style():
    plt.rcParams.update({
        "figure.figsize": (9.5, 5.4),
        "figure.dpi": 160,
        "savefig.dpi": 180,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 10,
    })


def save_horizon_figures():
    path = TABLES / "horizon_results.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path).sort_values("forecast_horizon_days").copy()
    df["horizon_label"] = df["forecast_horizon_days"].astype(int).astype(str) + "d"

    written = []

    if "diagnostic_sharpe" in df.columns:
        fig, ax = plt.subplots()
        ax.bar(df["horizon_label"], df["diagnostic_sharpe"])
        ax.axhline(0, linewidth=1)
        ax.set_title("Diagnostic Sharpe by forecast horizon")
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Diagnostic Sharpe")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = FIGURES / "horizon_diagnostic_sharpe.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out.relative_to(ROOT))

    if "average_rank_ic" in df.columns:
        fig, ax = plt.subplots()
        ax.bar(df["horizon_label"], df["average_rank_ic"])
        ax.axhline(0, linewidth=1)
        ax.set_title("Average Rank IC by forecast horizon")
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Average Rank IC")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = FIGURES / "horizon_rank_ic.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out.relative_to(ROOT))

    return written


def save_ablation_figure():
    path = TABLES / "ablation_results.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path).copy()
    if "ablation_name" not in df.columns or "diagnostic_sharpe" not in df.columns:
        return []

    df["label"] = df["ablation_name"].map(pretty_label)
    df = df.sort_values("diagnostic_sharpe", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(df["label"], df["diagnostic_sharpe"])
    ax.axvline(0, linewidth=1)
    ax.set_title("Diagnostic Sharpe by ablation variant")
    ax.set_xlabel("Diagnostic Sharpe")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = FIGURES / "ablation_diagnostic_sharpe.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return [out.relative_to(ROOT)]


def find_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def save_benchmark_figure():
    possible_files = [
        TABLES / "benchmark_comparison.csv",
        TABLES / "baseline_comparison.csv",
        TABLES / "strategy_baseline_comparison.csv",
    ]

    path = next((p for p in possible_files if p.exists()), None)
    if path is None:
        return []

    df = pd.read_csv(path).copy()
    name_col = find_first_existing_column(
        df,
        ["strategy_name", "scenario_name", "baseline_name", "model_name", "name"],
    )
    sharpe_col = find_first_existing_column(
        df,
        ["diagnostic_sharpe", "sharpe", "strategy_sharpe"],
    )

    if name_col is None or sharpe_col is None:
        return []

    df["label"] = df[name_col].map(pretty_label)
    df = df.sort_values(sharpe_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.barh(df["label"], df[sharpe_col])
    ax.axvline(0, linewidth=1)
    ax.set_title("ML and baseline diagnostic Sharpe comparison")
    ax.set_xlabel("Diagnostic Sharpe")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = FIGURES / "benchmark_diagnostic_sharpe.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return [out.relative_to(ROOT)]


def save_feature_importance_figures():
    written = []

    candidates = sorted(TABLES.glob("*feature_importance*.csv"))
    for path in candidates:
        df = pd.read_csv(path).copy()
        feature_col = find_first_existing_column(df, ["feature", "feature_name", "name"])
        importance_col = find_first_existing_column(df, ["importance", "feature_importance", "mean_importance"])

        if feature_col is None or importance_col is None:
            continue

        df = df.sort_values(importance_col, ascending=False).head(15).copy()
        df["label"] = df[feature_col].map(pretty_label)
        df = df.sort_values(importance_col, ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(df["label"], df[importance_col])
        ax.set_title("Top feature importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        out_name = path.stem.replace("_feature_importance", "") + "_feature_importance.png"
        out = FIGURES / out_name
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out.relative_to(ROOT))

    return written


def main():
    if not REPORTS.exists():
        raise SystemExit(f"Missing reports directory: {REPORTS}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    set_public_plot_style()

    changed_text = replace_public_text_files()
    written_figures = []
    written_figures.extend(save_horizon_figures())
    written_figures.extend(save_ablation_figure())
    written_figures.extend(save_benchmark_figure())
    written_figures.extend(save_feature_importance_figures())

    print("Public label polish complete.")
    print(f"Text files changed: {len(changed_text)}")
    for path in changed_text:
        print(f"  changed: {path}")

    print(f"Figures regenerated: {len(written_figures)}")
    for path in written_figures:
        print(f"  figure:  {path}")


if __name__ == "__main__":
    main()
