"""Helpers used only by the 0624 YouTube video/narrative analysis."""

from itertools import combinations
import re

import numpy as np
import pandas as pd


def read_youtube_csv(path, encodings=("utf-8-sig", "utf-8", "latin1")):
    """Read the cleaned YouTube CSV with a small encoding fallback."""
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


def normalize_columns(df):
    """Strip column-name whitespace/BOM artifacts and remove unnamed export columns."""
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.replace("﻿", "", regex=False)
        .str.strip()
    )
    return result.loc[:, ~result.columns.astype(str).str.startswith("Unnamed:")]


def format_label(column):
    """Format snake_case variable names for tables and charts."""
    cleaned = (
        str(column)
        .replace("_rate", "")
        .replace("_mentioned", "")
        .replace("_positive", "")
        .replace("_negative", "")
        .replace("_", " ")
    )
    return cleaned.title()


def prepare_youtube_video_data(df, narrative_cols, year_range=(2018, 2020)):
    """Standardize YouTube data and create *_mentioned narrative flags.

    The 0624 notebook originally used AI-coded positive/negative columns such as
    health_positive and health_negative. The newer cleaned CSV already contains
    one 0/1 column per narrative theme, such as health or environment. This
    helper supports both formats so downstream analysis can remain unchanged.
    """
    result = normalize_columns(df)

    year_col = "year" if "year" in result.columns else "Year"
    country_col = "country" if "country" in result.columns else "Country"
    if year_col not in result.columns or country_col not in result.columns:
        raise KeyError("YouTube data must contain year/country columns.")

    result["Year"] = pd.to_numeric(result[year_col], errors="coerce")
    result["Country"] = result[country_col].astype("string").str.strip()
    result = result.dropna(subset=["Year", "Country"])

    if year_range is not None:
        start_year, end_year = year_range
        result = result[result["Year"].between(start_year, end_year)].copy()

    result["Year"] = result["Year"].astype(int)

    numeric_cols = ["view_count", "like_count", "comment_count", "duration_seconds"]
    for col in numeric_cols:
        numeric_col = f"{col}_numeric"
        if col in result.columns:
            result[numeric_col] = pd.to_numeric(result[col], errors="coerce")
        else:
            result[numeric_col] = np.nan
    result["view_count_numeric"] = result["view_count_numeric"].fillna(0)

    missing_narratives = []
    for narrative in narrative_cols:
        positive_col = f"{narrative}_positive"
        negative_col = f"{narrative}_negative"
        mentioned_col = f"{narrative}_mentioned"
        if narrative in result.columns:
            result[mentioned_col] = (
                pd.to_numeric(result[narrative], errors="coerce")
                .fillna(0)
                .gt(0)
                .astype(int)
            )
        elif positive_col in result.columns and negative_col in result.columns:
            ai_labels = (
                result[[positive_col, negative_col]]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
            result[mentioned_col] = ai_labels.eq(1).any(axis=1).astype(int)
        else:
            missing_narratives.append(narrative)

    if missing_narratives:
        raise KeyError(
            "Missing narrative columns. Expected either a direct 0/1 column or "
            f"positive/negative columns for: {missing_narratives}"
        )

    return result


def youtube_basic_descriptives(df):
    """Return compact descriptive information for the prepared YouTube video CSV."""
    year_min = int(df["Year"].min()) if len(df) else np.nan
    year_max = int(df["Year"].max()) if len(df) else np.nan
    year_range = f"{year_min}-{year_max}" if len(df) else ""
    channels = df["channel_title"].nunique() if "channel_title" in df.columns else np.nan

    rows = [
        ("videos", len(df)),
        ("countries", df["Country"].nunique()),
        ("years", year_range),
        ("channels", channels),
        ("total_views", int(df["view_count_numeric"].sum())),
        ("median_views", float(df["view_count_numeric"].median())),
        ("median_likes", float(df["like_count_numeric"].median())),
        ("median_comments", float(df["comment_count_numeric"].median())),
        ("median_duration_minutes", float(df["duration_seconds_numeric"].median() / 60)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def youtube_country_year_summary(df):
    """Summarize YouTube video volume and reach by country and year."""
    return (
        df.groupby(["Country", "Year"], as_index=False)
        .agg(
            video_count=("video_id", "count"),
            total_views=("view_count_numeric", "sum"),
            median_views=("view_count_numeric", "median"),
            active_channels=("channel_title", "nunique"),
        )
        .sort_values(["Country", "Year"])
    )


def summarize_youtube_narratives(df, narrative_cols):
    """Summarize AI narrative mention frequency and view share for YouTube videos."""
    total_views = df["view_count_numeric"].sum()
    rows = []
    for narrative in narrative_cols:
        mentioned_col = f"{narrative}_mentioned"
        mentioned = df[mentioned_col].eq(1)
        rows.append(
            {
                "narrative": narrative,
                "label": format_label(narrative),
                "mentioned_videos": int(mentioned.sum()),
                "mention_rate_pct": mentioned.mean() * 100,
                "views_on_mentioned_videos": int(df.loc[mentioned, "view_count_numeric"].sum()),
                "view_share_pct": (
                    df.loc[mentioned, "view_count_numeric"].sum() / total_views * 100
                    if total_views
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mention_rate_pct", ascending=False)


def aggregate_youtube_narratives(df, narrative_cols, group_cols=("Country", "Year")):
    """Aggregate AI narrative mention rates by country/year or another grouping."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    mention_cols = [f"{narrative}_mentioned" for narrative in narrative_cols]
    grouped = df.groupby(group_cols)

    summary = grouped.agg(
        video_count=("video_id", "count"),
        total_views=("view_count_numeric", "sum"),
    ).reset_index()

    rates = grouped[mention_cols].mean().reset_index()
    rates = rates.rename(
        columns={f"{narrative}_mentioned": f"{narrative}_rate" for narrative in narrative_cols}
    )
    return summary.merge(rates, on=group_cols, how="left")


def create_ols_coefficient_table(model, x_cols, scale=1.0):
    """Create a tidy coefficient table from a fitted statsmodels OLS model."""
    conf_int = model.conf_int().loc[x_cols]
    table = pd.DataFrame(
        {
            "variable": x_cols,
            "label": [format_label(col) for col in x_cols],
            "coefficient": model.params.loc[x_cols] * scale,
            "p_value": model.pvalues.loc[x_cols],
            "conf_low": conf_int[0] * scale,
            "conf_high": conf_int[1] * scale,
        }
    )
    return table.reset_index(drop=True)


def parse_codebook_expressions(codebook):
    """Convert semicolon-separated codebook expression text into tidy rows."""
    rows = []
    for _, row in codebook.iterrows():
        for rank, item in enumerate(str(row["top_matched_expressions"]).split("; "), start=1):
            match = re.match(r"(.+) \((\d+)\)$", item.strip())
            if match:
                rows.append(
                    {
                        "theme": format_label(row["theme_column"]),
                        "theme_column": row["theme_column"],
                        "expression": match.group(1),
                        "mention_count_videos": int(match.group(2)),
                        "rank_within_theme": rank,
                    }
                )
    return pd.DataFrame(rows)


def plot_theme_expression_clouds(expressions, max_terms_per_theme=8, ncols=None):
    """Plot one readable text-size expression figure for each narrative theme."""
    import matplotlib.pyplot as plt
    import textwrap

    if expressions.empty:
        raise ValueError("No expressions available for plotting.")

    themes = expressions["theme"].drop_duplicates().tolist()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#59A14F"]
    figures = []

    for theme in themes:
        data = (
            expressions[expressions["theme"].eq(theme)]
            .sort_values("mention_count_videos", ascending=False)
            .head(max_terms_per_theme)
            .reset_index(drop=True)
        )
        max_count = data["mention_count_videos"].max()
        min_count = data["mention_count_videos"].min()
        log_counts = np.log1p(data["mention_count_videos"])
        log_span = max(log_counts.max() - log_counts.min(), 1e-9)

        fig_height = max(4.5, 0.9 * len(data) + 1.8)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.set_title(theme, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.75, len(data) - 0.25)
        ax.axis("off")

        for i, row in data.iterrows():
            y = len(data) - i - 1
            size = 13 + 18 * (np.log1p(row["mention_count_videos"]) - log_counts.min()) / log_span
            label = textwrap.fill(str(row["expression"]), width=34)
            ax.text(
                5,
                y,
                label,
                ha="left",
                va="center",
                fontsize=size,
                color=colors[i % len(colors)],
                alpha=0.92,
            )
            ax.text(
                96,
                y,
                f"{int(row['mention_count_videos'])}",
                ha="right",
                va="center",
                fontsize=10,
                color="#333333",
            )
            ax.hlines(y - 0.42, xmin=5, xmax=96, color="#E6E6E6", linewidth=0.6)

        ax.text(96, len(data) - 0.52, "videos", ha="right", va="bottom", fontsize=9, color="#666666")
        plt.tight_layout()
        figures.append((fig, ax))

    return figures


def run_theme_only_ols_search(regression_data, ols_cols, fit_model, y_col="log_total_value"):
    """Fit every non-empty combination of theme-rate variables and rank by adjusted R-squared."""
    model_rows = []
    fitted_models = {}
    for n_predictors in range(1, len(ols_cols) + 1):
        for combo in combinations(ols_cols, n_predictors):
            combo = tuple(combo)
            model = fit_model(regression_data, list(combo), y_col)
            fitted_models[combo] = model
            model_rows.append(
                {
                    "n_predictors": n_predictors,
                    "themes": ", ".join(format_label(col) for col in combo),
                    "rate_cols": combo,
                    "r_squared": model.rsquared,
                    "adjusted_r_squared": model.rsquared_adj,
                    "aic": model.aic,
                    "bic": model.bic,
                }
            )

    model_search_results = pd.DataFrame(model_rows)
    adjusted_ranked_models = (
        model_search_results
        .sort_values(["adjusted_r_squared", "r_squared"], ascending=False)
        .reset_index(drop=True)
    )
    selected_rate_cols = list(adjusted_ranked_models.loc[0, "rate_cols"])
    selected_model = fitted_models[tuple(selected_rate_cols)]
    return adjusted_ranked_models, selected_rate_cols, selected_model


def summarize_ols_choice(model, rate_cols, model_name="Highest adjusted R-squared theme-only OLS"):
    """Create a one-row summary for a selected theme-only OLS model."""
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "selection_rule": "highest adjusted R-squared",
                "n_observations": int(model.nobs),
                "n_predictors": len(rate_cols),
                "r_squared": model.rsquared,
                "adjusted_r_squared": model.rsquared_adj,
                "f_pvalue": model.f_pvalue,
                "themes": ", ".join(format_label(col) for col in rate_cols),
            }
        ]
    )


def plot_ols_coefficients(coefficient_table, title, color="#4C78A8"):
    """Plot OLS coefficients with confidence intervals."""
    import matplotlib.pyplot as plt

    plot_data = coefficient_table.sort_values("coefficient_per_10pp")
    y_positions = np.arange(len(plot_data))
    lower_error = plot_data["coefficient_per_10pp"] - plot_data["conf_low_per_10pp"]
    upper_error = plot_data["conf_high_per_10pp"] - plot_data["coefficient_per_10pp"]

    fig_height = max(4.5, 0.42 * len(plot_data) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.errorbar(
        plot_data["coefficient_per_10pp"],
        y_positions,
        xerr=[lower_error, upper_error],
        fmt="o",
        color=color,
        ecolor="gray",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["label"])
    ax.set_xlabel("Effect on log sales value per 10 pp higher mention rate")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def prepare_sales_country_year_value(df):
    """Aggregate sales to country-year total value for the YouTube merge."""
    result = df[["Country", "Year", "Value EUR"]].copy()
    result["Year"] = pd.to_numeric(result["Year"], errors="coerce")
    result["Value EUR"] = pd.to_numeric(result["Value EUR"], errors="coerce").fillna(0)
    result = result.dropna(subset=["Country", "Year"])
    result = result[result["Value EUR"] != 0]
    result = (
        result.groupby(["Country", "Year"], as_index=False)["Value EUR"]
        .sum()
        .rename(columns={"Value EUR": "Total Value EUR"})
    )
    result["Year"] = result["Year"].astype(int)
    return result
