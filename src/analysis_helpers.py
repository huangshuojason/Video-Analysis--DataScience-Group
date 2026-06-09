"""Reusable helpers for the plant-based food and YouTube narrative analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def print_basic_info(df):
    """Print basic DataFrame information and the first rows."""
    print(df.info())
    print(df.head())


def print_sales_overview(df):
    """Print a compact overview of the cleaned sales dataset."""
    print("Head")
    print(df.head(2))
    print("Tail")
    print(df.tail(2))
    print("\nUnique countries in the dataset:")
    print(df["Country"].unique())
    print("\nUnique years in the dataset:")
    print(df["Year"].unique())
    print("\nUnique product groups in the dataset:")
    print(df["Product Group"].unique())


def prepare_sales_data(df):
    """Clean and aggregate sales data to country-year-product group level."""
    keep_cols = ["Country", "Year", "Value EUR", "Volume kg/l", "Product Group"]
    cleaned = df[keep_cols].dropna()
    cleaned = cleaned[(cleaned["Value EUR"] != 0) & (cleaned["Volume kg/l"] != 0)]
    return (
        cleaned.groupby(["Country", "Year", "Product Group"], as_index=False)[
            ["Value EUR", "Volume kg/l"]
        ]
        .sum()
    )


def add_sales_log_columns(df):
    """Add log-transformed sales value and volume columns."""
    result = df.copy()
    result["log_value"] = np.log(result["Value EUR"])
    result["log_volume"] = np.log(result["Volume kg/l"])
    return result


def create_wide_value_table(df):
    """Create a country-year wide table by product group and add total value."""
    wide = df.pivot(
        index=["Country", "Year"],
        columns="Product Group",
        values="Value EUR",
    ).reset_index()
    wide.columns.name = None
    product_cols = wide.columns.drop(["Country", "Year"])
    wide["Total Value EUR"] = wide[product_cols].sum(axis=1)
    return wide


def create_share_table(wide_df, id_cols=("Country", "Year"), total_col="Total Value EUR"):
    """Convert product value columns into row-wise percentage shares."""
    share = wide_df.copy()
    product_cols = share.columns.drop(list(id_cols) + [total_col])
    share[product_cols] = share[product_cols].div(share[total_col], axis=0) * 100
    return share, product_cols


def combine_youtube_csvs(folder, pattern="youtube_plant_based_*.csv", exclude="all_countries"):
    """Load country-level YouTube CSV files, combine them, and remove duplicates."""
    folder = Path(folder)
    csv_files = [file for file in folder.glob(pattern) if exclude not in file.name]
    if not csv_files:
        raise ValueError("No CSV files found. Please check your folder path.")

    frames = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        temp_df["source_file"] = file.name
        frames.append(temp_df)

    combined = pd.concat(frames, ignore_index=True)
    if {"country", "video_id"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["country", "video_id"])
    else:
        print("Warning: country or video_id column not found. Skipping duplicate removal.")
    return combined, csv_files


def prepare_youtube_text(df, date_col="upload_date", text_cols=("title", "description", "summary")):
    """Create Year and combined lowercase text columns for narrative analysis."""
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result["Year"] = result[date_col].dt.year

    for col in text_cols:
        if col not in result.columns:
            result[col] = ""

    result["text_for_analysis"] = (
        result[list(text_cols)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    return result


def add_narrative_features(df, narrative_keywords, text_col="text_for_analysis"):
    """Add keyword count and 0/1 mention columns for each narrative."""
    result = df.copy()
    for narrative, keywords in narrative_keywords.items():
        count_col = f"{narrative}_count"
        mentioned_col = f"{narrative}_mentioned"
        result[count_col] = result[text_col].apply(
            lambda text: sum(keyword in text for keyword in keywords)
        )
        result[mentioned_col] = (result[count_col] > 0).astype(int)
    return result


def narrative_mention_columns(narrative_keywords):
    """Return the *_mentioned columns created from a narrative keyword dictionary."""
    return [f"{narrative}_mentioned" for narrative in narrative_keywords.keys()]


def summarize_narratives(df, group_cols, narrative_keywords):
    """Count videos mentioning each narrative by the chosen grouping columns."""
    mention_cols = narrative_mention_columns(narrative_keywords)
    return df.groupby(group_cols)[mention_cols].sum().reset_index()


def create_narrative_share_table(narrative_df, narrative_cols, country_col="country"):
    """Convert narrative mention counts into row-wise percentage shares."""
    share = narrative_df.copy()
    share["Year"] = share["Year"].astype(int)
    share[narrative_cols] = share[narrative_cols].div(share[narrative_cols].sum(axis=1), axis=0) * 100
    countries = sorted(share[country_col].unique())
    years = sorted(share["Year"].unique())
    return share, countries, years


def merge_narrative_sales(narrative_df, sales_df):
    """Merge country-year narrative counts with country-year sales value."""
    narrative = narrative_df.rename(columns={"country": "Country"}).copy()
    narrative["Year"] = narrative["Year"].astype(int)

    sales = sales_df[["Country", "Year", "Total Value EUR"]].copy()
    sales["Year"] = sales["Year"].astype(int)

    return narrative.merge(sales, on=["Country", "Year"], how="inner")


def fit_ols(df, x_cols, y_col):
    """Fit an OLS model with a constant term."""
    x = sm.add_constant(df[x_cols])
    y = df[y_col]
    return sm.OLS(y, x).fit()


def add_log_total_value(df):
    """Add log-transformed total sales value."""
    result = df.copy()
    result["log_total_value"] = np.log(result["Total Value EUR"])
    return result
