"""Reusable helpers for the final plant-based food and YouTube narrative analysis."""

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


def merge_narrative_sales(narrative_df, sales_df):
    """Merge country-year narrative rates with country-year sales value."""
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
