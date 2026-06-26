# %% [markdown]
# Project Summary and Research Questions
#
# This project examines how plant-based food markets and YouTube narratives developed across selected European countries from 2018 to 2020. It combines country-year-product sales data with YouTube video text data to compare market size, product-category composition, narrative prevalence, and the relationship between online narratives and plant-based food sales. Since sales value and volume are strongly correlated, the later analysis focuses mainly on Value EUR as the key market indicator. The YouTube analysis uses seven narrative dimensions: health, environment, animal welfare, food security, taste, price, and convenience. Positive and negative mentions are combined into one binary variable for each narrative dimension.
#
# - Main RQ: How are YouTube narratives about plant-based foods associated with plant-based food sales patterns across selected European countries from 2018 to 2020?
# - Sub-RQ1: How do plant-based food sales values vary across countries, years, and product groups?
# - Sub-RQ2: How does the product-category composition of plant-based food sales differ across countries and over time?
# - Sub-RQ3: Which YouTube narrative dimensions are most frequently mentioned across countries and years?
# - Sub-RQ4: To what extent are the seven YouTube narrative dimensions associated with total plant-based food sales value at the country-year level?

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis_helpers import (
    add_sales_log_columns,
    create_share_table,
    create_wide_value_table,
    fit_ols,
    prepare_sales_data,
    print_basic_info,
    print_sales_overview,
)
print(f"current working directory:{os.getcwd()}")
print("Files in current directory:")
print(os.listdir("."))

# %%
df = pd.read_csv("../data/Clean/plant_based_food_sales_data.csv")
print_basic_info(df)

# %%
df = prepare_sales_data(df)

# %%
print_sales_overview(df)

# %%
df[['Value EUR', 'Volume kg/l']].hist(figsize=(10, 4))

# %% [markdown]
# The histograms show that both sales value and sales volume are strongly right-skewed, with most observations concentrated at lower levels and only a small number of observations showing very large values. This indicates substantial variation across countries, years, and product categories. Since the dataset contains both value and volume information, the subsequent analysis examines whether higher sales volumes are associated with higher sales values, which would suggest a broadly positive value–volume relationship. Applying log transformation helps reduce the influence of extreme observations and makes the variables more suitable for regression analysis when testing this relationship.

# %%
import numpy as np
df = add_sales_log_columns(df)
df[['log_value', 'log_volume']].hist(figsize=(10, 4))

# %% [markdown]
# After log transformation, the distributions of log_value and log_volume become less right-skewed compared with the original variables. The values are more spread out and less dominated by extreme observations, making them more suitable for correlation and OLS regression analysis.

# %%
import statsmodels.api as sm
X = sm.add_constant(df[['log_volume']])
model_log = fit_ols(df, ['log_volume'], 'log_value')
print(model_log.summary())

# %%
plt.figure(figsize=(8, 5))
plt.scatter(df['log_volume'], df['log_value'], alpha=0.7)
plt.plot(df['log_volume'], model_log.predict(X),color='red')
plt.title('OLS Regression: Log Value EUR vs Log Volume kg/l')
plt.xlabel('Log Volume kg/l')
plt.ylabel('Log Value EUR')
plt.tight_layout()
plt.show()

# %% [markdown]
# The log-log OLS regression shows a strong positive relationship between Volume kg/l and Value EUR. The model has a high R-squared value of 0.957, indicating that log volume explains most of the variation in log sales value. The coefficient of log_volume is positive and statistically significant, suggesting that higher sales volume is strongly associated with higher sales value. Therefore, since value and volume are highly correlated, the following analysis focuses on Value EUR as the main indicator and does not discuss Volume kg/l separately.

# %%
df_wide_value = create_wide_value_table(df)
df_wide_value.head()

# %%
country_descriptive = df_wide_value.groupby(['Country'])[['Total Value EUR']].describe()
country_descriptive

# %%
mean_value = country_descriptive[('Total Value EUR', 'mean')].sort_values(ascending=False)
mean_value.plot(kind='bar', figsize=(10, 5))
plt.title('Average plant-based food sales by country, 2018–2020')
plt.xlabel('Country')
plt.ylabel('Mean Value EUR')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# The figure shows a clear concentration of plant-based food sales in a small number of markets from 2018 to 2020. The United Kingdom, Italy, and Spain accounted for the highest average sales values, suggesting that these countries had relatively larger and more developed plant-based food markets during this period. By contrast, countries such as Denmark and Romania showed much smaller sales values, indicating that market size differed substantially across Europe. This pattern suggests that plant-based food market development was uneven, with growth opportunities likely depending on country-specific market scale and consumer demand.

# %%
df_wide_value_wide = df_wide_value.pivot(
    index='Year',
    columns='Country',
    values='Total Value EUR')
ax = df_wide_value_wide.plot(
    figsize=(12, 6),
    marker='o',
    legend=False)
for country in df_wide_value_wide.columns:
    last_year = df_wide_value_wide.index[-1]
    last_value = df_wide_value_wide[country].iloc[-1]
    ax.text(last_year + 0.03, last_value, country, va='center')

plt.title('Annual plant-based food sales across selected European countries, 2018–2020')
plt.xlabel('Year')
plt.ylabel('Total Value EUR')
plt.xticks(df_wide_value_wide.index)
plt.xlim(df_wide_value_wide.index.min(), df_wide_value_wide.index.max() + 0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# Plant-based food sales increased in most selected European countries from 2018 to 2020, but the pace and scale of growth differed clearly across markets. The United Kingdom showed the strongest upward trend and became the leading market by 2020, while Spain and Italy remained consistently large markets. In contrast, countries such as Denmark and Romania stayed at much lower sales levels, suggesting that market expansion was uneven and mainly driven by a few major countries.

# %%
df_share, product_cols = create_share_table(df_wide_value)
df_share.head()

# %%
years = sorted(df_share['Year'].unique())
countries = sorted(df_share['Country'].unique())
fig, axes = plt.subplots(1, len(years), figsize=(18, 6), sharey=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(product_cols)))
for i, year in enumerate(years):
    ax = axes[i]
    data = (df_share[df_share['Year'] == year].set_index('Country').reindex(countries)[product_cols].fillna(0))
    bottom = np.zeros(len(countries))
    for j, product in enumerate(product_cols):
        values = data[product].values
        ax.bar(countries,values,bottom=bottom,label=product,color=colors[j])
        if i > 0:
            prev_year = years[i - 1]
            prev_data = (df_share[df_share['Year'] == prev_year].set_index('Country').reindex(countries)[product_cols].fillna(0))
            change = data[product].values - prev_data[product].values
            for x, value, base, diff in zip(range(len(countries)), values, bottom, change):
                if value > 3:
                    if diff > 0:
                        ax.text(x, base + value / 2, '↑', ha='center', va='center')
                    elif diff < 0:
                        ax.text(x, base + value / 2, '↓', ha='center', va='center')
        bottom += values
    ax.set_title(str(year))
    ax.set_xlabel('Country')
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.set_ylim(0, 100)
axes[0].set_ylabel('Share of Total Value EUR (%)')

plt.suptitle('Product Share of Total Value EUR by Country and Year')
plt.legend(title='Product Group', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# The figure shows that the contribution of different product categories to total plant-based food sales varied substantially across countries and years. Plant-based milk and drinks accounted for a large share in many countries, especially in Denmark, Spain, Austria, Belgium, and France, indicating that this category was a key driver of total sales. In contrast, plant-based meat/fish alternatives and ready meals represented a particularly large share in the United Kingdom and the Netherlands, suggesting that these markets were more strongly shaped by meat-alternative products. The arrows show that category shares changed over time, but the overall market structure remained relatively stable in several countries. This suggests that plant-based food markets were not homogeneous across Europe; instead, each country appeared to have a different product-category profile, which may reflect differences in consumer preferences and market development pathways.

# %% [markdown]
# Cleaned YouTube Narrative Data
#
# The following section directly uses the cleaned aspect-coded CSV in `data/Clean`. The CSV contains positive and negative source columns for each narrative dimension, but this notebook combines each pair for analysis. For example, `health` equals 1 when either `health_positive` or `health_negative` is 1. This avoids double-counting videos that mention both positive and negative evaluations of the same dimension.
#
# Because the sales dataset covers 2018-2020, the YouTube data is filtered to the same period before the sales relationship and OLS regression analysis.

# %%
youtube_path = Path("../data/Clean/ALL_11countries_2017_2020_aspect_sentiment.csv")
df_youtube_all = pd.read_csv(youtube_path, encoding="utf-8-sig")

narrative_cols = [
    "health",
    "environment",
    "animal_welfare",
    "food_security",
    "taste",
    "price",
    "convenience",
]
sentiment_source_cols = [f"{narrative}_{sentiment}" for narrative in narrative_cols for sentiment in ["positive", "negative"]]

def format_narrative_label(column):
    return column.replace("_", " ").title()

narrative_meta = pd.DataFrame({
    "narrative": narrative_cols,
    "positive_source_column": [f"{col}_positive" for col in narrative_cols],
    "negative_source_column": [f"{col}_negative" for col in narrative_cols],
})
narrative_meta["label"] = narrative_meta["narrative"].apply(format_narrative_label)

df_youtube = df_youtube_all.copy()
df_youtube["Year"] = pd.to_numeric(df_youtube["year"], errors="coerce").astype("Int64")
df_youtube["Country"] = df_youtube["country"].astype(str).str.strip()
df_youtube[sentiment_source_cols] = (
    df_youtube[sentiment_source_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .astype(int)
)
for narrative in narrative_cols:
    source_cols = [f"{narrative}_positive", f"{narrative}_negative"]
    df_youtube[narrative] = df_youtube[source_cols].max(axis=1).astype(int)

df_youtube["view_count_numeric"] = pd.to_numeric(df_youtube.get("view_count"), errors="coerce").fillna(0)
df_youtube_1820 = df_youtube[df_youtube["Year"].between(2018, 2020)].copy()

print("Loaded cleaned YouTube data from:", youtube_path)
print("All YouTube rows:", len(df_youtube_all))
print("YouTube rows used for 2018-2020 analysis:", len(df_youtube_1820))
print("Videos by year:")
print(df_youtube_1820["Year"].value_counts().sort_index())
print("Countries:", sorted(df_youtube_1820["Country"].dropna().unique()))
display(narrative_meta)
display(df_youtube_1820[["Country", "Year", "title"] + narrative_cols].head())

# %%
example_rows = []
for narrative in narrative_cols:
    examples = (
        df_youtube_1820[df_youtube_1820[narrative].eq(1)]
        .sort_values("view_count_numeric", ascending=False)
        .head(3)
    )
    for _, row in examples.iterrows():
        example_rows.append({
            "narrative": format_narrative_label(narrative),
            "country": row["Country"],
            "year": int(row["Year"]),
            "title": row["title"],
            "url": row.get("url", ""),
            "view_count": int(row["view_count_numeric"]),
        })

example_videos = pd.DataFrame(example_rows)
display(example_videos)

# %% [markdown]
# The example table is used as a content check for the combined narrative categories. Each category now includes both positive and negative mentions, so the table should be interpreted as showing videos that discuss a topic, not videos that necessarily evaluate it positively. For example, the health category includes both health benefits and health risks.
# %%
narrative_by_year = df_youtube_1820.groupby("Year")[narrative_cols].sum()
narrative_rate_by_year = df_youtube_1820.groupby("Year")[narrative_cols].mean().mul(100)
narrative_totals = df_youtube_1820[narrative_cols].sum().sort_values(ascending=False)
narrative_rate_by_country = df_youtube_1820.groupby("Country")[narrative_cols].mean().mul(100)

display(narrative_totals.rename("mentioned_videos").to_frame())
display(narrative_by_year)

# %%
plot_totals = narrative_totals.sort_values()
plot_labels = [format_narrative_label(col) for col in plot_totals.index]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(plot_labels, plot_totals.values, color="#4C78A8")
ax.set_title("Total YouTube Videos Mentioning Each Narrative, 2018-2020")
ax.set_xlabel("Number of videos")
ax.set_ylabel("Narrative category")
for i, value in enumerate(plot_totals.values):
    ax.text(value + 5, i, f"{int(value)}", va="center", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# After combining positive and negative mentions, health is the most frequent narrative, followed by taste and convenience. Environment, price, animal welfare, and food security appear much less often. This suggests that YouTube discussion of plant-based foods is mainly organized around personal health, eating experience, and practical use, while broader ethical or system-level narratives are less visible in the title and description text.
# %%
heatmap_year = narrative_rate_by_year[narrative_cols].T
heatmap_labels = [format_narrative_label(col) for col in heatmap_year.index]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(heatmap_year.values, aspect="auto", cmap="YlGnBu")
ax.set_xticks(range(len(heatmap_year.columns)))
ax.set_xticklabels(heatmap_year.columns.astype(int))
ax.set_yticks(range(len(heatmap_labels)))
ax.set_yticklabels(heatmap_labels)
ax.set_xlabel("Year")
ax.set_title("Narrative Mention Rate by Year (% of videos)")

for row_i in range(heatmap_year.shape[0]):
    for col_i in range(heatmap_year.shape[1]):
        value = heatmap_year.iloc[row_i, col_i]
        ax.text(col_i, row_i, f"{value:.1f}", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=ax, label="% of videos")
plt.tight_layout()
plt.show()

# %% [markdown]
# The yearly heatmap shows that health remains the most common narrative in every year, appearing in roughly one quarter of videos. Taste is also consistently visible, while convenience increases from 2018 to 2020. Environment declines over time in this dataset, and food security remains rare. These patterns suggest that the most stable YouTube narratives are consumer-facing topics rather than abstract system-level topics.
# %%
country_order = narrative_rate_by_country.mean(axis=1).sort_values(ascending=False).index
heatmap_country = narrative_rate_by_country.loc[country_order, narrative_cols]
country_labels = heatmap_country.index.tolist()
column_labels = [format_narrative_label(col) for col in heatmap_country.columns]

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heatmap_country.values, aspect="auto", cmap="YlGnBu")
ax.set_xticks(range(len(column_labels)))
ax.set_xticklabels(column_labels, rotation=45, ha="right")
ax.set_yticks(range(len(country_labels)))
ax.set_yticklabels(country_labels)
ax.set_title("Narrative Mention Rate by Country, 2018-2020 (% of videos)")

for row_i in range(heatmap_country.shape[0]):
    for col_i in range(heatmap_country.shape[1]):
        value = heatmap_country.iloc[row_i, col_i]
        if value >= 10:
            ax.text(col_i, row_i, f"{value:.0f}", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=ax, label="% of videos")
plt.tight_layout()
plt.show()

# %% [markdown]
# The country heatmap shows that health, taste, and convenience are the main narrative dimensions across most countries, but the emphasis differs by market. Some countries show stronger health framing, while others show more taste or convenience framing. This suggests that plant-based food communication is broadly similar across countries, but local markets differ in which consumer-facing narrative receives more attention.
# %%
sales_path = Path("../data/Clean/plant_based_food_sales_data.csv")
sales_raw_for_merge = pd.read_csv(sales_path)

sales_country_year = sales_raw_for_merge[["Country", "Year", "Value EUR"]].copy()
sales_country_year["Year"] = pd.to_numeric(sales_country_year["Year"], errors="coerce").astype("Int64")
sales_country_year["Value EUR"] = pd.to_numeric(sales_country_year["Value EUR"], errors="coerce").fillna(0)
sales_country_year = sales_country_year.dropna(subset=["Country", "Year"])
sales_country_year = sales_country_year[sales_country_year["Value EUR"] != 0]
sales_country_year = (
    sales_country_year
    .groupby(["Country", "Year"], as_index=False)["Value EUR"]
    .sum()
    .rename(columns={"Value EUR": "Total Value EUR"})
)
sales_country_year["Year"] = sales_country_year["Year"].astype(int)

narrative_country_year = (
    df_youtube_1820
    .groupby(["Country", "Year"])
    .agg(video_count=("video_id", "count"), **{col: (col, "sum") for col in narrative_cols})
    .reset_index()
)
narrative_country_year["Year"] = narrative_country_year["Year"].astype(int)
for col in narrative_cols:
    narrative_country_year[f"{col}_rate"] = narrative_country_year[col] / narrative_country_year["video_count"]

missing_sales_country_year = (
    narrative_country_year[["Country", "Year"]]
    .merge(sales_country_year[["Country", "Year"]], on=["Country", "Year"], how="left", indicator=True)
    .query("_merge == 'left_only'")
    .drop(columns="_merge")
)

narrative_sales_country_year = narrative_country_year.merge(
    sales_country_year,
    on=["Country", "Year"],
    how="inner",
)
narrative_sales_country_year["log_total_value"] = np.log(narrative_sales_country_year["Total Value EUR"])

print("Country-year observations in YouTube data:", len(narrative_country_year))
print("Country-year observations used in sales relationship analysis:", len(narrative_sales_country_year))
print("Country-year observations without usable sales value:")
display(missing_sales_country_year)
display(narrative_sales_country_year.head())

# %% [markdown]
# For the sales relationship analysis, YouTube narratives are aggregated to the country-year level and merged with total sales value. This keeps the sales analysis aligned with the 2018-2020 period. The merge uses non-zero `Value EUR`; therefore Germany is listed as unmatched because the clean sales file does not contain usable non-zero sales value for Germany in these years. The YouTube-only narrative charts above still include Germany.
# %%
rate_cols = [f"{col}_rate" for col in narrative_cols]

corr_table = pd.DataFrame({
    "narrative": narrative_cols,
    "corr_with_log_total_value": [
        narrative_sales_country_year[f"{col}_rate"].corr(narrative_sales_country_year["log_total_value"])
        for col in narrative_cols
    ],
})
corr_table = corr_table.merge(narrative_meta[["narrative", "label"]], on="narrative", how="left")
corr_table = corr_table.sort_values("corr_with_log_total_value", ascending=False)
display(corr_table[["label", "corr_with_log_total_value"]])

plot_corr = corr_table.sort_values("corr_with_log_total_value")
colors = ["#4C78A8" if value >= 0 else "#E45756" for value in plot_corr["corr_with_log_total_value"]]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(plot_corr["label"], plot_corr["corr_with_log_total_value"], color=colors)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Correlation Between Narrative Mention Rate and Log Sales Value")
ax.set_xlabel("Correlation with log(Total Value EUR)")
ax.set_ylabel("Narrative category")
plt.tight_layout()
plt.show()

# %% [markdown]
# The correlation chart is useful as an exploratory first look, but it does not control for overlap between narratives. Therefore, the final analysis uses OLS regression, following the method introduced in class. Before fitting the OLS model, the next heatmap checks how strongly the seven narrative variables are correlated with each other.
# %%
rate_corr = narrative_sales_country_year[rate_cols].corr()
rate_corr.index = [format_narrative_label(col.replace("_rate", "")) for col in rate_corr.index]
rate_corr.columns = [format_narrative_label(col.replace("_rate", "")) for col in rate_corr.columns]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(rate_corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(rate_corr.columns)))
ax.set_xticklabels(rate_corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(rate_corr.index)))
ax.set_yticklabels(rate_corr.index)
ax.set_title("Correlation Among Narrative Mention Rates")

for row_i in range(rate_corr.shape[0]):
    for col_i in range(rate_corr.shape[1]):
        value = rate_corr.iloc[row_i, col_i]
        ax.text(col_i, row_i, f"{value:.2f}", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=ax, label="Correlation")
plt.tight_layout()
plt.show()

# %% [markdown]
# The heatmap shows whether the seven narrative rates tend to appear together in the same country-year contexts. This matters for OLS interpretation because a coefficient estimates the association of one narrative while holding the other included narratives constant. If two narrative variables overlap strongly, their individual coefficients may become less stable.
# %%
# OLS regression: combined narrative mention rates and plant-based food sales
# The dependent variable is log(Total Value EUR).
# Each explanatory variable is the share of videos in a country-year that mention a narrative dimension.

regression_data = narrative_sales_country_year[
    ["Country", "Year", "Total Value EUR", "log_total_value", "video_count"] + rate_cols
].dropna().copy()

combined_narrative_ols_model = fit_ols(regression_data, rate_cols, "log_total_value")

ols_model_summary = pd.DataFrame([{
    "model": "Combined 7-narrative OLS",
    "n_observations": int(combined_narrative_ols_model.nobs),
    "n_predictors": len(rate_cols),
    "r_squared": combined_narrative_ols_model.rsquared,
    "adjusted_r_squared": combined_narrative_ols_model.rsquared_adj,
    "f_pvalue": combined_narrative_ols_model.f_pvalue,
}])
display(ols_model_summary)

conf_int = combined_narrative_ols_model.conf_int().loc[rate_cols]
ols_coefficients = pd.DataFrame({
    "variable": rate_cols,
    "coefficient": combined_narrative_ols_model.params.loc[rate_cols],
    "p_value": combined_narrative_ols_model.pvalues.loc[rate_cols],
    "conf_low": conf_int[0],
    "conf_high": conf_int[1],
}).reset_index(drop=True)
ols_coefficients["narrative"] = ols_coefficients["variable"].str.replace("_rate", "", regex=False)
ols_coefficients["label"] = ols_coefficients["narrative"].apply(format_narrative_label)
ols_coefficients["coefficient_per_10pp_increase"] = ols_coefficients["coefficient"] * 0.10
ols_coefficients["conf_low_per_10pp"] = ols_coefficients["conf_low"] * 0.10
ols_coefficients["conf_high_per_10pp"] = ols_coefficients["conf_high"] * 0.10

display(ols_coefficients[[
    "label",
    "coefficient_per_10pp_increase",
    "p_value",
    "conf_low_per_10pp",
    "conf_high_per_10pp",
]].sort_values("coefficient_per_10pp_increase", ascending=False))

print("Combined 7-narrative OLS summary")
print(combined_narrative_ols_model.summary())

# %% [markdown]
# The OLS model estimates the association between the seven combined narrative mention rates and log sales value at the country-year level. Each coefficient shows the expected change in log sales value when a narrative's mention rate increases by 10 percentage points, holding the other six narrative rates constant. This regression goes beyond simple correlation because it controls for overlap between narrative dimensions. However, the results should still be interpreted as exploratory associations rather than causal effects because the sample is small and country-level market differences may also matter.
# %%
plot_coefficients = ols_coefficients.sort_values("coefficient_per_10pp_increase")
y_positions = np.arange(len(plot_coefficients))
lower_error = plot_coefficients["coefficient_per_10pp_increase"] - plot_coefficients["conf_low_per_10pp"]
upper_error = plot_coefficients["conf_high_per_10pp"] - plot_coefficients["coefficient_per_10pp_increase"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(
    plot_coefficients["coefficient_per_10pp_increase"],
    y_positions,
    xerr=[lower_error, upper_error],
    fmt="o",
    color="#4C78A8",
    ecolor="gray",
    capsize=3,
)
ax.axvline(0, color="black", linewidth=1)
ax.set_yticks(y_positions)
ax.set_yticklabels(plot_coefficients["label"])
ax.set_xlabel("OLS coefficient per 10 percentage point increase in mention rate")
ax.set_title("OLS Coefficients: Narrative Mention Rates and Log Sales Value")
plt.tight_layout()
plt.show()

# %% [markdown]
# The coefficient plot visualizes the OLS regression results for the seven combined narratives. Points to the right of zero indicate narratives that are positively associated with log sales value after controlling for the other narratives; points to the left indicate negative associations. Wide confidence intervals would indicate uncertainty, so the direction, size, and statistical significance of each coefficient should be discussed together rather than relying only on the visual position of the point.
