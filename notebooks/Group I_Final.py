# %% [markdown]
# Project Summary and Research Questions
#
# This project examines how plant-based food markets and YouTube narratives developed across selected European countries from 2018 to 2020. It combines country-year-product sales data with YouTube video metadata and title/description-based narrative coding to compare market size, product-category composition, narrative prevalence, and the relationship between online narratives and plant-based food sales. 
#
# - Main RQ: How were YouTube narratives about plant-based foods associated with sales patterns across selected European countries from 2018 to 2020?
# - Sub-RQ1: How did plant-based food sales vary across countries and years?
# - Sub-RQ2: Which YouTube narrative themes were most frequently mentioned across countries and years?
# - Sub-RQ3: How were coded YouTube narrative mention rates associated with total plant-based food sales value at the country-year level?
# YouTube video (Final presentation): https://youtu.be/f_vWz5yEDJ0

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
# The histograms show the original sales value and volume distributions before any transformation. Both variables are strongly right-skewed: most country-year-product observations are clustered at relatively low levels, while a small number of observations are much larger. This means the dataset contains very uneven market sizes and product outcomes, so a few large observations could dominate the analysis if the raw values are used directly. Key takeaway: the sales data need a transformation or careful interpretation before regression analysis.

# %%
import numpy as np
df = add_sales_log_columns(df)
df[['log_value', 'log_volume']].hist(figsize=(10, 4))

# %% [markdown]
# After log transformation, the distributions of sales value and volume become less extremely right-skewed. The log scale compresses very large observations and makes differences among smaller and medium-sized observations easier to see. This does not remove market variation, but it makes the data more balanced for statistical modeling. Key takeaway: the log-transformed variables are more suitable for correlation and OLS regression than the original raw values.

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
# The scatter plot shows a strong positive relationship between log sales volume and log sales value. The fitted regression line closely follows the overall pattern of the points, and the model R-squared is about 0.957. This means that higher sales volume is strongly associated with higher sales value in this dataset. Key takeaway: because value and volume move closely together, the rest of the analysis can focus on Value EUR as the main market-size indicator without losing the main sales pattern.

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
# This bar chart compares average total plant-based food sales by country from 2018 to 2020. The United Kingdom, Italy, and Spain form the largest market group, with much higher average sales values than most other countries. France and the Netherlands are also important markets, while Denmark and Romania are much smaller in comparison. Key takeaway: plant-based food market size is highly uneven across Europe, so market opportunities and launch priorities should be country-specific rather than treated as one uniform European market.

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
# This line chart shows how total plant-based food sales changed by country from 2018 to 2020. Most countries experienced growth during the period, but the scale and speed of growth were not the same. The United Kingdom stands out because it combines the largest sales scale with the clearest upward trend, while Italy remains large but grows more slowly. Key takeaway: growth is broad but uneven, meaning the strongest markets are not only those with high sales today but also those with strong upward momentum.

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
# This stacked bar chart shows each country's plant-based product-category mix by year, using each category's share of total sales value. Plant-based milk and drinks dominate many countries, especially markets such as Denmark, Spain, Austria, Belgium, and France. The United Kingdom and the Netherlands look different because plant-based meat/fish alternatives and ready meals take a much larger share of their sales. The category shares change over time, but several countries keep a fairly stable product structure. Key takeaway: plant-based food markets are not homogeneous, so product portfolio strategy should be adapted to each country's category profile.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = next(
    (
        path
        for path in [Path.cwd(), Path.cwd().parent, Path("..").resolve()]
        if (path / "data").exists() and (path / "src").exists()
    ),
    Path("..").resolve(),
)
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from analysis_helpers import add_log_total_value, fit_ols, merge_narrative_sales
from youtube_analysis_helpers import (
    aggregate_youtube_narratives,
    create_ols_coefficient_table,
    format_label,
    parse_codebook_expressions,
    plot_ols_coefficients,
    plot_theme_expression_clouds,
    prepare_sales_country_year_value,
    prepare_youtube_video_data,
    read_youtube_csv,
    run_theme_only_ols_search,
    summarize_ols_choice,
    summarize_youtube_narratives,
    youtube_basic_descriptives,
    youtube_country_year_summary,
)

YOUTUBE_FILE = "ALL_11countries_2018_2020_aspect_sentiment_narrative_coded.csv"
CODEBOOK_FILE = "narrative_coding_codebook.csv"
narrative_cols = [
    "health",
    "environment",
    "taste",
    "animal_ethics",
    "recipe_cooking",
    "convenience_easy",
    "fitness_weight",
    "challenge_transition",
    "product_review_alternatives",
    "budget_shopping",
    "restaurant_travel",
]
ols_cols = [f"{col}_rate" for col in narrative_cols]

# %%
youtube_path = PROJECT_ROOT / "data/Clean" / YOUTUBE_FILE
df_youtube_raw = read_youtube_csv(youtube_path)
df_youtube = prepare_youtube_video_data(
    df_youtube_raw,
    narrative_cols,
    year_range=(2018, 2020),
)

print("Loaded YouTube CSV from:", youtube_path)
print("Raw rows:", len(df_youtube_raw))
print("Rows used for 2018-2020 analysis:", len(df_youtube))

display(youtube_basic_descriptives(df_youtube))

video_count_table = (
    youtube_country_year_summary(df_youtube)
    .pivot(index="Country", columns="Year", values="video_count")
    .fillna(0)
    .astype(int)
)
display(video_count_table)

# %%
top_youtube_videos = (
    df_youtube
    .sort_values("view_count_numeric", ascending=False)
    [[
        "Country",
        "Year",
        "title",
        "channel_title",
        "view_count_numeric",
        "like_count_numeric",
        "comment_count_numeric",
        "url",
    ]]
    .head(10)
    .rename(
        columns={
            "view_count_numeric": "views",
            "like_count_numeric": "likes",
            "comment_count_numeric": "comments",
        }
    )
)

display(top_youtube_videos)

# %% [markdown]
# The descriptive tables keep the basic information from the YouTube video CSV: total rows, countries, years, channels, views, and video counts by country-year. The top-video table is useful because YouTube attention is highly unequal; a small number of videos can account for a large share of total views.

# %%
narrative_summary = summarize_youtube_narratives(df_youtube, narrative_cols)
display(narrative_summary)

plot_narratives = narrative_summary.sort_values("mention_rate_pct")
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(plot_narratives["label"], plot_narratives["mention_rate_pct"], color="#4C78A8")
ax.set_title("YouTube Narrative Mention Rate, 2018-2020")
ax.set_xlabel("Videos mentioning narrative (%)")
ax.set_ylabel("Narrative")
for y_pos, value in enumerate(plot_narratives["mention_rate_pct"]):
    ax.text(value + 0.8, y_pos, f"{value:.1f}%", va="center", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# This bar chart ranks YouTube narrative themes by how often they appear in video titles or descriptions from 2018 to 2020. Recipe/cooking is the most common narrative, followed by health, taste, and convenience/easy messages. This suggests that plant-based food content on YouTube is often framed around practical everyday use, not only around abstract values or ideology. Key takeaway: the most visible online narratives are about how to cook plant-based food, whether it is healthy, whether it tastes good, and whether it is easy to adopt.

# %% [markdown]
# Narrative Coding Codebook
#
# The following theme-expression figures summarize the codebook for the eleven YouTube narrative themes. Each figure shows the most frequent words or phrases that triggered one narrative code, and larger expressions or higher counts mean that the expression appeared in more videos. This helps make the YouTube coding process transparent because the themes are linked back to concrete title and description language rather than only researcher interpretation. Key takeaway: the narrative analysis is based on observable expressions such as recipe, healthy, delicious, easy, budget, and go vegan, which makes the later theme-level findings easier to interpret and evaluate.

# %%
codebook_path = PROJECT_ROOT / "data/Clean" / CODEBOOK_FILE
narrative_codebook = pd.read_csv(codebook_path, encoding="utf-8-sig")

codebook_view = narrative_codebook.copy()
codebook_view["theme"] = codebook_view["theme_column"].apply(format_label)
codebook_view["share_of_valid_videos_pct"] = codebook_view["share_of_valid_videos"] * 100
display(codebook_view[[
    "theme",
    "videos_flagged",
    "share_of_valid_videos_pct",
    "top_matched_expressions",
]])

codebook_expressions = parse_codebook_expressions(narrative_codebook)
display(codebook_expressions.head(20))

plot_theme_expression_clouds(codebook_expressions, max_terms_per_theme=7)
plt.show()

# %%
year_rates = aggregate_youtube_narratives(df_youtube, narrative_cols, ("Year",)).set_index("Year")
year_rate_table = year_rates[ols_cols].mul(100)
year_rate_table.columns = [format_label(col) for col in year_rate_table.columns]
display(year_rate_table.round(1))

top_narratives = narrative_summary.head(4)["narrative"].tolist()
top_year_cols = [format_label(f"{col}_rate") for col in top_narratives]
ax = year_rate_table[top_year_cols].plot(figsize=(9, 5), marker="o")
ax.set_title("Most Common YouTube Narratives by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Videos mentioning narrative (%)")
ax.set_xticks(year_rate_table.index)
ax.legend(title="Narrative", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.78, 1])
plt.show()

# %% [markdown]
# This line chart tracks the most common YouTube narrative themes across 2018, 2019, and 2020. Recipe/cooking remains the leading theme throughout the period, showing that practical food preparation is consistently central to plant-based YouTube content. Taste and convenience/easy messages become more visible by 2020, suggesting a shift toward everyday adoption and consumer experience. Key takeaway: YouTube narratives increasingly make plant-based eating feel practical, familiar, and easier to try.

# %%
sales_country_year_for_merge = prepare_sales_country_year_value(
    pd.read_csv(PROJECT_ROOT / "data/Clean/plant_based_food_sales_data.csv")
)
narrative_country_year = aggregate_youtube_narratives(df_youtube, narrative_cols)
narrative_sales_country_year = merge_narrative_sales(narrative_country_year, sales_country_year_for_merge)
narrative_sales_country_year = add_log_total_value(narrative_sales_country_year)

missing_sales_country_year = (
    narrative_country_year[["Country", "Year"]]
    .merge(sales_country_year_for_merge[["Country", "Year"]], on=["Country", "Year"], how="left", indicator=True)
    .query("_merge == 'left_only'")
    .drop(columns="_merge")
)

print("Country-year observations in YouTube data:", len(narrative_country_year))
print("Country-year observations used in sales relationship analysis:", len(narrative_sales_country_year))
print("Country-year observations without usable sales value:")
display(missing_sales_country_year)
display(narrative_sales_country_year.head())

# %%
corr_table = pd.DataFrame({
    "narrative": narrative_cols,
    "label": [format_label(col) for col in narrative_cols],
    "corr_with_log_total_value": [
        narrative_sales_country_year[f"{col}_rate"].corr(narrative_sales_country_year["log_total_value"])
        for col in narrative_cols
    ],
}).sort_values("corr_with_log_total_value", ascending=False)

display(corr_table)

plot_corr = corr_table.sort_values("corr_with_log_total_value")
colors = ["#4C78A8" if value >= 0 else "#E45756" for value in plot_corr["corr_with_log_total_value"]]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(plot_corr["label"], plot_corr["corr_with_log_total_value"], color=colors)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Correlation Between Narrative Mention Rate and Log Sales Value")
ax.set_xlabel("Correlation with log(Total Value EUR)")
ax.set_ylabel("Narrative")
plt.tight_layout()
plt.show()

# %% [markdown]
# This correlation chart gives an exploratory first look at how each YouTube narrative mention rate is associated with country-year log sales value. Taste, budget/shopping, and convenience/easy show the strongest positive correlations in this sample, meaning these themes tend to appear more often in higher-sales country-year markets. Environment is slightly negative here, which does not mean environmental messaging is unimportant, but it is not the narrative most closely aligned with higher sales in this dataset. The chart does not control for overlap between narratives, so it should be interpreted as a directional screening step rather than causal evidence. Key takeaway: practical consumer-facing messages seem more closely connected with higher sales than purely value-based messages in this exploratory analysis.

# %%
regression_data = narrative_sales_country_year[
    ["Country", "Year", "Total Value EUR", "log_total_value"] + ols_cols
].dropna().copy()

adjusted_ranked_models, adjusted_best_rate_cols, adjusted_best_ols_model = run_theme_only_ols_search(
    regression_data,
    ols_cols,
    fit_ols,
)

display(adjusted_ranked_models.drop(columns="rate_cols").head(20))
display(summarize_ols_choice(adjusted_best_ols_model, adjusted_best_rate_cols))

adjusted_best_ols_coefficients = create_ols_coefficient_table(
    adjusted_best_ols_model,
    adjusted_best_rate_cols,
    scale=0.10,
).rename(columns={
    "coefficient": "coefficient_per_10pp",
    "conf_low": "conf_low_per_10pp",
    "conf_high": "conf_high_per_10pp",
})
display(adjusted_best_ols_coefficients.sort_values("coefficient_per_10pp", ascending=False))

print("Highest adjusted R-squared theme-only OLS summary")
print(adjusted_best_ols_model.summary())

# %%
plot_ols_coefficients(
    adjusted_best_ols_coefficients,
    "Highest Adjusted R-Squared Theme-Only OLS",
    "#4C78A8",
)
plt.show()

# %% [markdown]
# This coefficient plot shows the best adjusted R-squared theme-only OLS model selected from the coded YouTube narrative rates. The selected themes are taste, challenge/transition, and budget/shopping, and all three coefficients are positive. This means that country-year markets with more content about taste, beginner-friendly transition, and affordable shopping tend to have higher plant-based food sales value, after considering these selected themes together. Because the sample is small and observational, the result should be read as an exploratory association rather than proof of causality. Key takeaway: the strongest model points toward practical adoption messaging: plant-based food should be presented as tasty, easy to start, and financially accessible.

# %% [markdown]
# ## Population-Scaled Sales Adjustment (Group B Feedback)
#
# Following today's discussion and Group B's suggestion, this added section scales the sales data by national population before repeating the narrative-sales correlation check. The original sales analysis above is kept unchanged because total sales still measure market size. This adjustment adds a second view: sales intensity per person.
#
# Population data are pulled from the World Bank API using the World Development Indicators population series `SP.POP.TOTL` for 2018-2020.

# %%
import json
import time
import urllib.request

WORLD_BANK_POPULATION_INDICATOR = "SP.POP.TOTL"
WORLD_BANK_POPULATION_SOURCE = "World Bank, World Development Indicators, SP.POP.TOTL"
WORLD_BANK_COUNTRY_CODES = {
    "Austria": "AUT",
    "Belgium": "BEL",
    "Denmark": "DNK",
    "France": "FRA",
    "Germany": "DEU",
    "Italy": "ITA",
    "Netherlands": "NLD",
    "Poland": "POL",
    "Romania": "ROU",
    "Spain": "ESP",
    "United Kingdom": "GBR",
}


def fetch_world_bank_population(country_code_map, start_year=2018, end_year=2020):
    """Fetch country-year population from the World Bank API."""
    iso_to_country = {iso: country for country, iso in country_code_map.items()}
    country_code_path = ";".join(sorted(iso_to_country))
    url = (
        "https://api.worldbank.org/v2/country/"
        f"{country_code_path}/indicator/{WORLD_BANK_POPULATION_INDICATOR}"
        f"?format=json&date={start_year}:{end_year}&per_page=1000"
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "group-i-population-scaling-analysis"},
    )
    last_error = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            break
        except Exception as error:
            last_error = error
            if attempt == 3:
                raise RuntimeError(
                    "World Bank population API request failed after 3 attempts. "
                    "Re-run this cell with internet access."
                ) from last_error
            print(f"World Bank API attempt {attempt} failed; retrying...")
            time.sleep(2)

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError("World Bank API did not return the expected population payload.")

    rows = []
    for item in payload[1]:
        iso_code = item.get("countryiso3code")
        population = item.get("value")
        if iso_code in iso_to_country and population is not None:
            rows.append(
                {
                    "Country": iso_to_country[iso_code],
                    "Year": int(item["date"]),
                    "iso3_code": iso_code,
                    "population": population,
                    "population_source": WORLD_BANK_POPULATION_SOURCE,
                }
            )

    if not rows:
        raise RuntimeError("World Bank API returned no usable population rows.")

    population_df = pd.DataFrame(rows)
    expected = pd.MultiIndex.from_product(
        [sorted(country_code_map), range(start_year, end_year + 1)],
        names=["Country", "Year"],
    ).to_frame(index=False)

    population_df = expected.merge(population_df, on=["Country", "Year"], how="left")
    population_df["population"] = pd.to_numeric(
        population_df["population"], errors="coerce"
    ).astype("Int64")
    return population_df, url
# %%
sales_countries_for_population = sorted(sales_country_year_for_merge["Country"].unique())
missing_country_codes = sorted(
    set(sales_countries_for_population) - set(WORLD_BANK_COUNTRY_CODES)
)
if missing_country_codes:
    raise KeyError(f"Missing World Bank ISO3 codes for: {missing_country_codes}")

population_country_year, world_bank_population_url = fetch_world_bank_population(
    {
        country: WORLD_BANK_COUNTRY_CODES[country]
        for country in sales_countries_for_population
    },
    start_year=2018,
    end_year=2020,
)

print("World Bank population API:", world_bank_population_url)
display(population_country_year.sort_values(["Country", "Year"]))

missing_population_rows = population_country_year[
    population_country_year["population"].isna()
]
if len(missing_population_rows):
    print("Missing population rows:")
    display(missing_population_rows)

# %%
population_scaled_sales = narrative_sales_country_year.merge(
    population_country_year[["Country", "Year", "population"]],
    on=["Country", "Year"],
    how="left",
)
population_scaled_sales = population_scaled_sales.dropna(subset=["population"]).copy()
population_scaled_sales["population"] = pd.to_numeric(
    population_scaled_sales["population"], errors="coerce"
)
population_scaled_sales["value_per_capita_eur"] = (
    population_scaled_sales["Total Value EUR"] / population_scaled_sales["population"]
)
population_scaled_sales["value_per_100k_people_eur"] = (
    population_scaled_sales["value_per_capita_eur"] * 100_000
)
population_scaled_sales["log_value_per_capita"] = np.log(
    population_scaled_sales["value_per_capita_eur"]
)

display(
    population_scaled_sales[
        [
            "Country",
            "Year",
            "Total Value EUR",
            "population",
            "value_per_capita_eur",
            "value_per_100k_people_eur",
            "log_value_per_capita",
        ]
    ]
    .sort_values(["Country", "Year"])
    .round(
        {
            "Total Value EUR": 0,
            "value_per_capita_eur": 2,
            "value_per_100k_people_eur": 0,
            "log_value_per_capita": 3,
        }
    )
)

# %%
country_per_capita_sales = (
    population_scaled_sales
    .groupby("Country")["value_per_capita_eur"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(country_per_capita_sales.index, country_per_capita_sales.values, color="#54A24B")
ax.set_title("Average Plant-Based Food Sales per Capita, 2018-2020")
ax.set_xlabel("Country")
ax.set_ylabel("Mean sales value per person (EUR)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %%
population_scaled_corr_table = pd.DataFrame({
    "narrative": narrative_cols,
    "label": [format_label(col) for col in narrative_cols],
    "corr_with_log_sales_per_capita": [
        population_scaled_sales[f"{col}_rate"].corr(
            population_scaled_sales["log_value_per_capita"]
        )
        for col in narrative_cols
    ],
}).sort_values("corr_with_log_sales_per_capita", ascending=False)

if "corr_table" in globals():
    population_corr_comparison = corr_table[
        ["narrative", "label", "corr_with_log_total_value"]
    ].merge(
        population_scaled_corr_table,
        on=["narrative", "label"],
        how="inner",
    )
    display(
        population_corr_comparison.sort_values(
            "corr_with_log_sales_per_capita", ascending=False
        )
    )
else:
    display(population_scaled_corr_table)

plot_population_corr = population_scaled_corr_table.sort_values(
    "corr_with_log_sales_per_capita"
)
colors = [
    "#4C78A8" if value >= 0 else "#E45756"
    for value in plot_population_corr["corr_with_log_sales_per_capita"]
]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(
    plot_population_corr["label"],
    plot_population_corr["corr_with_log_sales_per_capita"],
    color=colors,
)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Correlation Between Narrative Mention Rate and Population-Scaled Sales")
ax.set_xlabel("Correlation with log(sales EUR per capita)")
ax.set_ylabel("Narrative")
plt.tight_layout()
plt.show()

# %% [markdown]
# This population-scaled check responds directly to today's discussion and Group B's suggestion. After dividing sales by World Bank population, the analysis shifts from total market size to market intensity. In the current 2018-2020 data, the Netherlands and Belgium become especially visible on a per-capita basis, while larger countries no longer dominate only because they have more people.
#
# The correlation chart also changes the narrative screen: fitness/weight, challenge/transition, and health show stronger positive correlations with log sales per capita than they did in the total-value view. This should still be interpreted as exploratory association, not causal evidence, because the country-year sample is small. Key takeaway: population scaling adds a fairer cross-country comparison and helps separate absolute market size from per-person adoption intensity.

# %% [markdown]
# # NEW OLS REGRESSION SECTION: Unscaled vs Population-Scaled Sales
#
# The population-scaled section above added a fairer per-person sales measure, but correlation alone is not enough. The following cells are the new OLS regression cells. This follow-up adds OLS regression and compares two outcomes directly:
#
# - unscaled market size: `log_total_value`
# - population-scaled market intensity: `log_value_per_capita`
#
# To make the first comparison fair, both models use the same country-year observations and the same narrative predictors selected by the earlier unscaled best adjusted R-squared model.

# %%
comparison_rate_cols = (
    list(adjusted_best_rate_cols)
    if "adjusted_best_rate_cols" in globals()
    else ["taste_rate", "challenge_transition_rate", "budget_shopping_rate"]
)
comparison_rate_cols = [
    col for col in comparison_rate_cols
    if col in population_scaled_sales.columns
]

ols_scale_comparison_data = population_scaled_sales[
    [
        "Country",
        "Year",
        "Total Value EUR",
        "population",
        "log_total_value",
        "log_value_per_capita",
    ] + comparison_rate_cols
].replace([np.inf, -np.inf], np.nan).dropna().copy()

unscaled_same_predictor_model = fit_ols(
    ols_scale_comparison_data,
    comparison_rate_cols,
    "log_total_value",
)
scaled_same_predictor_model = fit_ols(
    ols_scale_comparison_data,
    comparison_rate_cols,
    "log_value_per_capita",
)

same_predictor_ols_comparison = pd.DataFrame([
    {
        "model": "Unscaled OLS",
        "dependent_variable": "log_total_value",
        "meaning": "absolute market size",
        "n_observations": int(unscaled_same_predictor_model.nobs),
        "n_predictors": len(comparison_rate_cols),
        "r_squared": unscaled_same_predictor_model.rsquared,
        "adjusted_r_squared": unscaled_same_predictor_model.rsquared_adj,
        "aic": unscaled_same_predictor_model.aic,
        "bic": unscaled_same_predictor_model.bic,
        "predictors": ", ".join(format_label(col) for col in comparison_rate_cols),
    },
    {
        "model": "Population-scaled OLS",
        "dependent_variable": "log_value_per_capita",
        "meaning": "sales intensity per person",
        "n_observations": int(scaled_same_predictor_model.nobs),
        "n_predictors": len(comparison_rate_cols),
        "r_squared": scaled_same_predictor_model.rsquared,
        "adjusted_r_squared": scaled_same_predictor_model.rsquared_adj,
        "aic": scaled_same_predictor_model.aic,
        "bic": scaled_same_predictor_model.bic,
        "predictors": ", ".join(format_label(col) for col in comparison_rate_cols),
    },
])

display(same_predictor_ols_comparison)

print("Unscaled OLS: same predictors, outcome = log_total_value")
print(unscaled_same_predictor_model.summary())
print("\nPopulation-scaled OLS: same predictors, outcome = log_value_per_capita")
print(scaled_same_predictor_model.summary())

# %%
unscaled_same_predictor_coefficients = create_ols_coefficient_table(
    unscaled_same_predictor_model,
    comparison_rate_cols,
    scale=0.10,
).rename(columns={
    "coefficient": "unscaled_coefficient_per_10pp",
    "p_value": "unscaled_p_value",
    "conf_low": "unscaled_conf_low_per_10pp",
    "conf_high": "unscaled_conf_high_per_10pp",
})

scaled_same_predictor_coefficients = create_ols_coefficient_table(
    scaled_same_predictor_model,
    comparison_rate_cols,
    scale=0.10,
).rename(columns={
    "coefficient": "scaled_coefficient_per_10pp",
    "p_value": "scaled_p_value",
    "conf_low": "scaled_conf_low_per_10pp",
    "conf_high": "scaled_conf_high_per_10pp",
})

same_predictor_coefficient_comparison = (
    unscaled_same_predictor_coefficients[
        [
            "variable",
            "label",
            "unscaled_coefficient_per_10pp",
            "unscaled_p_value",
            "unscaled_conf_low_per_10pp",
            "unscaled_conf_high_per_10pp",
        ]
    ]
    .merge(
        scaled_same_predictor_coefficients[
            [
                "variable",
                "scaled_coefficient_per_10pp",
                "scaled_p_value",
                "scaled_conf_low_per_10pp",
                "scaled_conf_high_per_10pp",
            ]
        ],
        on="variable",
        how="inner",
    )
)

same_predictor_coefficient_comparison["coefficient_change_after_scaling"] = (
    same_predictor_coefficient_comparison["scaled_coefficient_per_10pp"]
    - same_predictor_coefficient_comparison["unscaled_coefficient_per_10pp"]
)

display(same_predictor_coefficient_comparison.sort_values(
    "scaled_coefficient_per_10pp",
    ascending=False,
))

plot_coef_compare = same_predictor_coefficient_comparison.sort_values(
    "scaled_coefficient_per_10pp"
)
y_positions = np.arange(len(plot_coef_compare))
bar_height = 0.36

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(
    y_positions - bar_height / 2,
    plot_coef_compare["unscaled_coefficient_per_10pp"],
    height=bar_height,
    label="Unscaled: log total sales",
    color="#4C78A8",
)
ax.barh(
    y_positions + bar_height / 2,
    plot_coef_compare["scaled_coefficient_per_10pp"],
    height=bar_height,
    label="Scaled: log sales per capita",
    color="#54A24B",
)
ax.axvline(0, color="black", linewidth=1)
ax.set_yticks(y_positions)
ax.set_yticklabels(plot_coef_compare["label"])
ax.set_xlabel("OLS coefficient per 10 pp higher mention rate")
ax.set_title("Same Predictors: Unscaled vs Population-Scaled OLS")
ax.legend()
plt.tight_layout()
plt.show()

# %%
scaled_regression_data = population_scaled_sales[
    [
        "Country",
        "Year",
        "Total Value EUR",
        "population",
        "log_value_per_capita",
    ] + ols_cols
].replace([np.inf, -np.inf], np.nan).dropna().copy()

scaled_ranked_models, scaled_best_rate_cols, scaled_best_ols_model = run_theme_only_ols_search(
    scaled_regression_data,
    ols_cols,
    fit_ols,
    y_col="log_value_per_capita",
)

display(scaled_ranked_models.drop(columns="rate_cols").head(20))
display(
    summarize_ols_choice(
        scaled_best_ols_model,
        scaled_best_rate_cols,
        model_name="Highest adjusted R-squared population-scaled OLS",
    )
)

scaled_best_ols_coefficients = create_ols_coefficient_table(
    scaled_best_ols_model,
    scaled_best_rate_cols,
    scale=0.10,
).rename(columns={
    "coefficient": "coefficient_per_10pp",
    "conf_low": "conf_low_per_10pp",
    "conf_high": "conf_high_per_10pp",
})
display(scaled_best_ols_coefficients.sort_values(
    "coefficient_per_10pp",
    ascending=False,
))

best_model_scale_comparison = pd.DataFrame([
    {
        "model": "Original unscaled best OLS",
        "dependent_variable": "log_total_value",
        "meaning": "absolute market size",
        "adjusted_r_squared": adjusted_best_ols_model.rsquared_adj,
        "r_squared": adjusted_best_ols_model.rsquared,
        "selected_themes": ", ".join(format_label(col) for col in adjusted_best_rate_cols),
    },
    {
        "model": "Population-scaled best OLS",
        "dependent_variable": "log_value_per_capita",
        "meaning": "sales intensity per person",
        "adjusted_r_squared": scaled_best_ols_model.rsquared_adj,
        "r_squared": scaled_best_ols_model.rsquared,
        "selected_themes": ", ".join(format_label(col) for col in scaled_best_rate_cols),
    },
])
display(best_model_scale_comparison)

print("Highest adjusted R-squared population-scaled OLS summary")
print(scaled_best_ols_model.summary())

# %%
plot_ols_coefficients(
    scaled_best_ols_coefficients,
    "Highest Adjusted R-Squared Population-Scaled OLS",
    "#54A24B",
)
plt.show()

# %% [markdown]
# ## Visible OLS Regression Result Summary
#
# This is the new OLS regression block added after the population-scaling feedback section. The code cells immediately above/below this note run the actual `statsmodels` OLS models and print the full regression summaries. This table is included so the new OLS results are visible even before re-running the notebook.
#
# ### Same Predictors, Different Outcome
#
# Both models use the same predictors selected by the original unscaled best model: `taste_rate`, `challenge_transition_rate`, and `budget_shopping_rate`.
#
# | OLS model | Outcome | Meaning | N | R-squared | Adjusted R-squared |
# |---|---|---|---:|---:|---:|
# | Unscaled OLS | `log_total_value` | Absolute market size | 30 | 0.410 | 0.342 |
# | Population-scaled OLS | `log_value_per_capita` | Sales intensity per person | 30 | 0.149 | 0.051 |
#
# | Predictor | Unscaled coefficient per 10pp | Population-scaled coefficient per 10pp | Interpretation of change |
# |---|---:|---:|---|
# | Taste | 0.952 | 0.329 | The association weakens after population scaling, suggesting part of the unscaled relationship reflects larger markets. |
# | Challenge Transition | 1.163 | 0.973 | The association remains positive after scaling, so it is still relevant for per-person market intensity. |
# | Budget Shopping | 1.760 | 0.394 | The association becomes much smaller after scaling, suggesting the original relationship is more tied to absolute market size. |
#
# ### Best Population-Scaled OLS
#
# When the dependent variable changes to `log_value_per_capita`, the highest adjusted R-squared model selects: `environment_rate`, `recipe_cooking_rate`, `fitness_weight_rate`, and `budget_shopping_rate`.
#
# | Population-scaled best OLS | R-squared | Adjusted R-squared |
# |---|---:|---:|
# | `log_value_per_capita` model | 0.302 | 0.191 |
#
# | Selected predictor | Coefficient per 10pp |
# |---|---:|
# | Environment | 0.982 |
# | Recipe Cooking | 0.249 |
# | Fitness Weight | 1.296 |
# | Budget Shopping | 1.142 |
#
# Key takeaway: the unscaled OLS is about absolute market opportunity, while the population-scaled OLS is about per-person adoption intensity. Scaling changes the interpretation and also changes which themes look strongest.

# %% [markdown]
# This OLS comparison separates two different questions. The unscaled model asks which YouTube narratives are associated with larger total markets. The population-scaled model asks which narratives are associated with stronger per-person adoption intensity after accounting for country population size.
#
# The same-predictor comparison is the most direct before/after scaling check because the rows and predictors are held constant. If a coefficient becomes smaller after scaling, part of the original relationship was likely connected to country size. If a coefficient stays positive or becomes stronger, that narrative is more closely related to per-capita market intensity, not only to large-population markets.
#
# The best-model comparison then shows whether the themes selected by adjusted R-squared change when the dependent variable changes from total sales to sales per capita. Key takeaway: population scaling does not replace the original OLS; it adds a second interpretation layer. The unscaled model is about absolute opportunity size, while the scaled model is about adoption intensity and fairer cross-country comparison.
