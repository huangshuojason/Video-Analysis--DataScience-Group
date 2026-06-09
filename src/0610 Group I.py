# %% [markdown]
# Project Summary
# 
# This project investigates the relationship between plant-based food sales and
# YouTube narratives across European countries from 2018 to 2020. The analysis
# combines two types of data: sales data for plant-based food products and
# YouTube video metadata collected through the YouTube Data API.
# 
# The sales analysis examines descriptive patterns across countries, years, and
# product groups. Average sales value in EUR and sales volume in kg/l are compared
# to identify market differences between countries, changes over time, and
# variation across product categories.
# 
# The YouTube analysis focuses on narrative themes in video titles and
# descriptions. Five narrative categories are measured using keyword-based text
# analysis: health, sustainability, environment, hedonism, and animal welfare.
# These narrative variables are summarized by country and year to explore how
# plant-based food discussions differ across markets and change over time.
# 
# Finally, the sales data and YouTube narrative data are merged at the
# country-year level. Correlation and regression analyses are used to explore
# whether YouTube attention and narrative intensity are associated with sales
# value and sales volume. The results are interpreted as exploratory rather than
# causal, since sales performance may also be influenced by prices, product
# availability, market maturity, and retail conditions.
# 
# - Main RQ: How are plant-based food sales patterns related to YouTube narratives across European countries from 2018 to 2020?
# - Sub-RQ1: How do plant-based food sales differ across countries, years, and product groups?
# - Sub-RQ2: What narratives are most commonly used in YouTube videos about plant-based food across countries and over time?
# - Sub-RQ3: To what extent are YouTube narrative variables associated with plant-based food sales value and sales volume?

# %%
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt

print(f"current working directory:{os.getcwd()}")
print("Files in current directory:")
print(os.listdir("."))

# %%
df = pd.read_csv("../data/Clean/plant_based_food_sales_data.csv")
def basics(df):
    print(df.info())
    print(df.head())
basics(df)

# %%
df= df[['Country', 'Year', 'Value EUR', 'Volume kg/l', 'Product Group']]
df = df.dropna()
df = df[(df['Value EUR'] != 0) & (df['Volume kg/l'] != 0)]
df = (df.groupby(['Country', 'Year', 'Product Group'], as_index=False)[['Value EUR', 'Volume kg/l']].sum())

# %%
def details(df):
    print('Head')
    print(df.head(2))
    print('Tail')
    print(df.tail(2))
    print("\nUnique countries in the dataset:")
    print(df.Country.unique())
    print("\nUnique years in the dataset:")
    print(df.Year.unique())
    print("\nUnique product groups in the dataset:")
    print(df['Product Group'].unique())
details(df)

# %%
df[['Value EUR', 'Volume kg/l']].hist(figsize=(10, 4))

# %% [markdown]
# The histograms show that both sales value and sales volume are strongly right-skewed, with most observations concentrated at lower levels and only a small number of observations showing very large values. This indicates substantial variation across countries, years, and product categories. Since the dataset contains both value and volume information, the subsequent analysis examines whether higher sales volumes are associated with higher sales values, which would suggest a broadly positive value–volume relationship. Applying log transformation helps reduce the influence of extreme observations and makes the variables more suitable for regression analysis when testing this relationship.

# %%
import numpy as np
df['log_value'] = np.log(df['Value EUR'])
df['log_volume'] = np.log(df['Volume kg/l'])
df[['log_value', 'log_volume']].hist(figsize=(10, 4))

# %% [markdown]
# After log transformation, the distributions of log_value and log_volume become less right-skewed compared with the original variables. The values are more spread out and less dominated by extreme observations, making them more suitable for correlation and OLS regression analysis.

# %%
import statsmodels.api as sm
X = sm.add_constant(df[['log_volume']])
y = df['log_value']
model_log = sm.OLS(y, X).fit()
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
df_wide_value = df.pivot(
    index=['Country', 'Year'],
    columns='Product Group',
    values='Value EUR')
df_wide_value = df_wide_value.reset_index()
df_wide_value.columns.name = None
df_wide_value.head()

# %%
df_wide_value['Total Value EUR'] = (df_wide_value.drop(columns=['Country', 'Year']).sum(axis=1))
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
product_cols = df_wide_value.columns.drop(['Country', 'Year', 'Total Value EUR'])
df_share = df_wide_value.copy()
df_share[product_cols] = (df_wide_value[product_cols].div(df_wide_value['Total Value EUR'], axis=0) * 100)
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
# The following analysis is based on video information related to plant-based foods collected using the YouTube API. Since the YouTube API requires a personal API key, and repeated execution may trigger quota limits, the API extraction code is not executed in the following notebook. Instead, the extracted CSV dataset is provided in the data/Raw directory. The original Python source code used for API extraction can be found in the src directory.

# %%
from pathlib import Path

# Folder where your country-level YouTube CSV files are saved
folder = Path("../data/Raw")

# Find all country-level CSV files
csv_files = list(folder.glob("youtube_plant_based_*.csv"))

# Exclude already-combined files if they exist
csv_files = [
    file for file in csv_files
    if "all_countries" not in file.name
]

print("Number of CSV files found:", len(csv_files))

for file in csv_files:
    print(file.name)

if len(csv_files) == 0:
    raise ValueError("No CSV files found. Please check your folder path.")

df_list = []

for file in csv_files:
    temp_df = pd.read_csv(file)

    # Add source file name for tracking
    temp_df["source_file"] = file.name

    df_list.append(temp_df)

# Combine all CSV files
df_all_youtube = pd.concat(df_list, ignore_index=True)

# Remove duplicated rows if the same video appears more than once for the same country
if {"country", "video_id"}.issubset(df_all_youtube.columns):
    df_all_youtube = df_all_youtube.drop_duplicates(
        subset=["country", "video_id"]
    )
else:
    print("Warning: country or video_id column not found. Skipping duplicate removal by country and video_id.")

# Save combined CSV
output_path = folder / "youtube_plant_based_all_countries.csv"

df_all_youtube.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nDone.")
print("Total rows:", len(df_all_youtube))
print("Saved to:", output_path)

display(df_all_youtube.head())

# %%
import re


# Load combined YouTube data
youtube_path = Path("../data/Raw/youtube_plant_based_all_countries.csv")

df_youtube = pd.read_csv(youtube_path)

print(df_youtube.shape)
display(df_youtube.head())

# %%
# Make sure date/year exists
df_youtube["upload_date"] = pd.to_datetime(df_youtube["upload_date"], errors="coerce")
df_youtube["Year"] = df_youtube["upload_date"].dt.year

# Combine text columns for narrative analysis
text_cols = ["title", "description", "summary"]

for col in text_cols:
    if col not in df_youtube.columns:
        df_youtube[col] = ""

df_youtube["text_for_analysis"] = (
    df_youtube[text_cols]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

# %%
narrative_keywords = {
    "health": [
        "health", "healthy", "nutrition", "nutritious", "protein", "diet",
        "wellbeing", "wellness", "fitness", "low fat", "cholesterol",
        "sugar free", "natural", "organic",
        "santé", "salud", "gesund", "gesundheit", "salute", "gezond",
    ],

    "sustainability": [
        "sustainable", "sustainability", "eco", "green", "ethical",
        "responsible", "future", "planet friendly",
        "durable", "durabilité", "sostenible", "nachhaltig",
        "sostenibilità", "duurzaam",
    ],

    "environment": [
        "environment", "environmental", "climate", "carbon", "co2",
        "emissions", "greenhouse gas", "planet", "earth", "pollution",
        "biodiversity", "water use", "land use",
        "climat", "clima", "klima", "ambiente", "milieu",
    ],

    "hedonism": [
        "taste", "tasty", "delicious", "flavour", "flavor", "yummy",
        "enjoy", "pleasure", "craving", "juicy", "crispy", "creamy",
        "comfort food", "indulgent",
        "goût", "délicieux", "sabor", "rico", "sabroso",
        "geschmack", "lecker", "gusto", "smaak", "lekker",
    ],

    "animal_welfare": [
        "animal welfare", "animal", "animals", "cruelty", "cruelty free",
        "ethical", "slaughter", "factory farming", "livestock",
        "cows", "pigs", "chickens",
        "bien-être animal", "bienestar animal", "tierschutz",
        "benessere animale", "dierenwelzijn",
    ],
}

# %%
# Make sure upload_date and Year are available
df_youtube["upload_date"] = pd.to_datetime(df_youtube["upload_date"], errors="coerce")
df_youtube["Year"] = df_youtube["upload_date"].dt.year

# Make sure text columns exist
text_cols = ["title", "description"]

for col in text_cols:
    if col not in df_youtube.columns:
        df_youtube[col] = ""

# Combine title and description for narrative analysis
df_youtube["text_for_analysis"] = (
    df_youtube[text_cols]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

# %%
import re

def count_keywords(text, keywords):
    if pd.isna(text):
        return 0
    
    text = str(text).lower()
    count = 0
    
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        count += len(re.findall(pattern, text))
    
    return count


for narrative, keywords in narrative_keywords.items():
    df_youtube[f"{narrative}_count"] = df_youtube["text_for_analysis"].apply(
        lambda x: count_keywords(x, keywords)
    )

df_youtube.head()

# %%
for narrative in narrative_keywords.keys():
    df_youtube[f"{narrative}_mentioned"] = (
        df_youtube[f"{narrative}_count"] > 0
    ).astype(int)

df_youtube.head()

# %%
narrative_count_cols = [
    f"{narrative}_count"
    for narrative in narrative_keywords.keys()
]

narrative_mentioned_cols = [
    f"{narrative}_mentioned"
    for narrative in narrative_keywords.keys()
]

print(narrative_count_cols)
print(narrative_mentioned_cols)

# %%
country_narrative_summary = df_youtube.groupby("country").agg(
    video_count=("video_id", "nunique"),
    **{col: (col, "sum") for col in narrative_count_cols},
    **{col: (col, "mean") for col in narrative_mentioned_cols}
).reset_index()

country_narrative_summary

# %%
country_year_narrative_summary = df_youtube.groupby(["country", "Year"]).agg(
    video_count=("video_id", "nunique"),
    **{col: (col, "sum") for col in narrative_count_cols},
    **{col: (col, "mean") for col in narrative_mentioned_cols}
).reset_index()

country_year_narrative_summary

# %%
from pathlib import Path

output_folder = Path("../data/Raw")

enriched_path = output_folder / "youtube_plant_based_all_countries_with_narratives.csv"
country_summary_path = output_folder / "youtube_narrative_summary_by_country.csv"
country_year_summary_path = output_folder / "youtube_narrative_summary_by_country_year.csv"

df_youtube.to_csv(enriched_path, index=False, encoding="utf-8-sig")
country_narrative_summary.to_csv(country_summary_path, index=False, encoding="utf-8-sig")
country_year_narrative_summary.to_csv(country_year_summary_path, index=False, encoding="utf-8-sig")

print("Saved enriched video-level data to:", enriched_path)
print("Saved country summary to:", country_summary_path)
print("Saved country-year summary to:", country_year_summary_path)

# %%

plot_df = country_narrative_summary.set_index("country")[narrative_count_cols]

plot_df.plot(
    kind="bar",
    figsize=(14, 6)
)

plt.title("Narrative Keyword Counts by Country")
plt.xlabel("Country")
plt.ylabel("Keyword Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# This figure shows the frequency of different narrative keywords in YouTube videos across countries. Health-related keywords appear most frequently in almost all countries, suggesting that health is the dominant narrative in plant-based food videos. France, Italy, the Netherlands, Spain, and the United Kingdom show relatively high overall narrative counts. Sustainability and environmental narratives are also visible, especially in the United Kingdom and the Netherlands. Hedonism-related keywords are particularly high in Italy, while animal welfare is more prominent in the Netherlands, Spain, and the United Kingdom. Overall, the results suggest that health is the main narrative, but the emphasis on other narratives varies across countries.

# %%
trend_df = country_year_narrative_summary.groupby("Year")[narrative_count_cols].sum()

trend_df.plot(
    kind="line",
    marker="o",
    figsize=(10, 6)
)

plt.title("Narrative Keyword Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Keyword Count")
plt.xticks([2018, 2019, 2020])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# This figure shows how narrative keyword counts changed from 2018 to 2020. Health-related keywords increased strongly over time and remained the most dominant narrative in every year. Sustainability and environmental narratives also rose steadily, suggesting growing attention to ecological concerns. Hedonism-related keywords dipped slightly in 2019 but increased again in 2020, while animal welfare showed a more moderate upward trend. Overall, the results indicate that plant-based food narratives became more prominent over time, especially around health, sustainability, and environmental themes.

# %%
sales_year_country = df.groupby(['Country', 'Year'])[['Value EUR', 'Volume kg/l']].mean().reset_index()

sales_year_country.head()

# %%
narrative_path = Path("../data/Raw/youtube_narrative_summary_by_country_year.csv")

narrative_df = pd.read_csv(narrative_path)

narrative_df = narrative_df.rename(columns={
    "country": "Country"
})

narrative_df.head()

# %%
narrative_df["health_per_video"] = narrative_df["health_count"] / narrative_df["video_count"]

narrative_df["sustainability_environment_per_video"] = (
    narrative_df["sustainability_count"] + narrative_df["environment_count"]
) / narrative_df["video_count"]

narrative_df.head()

# %%
merged_df = pd.merge(
    sales_year_country,
    narrative_df,
    on=["Country", "Year"],
    how="inner"
)

print(merged_df.shape)
display(merged_df.head())

# %%
corr1_df = merged_df[["health_per_video", "Value EUR"]].dropna()

X = sm.add_constant(corr1_df["health_per_video"])
y = corr1_df["Value EUR"]

model_health_value = sm.OLS(y, X).fit()

print(model_health_value.summary())

# %%
plt.figure(figsize=(8, 6))

sns.regplot(
    data=corr1_df,
    x="health_per_video",
    y="Value EUR",
    scatter_kws={"alpha": 0.7},
    line_kws={"color": "red"}
)

plt.title("Correlation between Health Narrative and Value EUR")
plt.xlabel("Health keyword count per video")
plt.ylabel("Average Value EUR")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# The OLS regression examines whether health-related YouTube narrative intensity is associated with average sales value in EUR. The coefficient for health_per_video is positive, suggesting that country-year observations with more health-related keywords per video tend to have higher average sales value. However, the relationship is not statistically significant, as the p-value is 0.337, which is above the 0.05 threshold. The R-squared value is also low at 0.037, meaning that health narrative intensity explains only about 3.7% of the variation in Value EUR.
# 
# The scatter plot shows the same pattern: although the regression line slopes upward, the points are widely dispersed and the confidence interval is broad. Therefore, the result should be interpreted as weak and exploratory. In this dataset, health narratives appear to have a positive but non-significant relationship with sales value.

# %%
corr2_df = merged_df[["sustainability_environment_per_video", "Volume kg/l"]].dropna()

X = sm.add_constant(corr2_df["sustainability_environment_per_video"])
y = corr2_df["Volume kg/l"]

model_sustainability_volume = sm.OLS(y, X).fit()

print(model_sustainability_volume.summary())

# %%
plt.figure(figsize=(8, 6))

sns.regplot(
    data=corr2_df,
    x="sustainability_environment_per_video",
    y="Volume kg/l",
    scatter_kws={"alpha": 0.7},
    line_kws={"color": "red"}
)

plt.title("Correlation between Sustainability/Environment Narrative and Volume")
plt.xlabel("Sustainability + environment keyword count per video")
plt.ylabel("Average Volume kg/l")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# 
# The regression examines whether sustainability and environment narrative intensity is related to average sales volume. The coefficient is negative, suggesting a weak negative relationship between sustainability/environment keywords per video and volume. However, the p-value is 0.565, which is above 0.05, so this relationship is not statistically significant.
# 
# The R-squared value is only 0.013, meaning that this narrative variable explains very little variation in sales volume. Overall, the result suggests that sustainability and environmental narratives are not strongly associated with sales volume in this dataset.

# %%
sales_year_country = df.groupby(['Country', 'Year'])[['Value EUR', 'Volume kg/l']].mean().reset_index()

narrative_path = "../data/Raw/youtube_narrative_summary_by_country_year.csv"
narrative_df = pd.read_csv(narrative_path)

narrative_df = narrative_df.rename(columns={"country": "Country"})

merged_df = pd.merge(
    sales_year_country,
    narrative_df,
    on=["Country", "Year"],
    how="inner"
)

# %%
narrative_cols = [
    "health_count",
    "sustainability_count",
    "environment_count",
    "hedonism_count",
    "animal_welfare_count"
]

for col in narrative_cols:
    merged_df[col.replace("_count", "_per_video")] = (
        merged_df[col] / merged_df["video_count"]
    )

merged_df["total_narrative_per_video"] = (
    merged_df["health_per_video"] +
    merged_df["sustainability_per_video"] +
    merged_df["environment_per_video"] +
    merged_df["hedonism_per_video"] +
    merged_df["animal_welfare_per_video"]
)

# %%
corr_vars = [
    "Value EUR",
    "Volume kg/l",
    "video_count",
    "health_per_video",
    "sustainability_per_video",
    "environment_per_video",
    "hedonism_per_video",
    "animal_welfare_per_video",
    "total_narrative_per_video"
]

corr_matrix = merged_df[corr_vars].corr()

corr_matrix

# %%
plt.figure(figsize=(11, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix: Sales and YouTube Narratives")
plt.tight_layout()
plt.show()

# %% [markdown]
# This correlation matrix shows the relationships between sales indicators and YouTube narrative variables. The strongest relationship is between Value EUR and Volume kg/l with a correlation of 0.94, indicating that higher sales volume is strongly associated with higher market value. In contrast, the correlations between sales variables and narrative variables are generally weak. For example, Value EUR has only weak positive correlations with video_count, health_per_video, hedonism_per_video, and total_narrative_per_video.
# 
# The matrix also shows that some narrative categories are strongly related to each other. For example, health_per_video is highly correlated with total_narrative_per_video, and environment_per_video is also strongly related to total narrative intensity. Overall, the results suggest that sales value and volume are closely connected, while YouTube narratives have weaker and more exploratory associations with sales performance.

# %%
sales_vars = ["Value EUR", "Volume kg/l"]

narrative_vars = [
    "video_count",
    "health_per_video",
    "sustainability_per_video",
    "environment_per_video",
    "hedonism_per_video",
    "animal_welfare_per_video",
    "total_narrative_per_video"
]

sales_narrative_corr = merged_df[sales_vars + narrative_vars].corr().loc[
    sales_vars,
    narrative_vars
]

plt.figure(figsize=(12, 4))

ax = sns.heatmap(
    sales_narrative_corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation between Sales Variables and YouTube Narratives", pad=15)

# Make y-axis labels readable
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=11
)

# Make x-axis labels readable
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    fontsize=10
)

# Add more margin on the left and bottom
plt.subplots_adjust(left=0.18, bottom=0.35, right=0.95, top=0.85)

plt.show()

# %% [markdown]
# This heatmap shows that the correlations between sales variables and YouTube narratives are generally weak. Value EUR has the strongest positive correlation with video_count at 0.34, suggesting that higher YouTube attention is somewhat associated with higher sales value. Hedonism, animal welfare, and total narrative intensity also show weak positive correlations with Value EUR. In contrast, sustainability has a weak negative correlation with both Value EUR and Volume kg/l. Overall, YouTube narratives show only limited correlation with sales performance.

# %%
corr_long = sales_narrative_corr.stack().reset_index()
corr_long.columns = ["sales_variable", "narrative_variable", "correlation"]

corr_long["abs_correlation"] = corr_long["correlation"].abs()

top_corr = corr_long.sort_values(
    "abs_correlation",
    ascending=False
).head(2)

top_corr


# %%
for _, row in top_corr.iterrows():
    sales_var = row["sales_variable"]
    narrative_var = row["narrative_variable"]
    corr_value = row["correlation"]

    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=merged_df,
        x=narrative_var,
        y=sales_var,
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "red"}
    )

    plt.title(
        f"{sales_var} and {narrative_var}\nCorrelation = {corr_value:.2f}"
    )
    plt.xlabel(narrative_var.replace("_", " "))
    plt.ylabel(sales_var)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# These two scatter plots show weak positive relationships between YouTube activity/narratives and sales value. The first plot shows that video_count has a modest positive correlation with Value EUR (r = 0.34), suggesting that country-year groups with more YouTube videos tend to have higher sales value. The second plot shows a weaker positive correlation between hedonism_per_video and Value EUR (r = 0.25), indicating that taste or enjoyment-related narratives may be slightly associated with higher sales value. However, both relationships are relatively weak, so they should be interpreted as exploratory rather than strong evidence.


