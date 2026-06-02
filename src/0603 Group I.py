# Converted from notebooks/0603 Group I.ipynb

# %% Cell 1: markdown
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


# %% Cell 2: code
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt

print(f"current working directory:{os.getcwd()}")
print("\nFiles in current directory:")
print(os.listdir("."))


# %% Cell 3: code
df = pd.read_csv("../data/Clean/plant_based_food_sales_data.csv")
df.info()


# %% Cell 4: code
df= df[['Country', 'Year', 'Value EUR', 'Volume kg/l', 'Product Group']]
df = df.dropna()
df = df[(df['Value EUR'] != 0) & (df['Volume kg/l'] != 0)]
def basics(df):
    print(df.info())
    print(df.head())
basics(df)


# %% Cell 5: code
def basics(df):
    print('')
    print('Head')
    print(df.head(2))
    print('')
    print('Tail')
    print(df.tail(2))
    print('')
    print("\nUnique countries in the dataset:")
    print(df.Country.unique())
    print('')
    print("\nUnique years in the dataset:")
    print(df.Year.unique())
    print('')
    print("\nUnique product groups in the dataset:")
    print(df['Product Group'].unique())
basics(df)


# %% Cell 6: code
country_descriptive = df.groupby('Country')[['Value EUR', 'Volume kg/l']].describe()
country_descriptive


# %% Cell 7: code
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

country_mean['Value EUR'].sort_values(ascending=False).plot(
    kind='bar',
    ax=axes[0],
    color='steelblue'
)

axes[0].set_title('Average Value EUR by Country')
axes[0].set_xlabel('Country')
axes[0].set_ylabel('Average Value EUR')
axes[0].tick_params(axis='x', rotation=45)

country_mean['Volume kg/l'].sort_values(ascending=False).plot(
    kind='bar',
    ax=axes[1],
    color='darkorange'
)

axes[1].set_title('Average Volume kg/l by Country')
axes[1].set_xlabel('Country')
axes[1].set_ylabel('Average Volume kg/l')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# %% Cell 8: markdown
# This above figure compares the average market value and average sales volume of plant-based food products across countries. Spain shows the highest average value in EUR and the highest average volume kg/l, indicating that it has the largest market scale among the countries included in the dataset. Italy, France, the United Kingdom, Belgium, and the Netherlands form a middle group, with relatively strong average value and volume compared with Austria, Romania, and Denmark. Overall, the two charts show a similar pattern: countries with higher average value also tend to have higher average volume, suggesting that larger monetary sales are generally associated with larger physical sales volumes. Spain is the clear outlier in both measures.


# %% Cell 9: code
year_descriptive = df.groupby('Year')[['Value EUR', 'Volume kg/l']].describe()
year_descriptive


# %% Cell 10: code
import matplotlib.pyplot as plt

year_mean = df.groupby('Year')[['Value EUR', 'Volume kg/l']].mean()

years = [2018, 2019, 2020]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Value EUR trend
axes[0].plot(
    year_mean.index,
    year_mean['Value EUR'],
    marker='o',
    linewidth=2,
    color='steelblue'
)

axes[0].set_title('Average Value EUR by Year')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Average Value EUR')
axes[0].set_xticks(years)
axes[0].set_xticklabels(['2018', '2019', '2020'])
axes[0].grid(True, alpha=0.3)

for x, y in zip(year_mean.index, year_mean['Value EUR']):
    axes[0].text(x, y, f'{y:,.0f}', ha='center', va='bottom')

# Volume kg/l trend
axes[1].plot(
    year_mean.index,
    year_mean['Volume kg/l'],
    marker='o',
    linewidth=2,
    color='darkorange'
)

axes[1].set_title('Average Volume kg/l by Year')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Volume kg/l')
axes[1].set_xticks(years)
axes[1].set_xticklabels(['2018', '2019', '2020'])
axes[1].grid(True, alpha=0.3)

for x, y in zip(year_mean.index, year_mean['Volume kg/l']):
    axes[1].text(x, y, f'{y:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# %% Cell 11: markdown
# This figure shows the yearly trend in average market value and average sales volume of plant-based food products from 2018 to 2020. Both indicators increased over the period. The average value rose from approximately EUR 7.34 million in 2018 to EUR 7.63 million in 2019, then increased more sharply to around EUR 8.95 million in 2020. A similar pattern can be seen in average volume, which remained relatively stable between 2018 and 2019, increasing only slightly from about 2.61 million kg/l to 2.63 million kg/l, before rising more noticeably to about 3.00 million kg/l in 2020. Overall, the results suggest that the plant-based food market expanded over time, with the strongest growth occurring between 2019 and 2020.


# %% Cell 12: code
Product_Group_descriptive = df.groupby('Product Group')[['Value EUR', 'Volume kg/l']].describe()
Product_Group_descriptive


# %% Cell 13: code
import matplotlib.pyplot as plt

product_mean = df.groupby('Product Group')[['Value EUR', 'Volume kg/l']].mean()

value_sorted = product_mean.sort_values('Value EUR', ascending=True)
volume_sorted = product_mean.sort_values('Volume kg/l', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

bars1 = axes[0].barh(
    value_sorted.index,
    value_sorted['Value EUR'],
    color='steelblue'
)

axes[0].set_title('Average Value EUR by Product Group')
axes[0].set_xlabel('Average Value EUR')
axes[0].set_ylabel('Product Group')
axes[0].grid(axis='x', alpha=0.3)

for bar in bars1:
    width = bar.get_width()
    axes[0].text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f'{width:,.0f}',
        va='center',
        ha='left'
    )

bars2 = axes[1].barh(
    volume_sorted.index,
    volume_sorted['Volume kg/l'],
    color='darkorange'
)

axes[1].set_title('Average Volume kg/l by Product Group')
axes[1].set_xlabel('Average Volume kg/l')
axes[1].set_ylabel('Product Group')
axes[1].grid(axis='x', alpha=0.3)

for bar in bars2:
    width = bar.get_width()
    axes[1].text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f'{width:,.0f}',
        va='center',
        ha='left'
    )

plt.tight_layout()
plt.show()


# %% Cell 14: markdown
# This figure compares the average market value and average sales volume across different plant-based product groups. Plant-based drinks and milk alternatives are the leading category in both measures, with the highest average value and by far the highest average volume. Plant-based yogurt is the second largest category, showing relatively strong performance in both value and volume. Meat alternatives also have a high average value, but their average volume is much lower than drinks and yogurt, suggesting a higher value per unit or a smaller but more valuable market segment. Cheese alternatives and plant-based ice cream are in the middle to lower range, while the general plant-based alternatives category records the lowest average value and volume. Overall, the results show that plant-based drinks and milk alternatives dominate the dataset, especially in terms of physical sales volume.


# %% Cell 15: code
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
reg_df = df[['Value EUR', 'Volume kg/l']].dropna()

# Remove zero or negative values if needed
reg_df = reg_df[
    (reg_df['Value EUR'] > 0) &
    (reg_df['Volume kg/l'] > 0)
]

# Define X and y
X = reg_df[['Volume kg/l']]
y = reg_df['Value EUR']

# Add constant / intercept
X = sm.add_constant(X)

# Fit OLS regression model
model = sm.OLS(y, X).fit()

# Show regression result
print(model.summary())

plt.figure(figsize=(8, 6))

sns.regplot(
    data=reg_df,
    x='Volume kg/l',
    y='Value EUR',
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red'}
)

plt.title('OLS Regression: Value EUR and Volume kg/l')
plt.xlabel('Volume kg/l')
plt.ylabel('Value EUR')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% Cell 16: markdown
# The OLS regression results show a positive and statistically significant relationship between sales volume and value in EUR. The coefficient of Volume kg/l is 1.5706, meaning that, on average, a one-unit increase in sales volume is associated with an increase of about 1.57 EUR in value. The p-value is 0.000, which is below the 0.05 significance level, indicating that this positive relationship is statistically significant.
# The R-squared value is 0.682, meaning that approximately 68.2% of the variation in Value EUR can be explained by Volume kg/l. This suggests that sales volume is an important predictor of market value.
# Although this positive relationship is intuitive, it is still necessary to test it statistically. In theory, higher sales volume should be associated with higher sales value. However, the relationship may not always be perfectly consistent because price changes can also affect value. For example, if prices increase or decrease significantly, the change in value may not move exactly in line with the change in sales volume. Therefore, testing the relationship helps us understand whether sales value mainly follows sales volume, or whether price variation may also play an important role.


# %% Cell 17: markdown
# API


# %% Cell 18: markdown
# > **Note:** The following code was used to extract video information through the YouTube API. Since it requires a personal API key and repeated execution may trigger API quota limits, it is included as a Markdown section rather than an executable code cell. The CSV files generated from this process are then combined in the next step into a complete CSV file and stored in the `data/Raw` folder.
# 
# 
# import requests
# import pandas as pd
# import time
# from pathlib import Path
# from datetime import datetime
# 
# API_KEY = "PASTE YOUR OWN API_KEY"
# 
# PAGES_PER_QUERY = 1
# REQUEST_SLEEP_SECONDS = 2
# 
# PUBLISHED_AFTER = "2018-01-01T00:00:00Z"
# PUBLISHED_BEFORE = "2021-01-01T00:00:00Z"
# 
# OUTPUT_DIR =Path("../data/Raw")
# 
# COUNTRIES = [
#     {"country": "Austria", "region_code": "AT", "language": "de", "local_name": "Austria Osterreich"},
#     {"country": "Belgium", "region_code": "BE", "language": "nl", "local_name": "Belgium Belgie Belgique"},
#     {"country": "Denmark", "region_code": "DK", "language": "da", "local_name": "Denmark Danmark"},
#     {"country": "France", "region_code": "FR", "language": "fr", "local_name": "France"},
#     {"country": "Italy", "region_code": "IT", "language": "it", "local_name": "Italy Italia"},
#     {"country": "Netherlands", "region_code": "NL", "language": "nl", "local_name": "Netherlands Nederland"},
#     {"country": "Romania", "region_code": "RO", "language": "ro", "local_name": "Romania"},
#     {"country": "Spain", "region_code": "ES", "language": "es", "local_name": "Spain Espana"},
#     {"country": "United Kingdom", "region_code": "GB", "language": "en", "local_name": "United Kingdom UK Britain"},
# ]
# 
# # Change this later if you want another country
# SELECTED_COUNTRY_NAME = "United Kingdom"
# 
# QUERY_TEMPLATES = [
#     "plant based food {country} consumer behaviour",
#     "plant based meat {country} consumers",
#     "vegan food {country} consumer trends",
#     "plant based diet {country} supermarket",
#     "meat alternatives {country} shoppers",
#     "vegan products {country} consumers",
#     "plant based food {local_name} sustainability",
#     "vegan {local_name} supermarket",
# ]
# 
# 
# def get_country_config(country_name):
#     for country in COUNTRIES:
#         if country["country"] == country_name:
#             return country
#     raise ValueError(f"Country not found: {country_name}")
# 
# 
# def youtube_get(endpoint, params, max_retries=5):
#     url = f"https://www.googleapis.com/youtube/v3/{endpoint}"
#     params = dict(params)
#     params["key"] = API_KEY
# 
#     for attempt in range(max_retries):
#         response = requests.get(url, params=params, timeout=30)
# 
#         if response.status_code == 429:
#             wait_time = 30 * (attempt + 1)
#             print(f"429 Too Many Requests. Waiting {wait_time} seconds...")
#             time.sleep(wait_time)
#             continue
# 
#         if response.status_code == 403:
#             print("403 error. You may have reached your YouTube API quota limit.")
#             print(response.text)
#             response.raise_for_status()
# 
#         response.raise_for_status()
#         time.sleep(REQUEST_SLEEP_SECONDS)
#         return response.json()
# 
#     raise Exception("Too many requests. Please wait and run again later.")
# 
# 
# def search_videos(country_config):
#     video_rows = []
#     seen = set()
# 
#     queries = [
#         template.format(
#             country=country_config["country"],
#             local_name=country_config["local_name"]
#         )
#         for template in QUERY_TEMPLATES
#     ]
# 
#     for query in queries:
#         print("Query:", query)
# 
#         page_token = None
# 
#         for _ in range(PAGES_PER_QUERY):
#             params = {
#                 "part": "snippet",
#                 "type": "video",
#                 "q": query,
#                 "maxResults": 50,
#                 "regionCode": country_config["region_code"],
#                 "relevanceLanguage": country_config["language"],
#                 "publishedAfter": PUBLISHED_AFTER,
#                 "publishedBefore": PUBLISHED_BEFORE,
#                 "safeSearch": "none",
#                 "order": "relevance",
#             }
# 
#             if page_token:
#                 params["pageToken"] = page_token
# 
#             data = youtube_get("search", params)
# 
#             for item in data.get("items", []):
#                 video_id = item.get("id", {}).get("videoId")
# 
#                 if video_id and video_id not in seen:
#                     seen.add(video_id)
#                     video_rows.append({
#                         "country": country_config["country"],
#                         "search_region_code": country_config["region_code"],
#                         "search_language": country_config["language"],
#                         "video_id": video_id,
#                     })
# 
#             page_token = data.get("nextPageToken")
# 
#             if not page_token:
#                 break
# 
#     return video_rows
# 
# 
# def fetch_video_details(video_ids):
#     all_items = []
# 
#     for start in range(0, len(video_ids), 50):
#         chunk = video_ids[start:start + 50]
# 
#         if len(chunk) == 0:
#             continue
# 
#         params = {
#             "part": "snippet,statistics,contentDetails",
#             "id": ",".join(chunk),
#             "maxResults": 50,
#         }
# 
#         data = youtube_get("videos", params)
#         all_items.extend(data.get("items", []))
# 
#     return all_items
# 
# 
# def fetch_channel_details(channel_ids):
#     channels = {}
#     unique_channel_ids = list(set(channel_ids))
# 
#     for start in range(0, len(unique_channel_ids), 50):
#         chunk = unique_channel_ids[start:start + 50]
# 
#         if len(chunk) == 0:
#             continue
# 
#         params = {
#             "part": "snippet,statistics,brandingSettings",
#             "id": ",".join(chunk),
#             "maxResults": 50,
#         }
# 
#         data = youtube_get("channels", params)
# 
#         for item in data.get("items", []):
#             channels[item["id"]] = item
# 
#     return channels
# 
# 
# def clean_text(text):
#     if text is None:
#         return ""
#     return " ".join(str(text).replace("\n", " ").replace("\r", " ").split())
# 
# 
# def build_final_rows(search_rows, video_details, channel_details):
#     details_by_id = {
#         item["id"]: item
#         for item in video_details
#     }
# 
#     final_rows = []
# 
#     for row in search_rows:
#         video_id = row["video_id"]
#         item = details_by_id.get(video_id)
# 
#         if item is None:
#             continue
# 
#         snippet = item.get("snippet", {})
#         statistics = item.get("statistics", {})
# 
#         channel_id = snippet.get("channelId", "")
#         channel = channel_details.get(channel_id, {})
# 
#         channel_country = (
#             channel
#             .get("brandingSettings", {})
#             .get("channel", {})
#             .get("country", "")
#         )
# 
#         published_at = snippet.get("publishedAt", "")
# 
#         final_rows.append({
#             "country": row["country"],
#             "search_region_code": row["search_region_code"],
#             "search_language": row["search_language"],
#             "channel_country": channel_country,
#             "default_language": snippet.get("defaultLanguage", ""),
#             "default_audio_language": snippet.get("defaultAudioLanguage", ""),
#             "video_id": video_id,
#             "url": f"https://www.youtube.com/watch?v={video_id}",
#             "title": clean_text(snippet.get("title", "")),
#             "upload_date": published_at[:10],
#             "published_at": published_at,
#             "channel": snippet.get("channelTitle", ""),
#             "channel_id": channel_id,
#             "view_count": statistics.get("viewCount", ""),
#             "like_count": statistics.get("likeCount", ""),
#             "comment_count": statistics.get("commentCount", ""),
#             "description": clean_text(snippet.get("description", "")),
#         })
# 
#     return final_rows
# 
# 
# country_config = get_country_config(SELECTED_COUNTRY_NAME)
# 
# print("==============================")
# print(f"Searching {country_config['country']}")
# print("==============================")
# 
# search_rows = search_videos(country_config)
# search_df = pd.DataFrame(search_rows)
# 
# print("\nTotal rows:", len(search_df))
# 
# if len(search_df) > 0:
#     print("Unique videos:", search_df["video_id"].nunique())
# 
#     unique_video_ids = search_df["video_id"].drop_duplicates().tolist()
# 
#     print("\nFetching video details...")
#     video_details = fetch_video_details(unique_video_ids)
# 
#     channel_ids = [
#         item.get("snippet", {}).get("channelId")
#         for item in video_details
#         if item.get("snippet", {}).get("channelId")
#     ]
# 
#     print("Fetching channel details...")
#     channel_details = fetch_channel_details(channel_ids)
# 
#     final_rows = build_final_rows(
#         search_rows,
#         video_details,
#         channel_details
#     )
# 
#     df_youtube = pd.DataFrame(final_rows)
# 
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 
#     stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     country_file_name = SELECTED_COUNTRY_NAME.lower().replace(" ", "_")
# 
#     csv_path = OUTPUT_DIR / f"youtube_plant_based_{country_file_name}_{stamp}.csv"
#     json_path = OUTPUT_DIR / f"youtube_plant_based_{country_file_name}_{stamp}.json"
# 
#     df_youtube.to_csv(csv_path, index=False, encoding="utf-8-sig")
#     df_youtube.to_json(json_path, orient="records", force_ascii=False, indent=2)
# 
#     print("\nDone.")
#     print("Rows saved:", len(df_youtube))
#     print("CSV saved to:", csv_path)
#     print("JSON saved to:", json_path)
# 
#     display(df_youtube.head())
# 
# else:
#     print("No videos found. Try changing queries or checking your API key.")


# %% Cell 19: code
import pandas as pd
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


# %% Cell 20: code
import pandas as pd
import re
from pathlib import Path

# Load combined YouTube data
youtube_path = Path("../data/Raw/youtube_plant_based_all_countries.csv")

df_youtube = pd.read_csv(youtube_path)

print(df_youtube.shape)
display(df_youtube.head())


# %% Cell 21: code
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


# %% Cell 22: code
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


# %% Cell 23: code
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


# %% Cell 24: code
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


# %% Cell 25: code
for narrative in narrative_keywords.keys():
    df_youtube[f"{narrative}_mentioned"] = (
        df_youtube[f"{narrative}_count"] > 0
    ).astype(int)

df_youtube.head()


# %% Cell 26: code
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


# %% Cell 27: code
country_narrative_summary = df_youtube.groupby("country").agg(
    video_count=("video_id", "nunique"),
    **{col: (col, "sum") for col in narrative_count_cols},
    **{col: (col, "mean") for col in narrative_mentioned_cols}
).reset_index()

country_narrative_summary


# %% Cell 28: code
country_year_narrative_summary = df_youtube.groupby(["country", "Year"]).agg(
    video_count=("video_id", "nunique"),
    **{col: (col, "sum") for col in narrative_count_cols},
    **{col: (col, "mean") for col in narrative_mentioned_cols}
).reset_index()

country_year_narrative_summary


# %% Cell 29: code
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


# %% Cell 30: code

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


# %% Cell 31: markdown
# This figure shows the frequency of different narrative keywords in YouTube videos across countries. Health-related keywords appear most frequently in almost all countries, suggesting that health is the dominant narrative in plant-based food videos. France, Italy, the Netherlands, Spain, and the United Kingdom show relatively high overall narrative counts. Sustainability and environmental narratives are also visible, especially in the United Kingdom and the Netherlands. Hedonism-related keywords are particularly high in Italy, while animal welfare is more prominent in the Netherlands, Spain, and the United Kingdom. Overall, the results suggest that health is the main narrative, but the emphasis on other narratives varies across countries.


# %% Cell 32: code
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


# %% Cell 33: markdown
# This figure shows how narrative keyword counts changed from 2018 to 2020. Health-related keywords increased strongly over time and remained the most dominant narrative in every year. Sustainability and environmental narratives also rose steadily, suggesting growing attention to ecological concerns. Hedonism-related keywords dipped slightly in 2019 but increased again in 2020, while animal welfare showed a more moderate upward trend. Overall, the results indicate that plant-based food narratives became more prominent over time, especially around health, sustainability, and environmental themes.


# %% Cell 34: code
sales_year_country = df.groupby(['Country', 'Year'])[['Value EUR', 'Volume kg/l']].mean().reset_index()

sales_year_country.head()


# %% Cell 35: code
narrative_path = Path("../data/Raw/youtube_narrative_summary_by_country_year.csv")

narrative_df = pd.read_csv(narrative_path)

narrative_df = narrative_df.rename(columns={
    "country": "Country"
})

narrative_df.head()


# %% Cell 36: code
narrative_df["health_per_video"] = narrative_df["health_count"] / narrative_df["video_count"]

narrative_df["sustainability_environment_per_video"] = (
    narrative_df["sustainability_count"] + narrative_df["environment_count"]
) / narrative_df["video_count"]

narrative_df.head()


# %% Cell 37: code
merged_df = pd.merge(
    sales_year_country,
    narrative_df,
    on=["Country", "Year"],
    how="inner"
)

print(merged_df.shape)
display(merged_df.head())


# %% Cell 38: code
corr1_df = merged_df[["health_per_video", "Value EUR"]].dropna()

X = sm.add_constant(corr1_df["health_per_video"])
y = corr1_df["Value EUR"]

model_health_value = sm.OLS(y, X).fit()

print(model_health_value.summary())


# %% Cell 39: code
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


# %% Cell 40: markdown
# The OLS regression examines whether health-related YouTube narrative intensity is associated with average sales value in EUR. The coefficient for health_per_video is positive, suggesting that country-year observations with more health-related keywords per video tend to have higher average sales value. However, the relationship is not statistically significant, as the p-value is 0.337, which is above the 0.05 threshold. The R-squared value is also low at 0.037, meaning that health narrative intensity explains only about 3.7% of the variation in Value EUR.
# 
# The scatter plot shows the same pattern: although the regression line slopes upward, the points are widely dispersed and the confidence interval is broad. Therefore, the result should be interpreted as weak and exploratory. In this dataset, health narratives appear to have a positive but non-significant relationship with sales value.


# %% Cell 41: code
corr2_df = merged_df[["sustainability_environment_per_video", "Volume kg/l"]].dropna()

X = sm.add_constant(corr2_df["sustainability_environment_per_video"])
y = corr2_df["Volume kg/l"]

model_sustainability_volume = sm.OLS(y, X).fit()

print(model_sustainability_volume.summary())


# %% Cell 42: code
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


# %% Cell 43: markdown
# 
# The regression examines whether sustainability and environment narrative intensity is related to average sales volume. The coefficient is negative, suggesting a weak negative relationship between sustainability/environment keywords per video and volume. However, the p-value is 0.565, which is above 0.05, so this relationship is not statistically significant.
# 
# The R-squared value is only 0.013, meaning that this narrative variable explains very little variation in sales volume. Overall, the result suggests that sustainability and environmental narratives are not strongly associated with sales volume in this dataset.


# %% Cell 44: code
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


# %% Cell 45: code
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


# %% Cell 46: code
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


# %% Cell 47: code
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


# %% Cell 48: markdown
# This correlation matrix shows the relationships between sales indicators and YouTube narrative variables. The strongest relationship is between Value EUR and Volume kg/l with a correlation of 0.94, indicating that higher sales volume is strongly associated with higher market value. In contrast, the correlations between sales variables and narrative variables are generally weak. For example, Value EUR has only weak positive correlations with video_count, health_per_video, hedonism_per_video, and total_narrative_per_video.
# 
# The matrix also shows that some narrative categories are strongly related to each other. For example, health_per_video is highly correlated with total_narrative_per_video, and environment_per_video is also strongly related to total narrative intensity. Overall, the results suggest that sales value and volume are closely connected, while YouTube narratives have weaker and more exploratory associations with sales performance.


# %% Cell 49: code
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


# %% Cell 50: markdown
# This heatmap shows that the correlations between sales variables and YouTube narratives are generally weak. Value EUR has the strongest positive correlation with video_count at 0.34, suggesting that higher YouTube attention is somewhat associated with higher sales value. Hedonism, animal welfare, and total narrative intensity also show weak positive correlations with Value EUR. In contrast, sustainability has a weak negative correlation with both Value EUR and Volume kg/l. Overall, YouTube narratives show only limited correlation with sales performance.


# %% Cell 51: code
corr_long = sales_narrative_corr.stack().reset_index()
corr_long.columns = ["sales_variable", "narrative_variable", "correlation"]

corr_long["abs_correlation"] = corr_long["correlation"].abs()

top_corr = corr_long.sort_values(
    "abs_correlation",
    ascending=False
).head(2)

top_corr


# %% Cell 52: code
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


# %% Cell 53: markdown
# These two scatter plots show weak positive relationships between YouTube activity/narratives and sales value. The first plot shows that video_count has a modest positive correlation with Value EUR (r = 0.34), suggesting that country-year groups with more YouTube videos tend to have higher sales value. The second plot shows a weaker positive correlation between hedonism_per_video and Value EUR (r = 0.25), indicating that taste or enjoyment-related narratives may be slightly associated with higher sales value. However, both relationships are relatively weak, so they should be interpreted as exploratory rather than strong evidence.

