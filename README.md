# Video-Analysis--DataScience-Group

# Examining the Impact of YouTube Narrative Information on Plant-Based Food Sales

## 👥 Team Members

- CONG Tianxiang (@congtx) - Lead Python development; YouTube API integration; data pipeline design; advanced scripting; code quality review.
- Euimin Keum (@github_handle) - Run and modify basic scripts; support data cleaning, manual validation, coding checks, and documentation of preprocessing steps.
- HUANG Shuo (@huangshuojason) - Econometric modeling; variable definition; regression analysis; interpretation of results; Python/R notebook implementation.


## Updated on May 19th

Milestone 1: Due May 20th (Week 5)
Choose a dataset relevant to the course content.
https://zenodo.org/records/6411841

Formulate a specific, testable research question.
- ❓ Research Question & 🎯 Hypothesis
- RQ: How does narrative information on the YouTube platform affect the sales of plant-based foods?
- Hypothesis: The volume of positive narratives about plant-based products on YouTube is positively associated with plant-based food sales.
Baseline: Using Python import data and show descriptive statistics.
How many countries, descriptive stats of key variables (count, mean, median, stdev, min, max)


-------------------------------------------


## 📁 Data Sources

| Source | Description | URL |
|--------|-------------|-----|
| YouTube Data API | Primary data source. Used to collect video metadata for plant-based food-related YouTube videos, including title, description, publication date, channel information, views, likes, and comments. | https://developers.google.com/youtube/v3 |
| Keyword Dictionary | A project-created dictionary for plant-based food keywords and narrative classification keywords |Stored in docs/data_details.md or src/config/ |

### Data Sources Details

D.1 YouTube Data API
Item	Planned Details
Unit of analysis	Individual YouTube video.
Market focus	Plant-based food products, including plant-based meat, vegan meat, meat alternatives, dairy alternatives, and sustainable food products.
Possible search keywords	plant-based meat; vegan meat; meat alternative; Beyond Meat; Impossible Foods; plant-based burger; vegan burger; sustainable food; alternative protein.
Geographic scope	To be decided based on feasibility. A global English-language sample is the simplest option. If needed, regionCode can be set in the API.
Granularity	Video-level cross-sectional data; if collected over several weeks, video-week or weekly aggregated analysis may also be possible.
Core variables	video_id, title, description, tags, published_at, channel_id, channel_title, duration, view_count, like_count, comment_count.
Derived variables	Narrative dummies: health, taste, environmental, skeptical/negative; log views; log likes; log comments; engagement rate; title length; upload timing.

D.2 Narrative Classification Data
Narrative Type	Definition	Example Keywords / Clues
Health-related	Videos that frame plant-based foods through health, nutrition, personal well-being, diet, or bodily benefits/risks.	healthy, nutrition, protein, diet, wellness, heart health, ultra-processed, ingredients.
Taste-related	Videos that frame plant-based foods through flavor, texture, sensory experience, cooking, or similarity to meat.	taste test, delicious, flavor, texture, juicy, like meat, recipe, cooking.
Environmental	Videos that frame plant-based foods through sustainability, climate, emissions, animals, ethics, or environmental impact.	sustainable, climate, carbon, emissions, planet, environment, animal welfare.
Skeptical / Negative	Videos that frame plant-based foods critically or controversially.	fake meat, artificial, unhealthy, expensive, scam, controversy, backlash.


## 📂 Folder Structure

### Folder Structure Notes
- All projects MUST follow this standardized folder structure
- `data/raw/` - **NEVER** edit manually; store original data here
- `data/clean/` - Cleaned datasets ready for analysis
- `data/temp/` - Temporary files (can be deleted)
- `notebooks/` - Jupyter notebooks for analysis
- `src/` - Python code
- `reports/` - Final outputs: plots, summaries, model files
- `docs/` - Project documentation, README, presentations

### Folder Structure Tree

```tree
project/
├── data/
│   ├── raw/                   # Original, immutable data
│   │   ├── world_bank_raw.csv
│   │   └── imf_financials_raw.csv
│   ├── clean/                 # Cleaned, transformed data
│   │   ├── world_bank_clean.csv
│   │   └── imf_merged_clean.csv
│   └── temp/                  # Temporary working files
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_eda_worldbank.ipynb
│   ├── 02_regression_analysis.ipynb
│   └── 03_policy_simulations.ipynb
├── src/                       # Production-ready scripts
│   ├── download_worldbank.py  # API/Scraping script
│   ├── clean_data.py          # Merging and cleaning logic
│   └── visualize_worldbank.py # Chart generation functions
├── reports/                   # Final outputs
│   ├── figures/               # Saved .png plots for the memo
│   │   ├── gdp_trend_line.png
│   │   └── debt_distribution.png
│   ├── policy_memo_final.pdf
│   └── regression_results.txt
└── docs/                      # Documentation
    ├── data_details.md        # Data dictionary & column definitions
    ├── data_architecture.md   # Pipeline logic and join keys
    ├── policy_context.md      # Political background & stakeholders
```

## 📅 Timeline

| Milestone | Deadline | Deliverable |
|-----------|----------|-------------|
| M1        | Date     | Output      |
| M2        | Date     | Output      |
| M3        | Date     | Output      |

## 🤝 Contributions

| Member | Tasks |
|--------|-------|
| CONG Tianxiang|  Lead Python development; YouTube API integration; data pipeline design; advanced scripting; code quality review. |
| Euimin Keum   | Run and modify basic scripts; support data cleaning, manual validation, coding checks, and documentation of preprocessing  |
| HUANG Shuo| Econometric modeling; variable definition; regression analysis; interpretation of results; Python/R notebook implementation.s |

## 🔗 References
- YouTube Data API Documentation: https://developers.google.com/youtube/v3
- Pandas Documentation: https://pandas.pydata.org/docs/
- Statsmodels Documentation: https://www.statsmodels.org/
- Suggested methodology topics: social media analytics, narrative/framing analysis, sentiment analysis, econometric modeling of online engagement, and consumer attention proxies.
