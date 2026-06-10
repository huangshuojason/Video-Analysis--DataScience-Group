# Video-Analysis--DataScience-Group
# Plant-Based Food Sales and YouTube Narratives in Europe: An Exploratory Analysis from 2018 to 2020

Project Summary
This project examines how plant-based food markets and YouTube narratives developed across selected European countries from 2018 to 2020. It combines country-year-product sales data with YouTube video text data to compare market size, product-category composition, narrative prevalence, and the relationship between online narratives and plant-based food sales. Since sales value and volume are strongly correlated, the later analysis focuses mainly on Value EUR as the key market indicator.

## 👥 Team Members
Each member’s contribution will be added after their tasks are completed.
- CONG Tianxiang (@congtx) - Research design; Theory development; YouTube API data extraction; Code review
- HUANG Shuo (@huangshuojason) - Research design; YouTube API data extraction; Panel data acquisition; visualization
- Okamoto Junichi (@okamoto-junichi22-debug)- Research design

## Updated on June 10th

- ❓ Research Question
- Main RQ: How are YouTube narratives about plant-based foods associated with plant-based food sales patterns across selected European countries from 2018 to 2020?
- Sub-RQ1: How do plant-based food sales values vary across countries, years, and product groups?
- Sub-RQ2: How does the product-category composition of plant-based food sales differ across countries and over time?
- Sub-RQ3: Which YouTube narratives are most frequently mentioned across countries and years?
- Sub-RQ4: To what extent are YouTube narrative mentions associated with total plant-based food sales value at the country-year level?



-------------------------------------------


## 📁 Data Sources

| Source | Description | URL |
|--------|-------------|-----|
| YouTube Data API | Used to collect YouTube video metadata related to plant-based food narratives, including video title, description, upload date, channel information, views, likes, and comments. | [https://developers.google.com/youtube/v3](https://developers.google.com/youtube/v3) |
| Project-created Narrative Keyword Dictionary |  A keyword-based dictionary created for this project to classify YouTube video narratives into health, sustainability, environment, hedonism, and animal welfare themes. | Created in the analysis code|
| European Plant-Based Foods Sales Data  | ESales data for plant-based food products across European countries. Key variables include country, year, product group, Value EUR, and Volume kg/l. |[[Stored in docs/data_details.md or src/config/](https://zenodo.org/records/6411841) ](https://zenodo.org/records/6411841 )|

### Data Sources Details

| Item | Details |
|------|---------|
| Unit of analysis | Individual YouTube video |
| Market focus | Plant-based food products, including plant-based meat, vegan food, meat alternatives, dairy alternatives, and plant-based diets |
| Geographic scope | Selected European countries, including Austria, Belgium, Denmark, France, Italy, Netherlands, Romania, Spain, and United Kingdom |
| Time period | Videos published between 2018 and 2020 |
| Search keywords | Search terms included plant-based food, plant-based meat, vegan food, vegan products, meat alternatives, plant-based diet, supermarket, sustainability, and country-specific terms |
| Core variables | video_id, title, description, upload_date, country, channel, channel_id, view_count, like_count, comment_count |
| Derived variables | Year, narrative keyword counts, narrative mentioned dummies, narrative intensity per video, and country-year narrative summaries |

#### D.2 Narrative Classification Data

| Narrative Type | Definition | Example Keywords / Clues |
|----------------|------------|---------------------------|
| Health | Videos that frame plant-based foods through health, nutrition, diet, personal well-being, or bodily benefits | health, healthy, nutrition, protein, diet, wellness, organic |
| Sustainability | Videos that frame plant-based foods through sustainable consumption, ethical responsibility, or future-oriented food systems | sustainable, sustainability, eco, green, ethical, responsible |
| Environment | Videos that discuss climate, emissions, pollution, biodiversity, land use, or environmental impacts | environment, climate, carbon, CO2, emissions, planet, pollution |
| Hedonism | Videos that frame plant-based foods through taste, pleasure, texture, enjoyment, or sensory experience | taste, tasty, delicious, flavour, juicy, crispy, creamy |
| Animal Welfare | Videos that frame plant-based foods through animal welfare, cruelty-free consumption, factory farming, or ethical treatment of animals | animal welfare, cruelty free, slaughter, factory farming, livestock |

#### D.3 European Plant-Based Foods Sales Data

The sales dataset contains information on plant-based food products sold in European countries. The original dataset covers plant-based food sales between 2017 and 2020, while this project focuses on the 2018-2020 period to match the YouTube data collection window.

The main variables used in this analysis are:

| Variable | Description |
|----------|-------------|
| Country | Country where the sales data were recorded |
| Year | Year of observation |
| Product Group | Category of plant-based food product |
| Value EUR | Sales value measured in euros |
| Volume kg/l | Sales volume measured in kilograms or litres |

The sales data are used to examine differences in plant-based food markets across countries, years, and product groups. The YouTube narrative data are then aggregated at the country-year level and merged with the sales data to explore possible correlations between online narratives and plant-based food sales performance.


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
| M1 | 5.20 | Research design; sales data acquisition; descriptive analysis of sales data |
| M2 | 6.3 | YouTube API data collection; narrative keyword analysis; visualizations; correlation and regression analysis |
| M3        | Date     | Output      |

## 🤝 Contributions
Each member’s contribution will be added after their tasks are completed.
| Member | Tasks |
|--------|-------|
| CONG Tianxiang|  Research design; Theory development; YouTube API data extraction; advanced code review|  
| HUANG Shuo| Research design; YouTube API data extraction; Panel data acquisition|  Visualization
| Okamoto Junichi | Research design


## 🔗 References
- YouTube Data API Documentation: https://developers.google.com/youtube/v3
- Panel data: European plant based foods sales data: [https://zenodo.org/records/641184](https://zenodo.org/records/6411841)
