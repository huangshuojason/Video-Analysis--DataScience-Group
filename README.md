# Plant-Based Food Sales and YouTube Narratives in Europe

This project explores how plant-based food markets and YouTube narratives developed across selected European countries from 2018 to 2020. It combines country-year sales data with YouTube video metadata and text-based narrative labels to compare market size, product-category composition, narrative prevalence, and the relationship between online narratives and plant-based food sales.

The current analysis focuses on `Value EUR` as the main sales indicator because sales value and sales volume are strongly correlated in the cleaned sales dataset.

## Research Questions

- Main RQ: How are YouTube narratives about plant-based foods associated with plant-based food sales patterns across selected European countries from 2018 to 2020?
- Sub-RQ1: How do plant-based food sales values vary across countries, years, and product groups?
- Sub-RQ2: How does the product-category composition of plant-based food sales differ across countries and over time?
- Sub-RQ3: Which YouTube narrative dimensions are most frequently mentioned across countries and years?
- Sub-RQ4: To what extent are the seven combined YouTube narrative dimensions associated with total plant-based food sales value at the country-year level?

## Team Members

| Member | Main Contributions |
|--------|--------------------|
| CONG Tianxiang (@congtx) | Research design; theory development; YouTube API data extraction; code review; narrative coding|
| HUANG Shuo (@huangshuojason) | Research design; YouTube API data extraction; panel data acquisition; visualization; narrative coding update |
| Okamoto Junichi (@okamoto-junichi22-debug) | Research design; code review |

## Timeline / Milestones

This milestone table follows the course project template and records how the project developed from research design to the current 7-dimension narrative analysis.

| Milestone | Deadline / Date | Deliverable | Status |
|-----------|-----------------|-------------|--------|
| M1 | 5.20 | Research design; sales data acquisition; initial descriptive analysis of plant-based food sales by country, year, and product group | Completed |
| M2 | 6.3 | YouTube API data collection; initial narrative keyword analysis; first visualizations; preliminary correlation and regression analysis | Completed |
| M3 | 6.10 | Integrated sales and YouTube narrative analysis in the group notebook; refined research questions and interpretation of key figures | Completed |
| M4 | 6.24 | Updated the YouTube workflow to use the cleaned aspect-sentiment dataset directly; combined positive and negative labels into seven narrative dimensions; updated notebook analysis, figures, interpretations, and README | Completed |
| Final | TBA | Final notebook, presentation/report materials, and cleaned project repository for submission | In progress |

## Current Analysis Workflow

The main notebook is:

- `notebooks/0627 Group I.ipynb`

The notebook is organized into two main parts:

1. Sales analysis
   - Load the cleaned plant-based food sales dataset.
   - Aggregate sales to country-year-product group level.
   - Compare total sales value across countries and years.
   - Examine product-category composition using share plots.

2. YouTube narrative analysis
   - Load the cleaned and aspect-coded YouTube dataset directly from `data/Clean`.
   - Filter YouTube videos to 2018-2020 to match the sales data period.
   - Analyze seven narrative dimensions by combining each positive and negative pair into one narrative variable.
   - Show representative video titles for each narrative category.
   - Visualize narrative counts and mention rates by year and country.
   - Merge country-year narrative measures with country-year sales value.
   - Estimate OLS regressions between narrative mention rates and log total sales value.
   - Add short markdown interpretations after each generated figure.

The YouTube narrative analysis now uses direct 0/1 theme columns coded from video titles and descriptions.

## Data Sources

| Dataset | File / Source | Description |
|---------|---------------|-------------|
| Cleaned YouTube narrative data | `data/Clean/ALL_11countries_2017_2020_aspect_sentiment_narrative_coded.csv` | YouTube video metadata with title/description-based 0/1 narrative theme columns for 11 European countries. The raw video collection covers 2017-2020, while the notebook filters to 2018-2020 for the main analysis. |
| YouTube narrative codebook | `data/Clean/narrative_coding_codebook.csv` | Codebook for the narrative themes, including each theme's description, video counts, and most frequent matched expressions used in the text-size figures. |
| Cleaned plant-based sales data | `data/Clean/plant_based_food_sales_data.csv` | Plant-based food sales data by country, year, and product group. Key variables include `Country`, `Year`, `Product Group`, `Value EUR`, and `Volume kg/l`. |
| YouTube Data API | https://developers.google.com/youtube/v3 | Used earlier in the project to collect YouTube video metadata such as title, description, upload date, channel information, views, likes, and comments. The current notebook does not re-run the API extraction. |
| European plant-based foods sales data | https://zenodo.org/records/6411841 | Source for the plant-based food panel sales dataset. |

## YouTube Narrative Variables

The cleaned YouTube dataset contains one binary column for each narrative theme. A value of `1` means the title or description matched at least one expression for that theme.

| Narrative Theme | Interpretation |
|-----------------|----------------|
| `health` | Health, nutrition, nutrients, dietary restrictions, or disease prevention framing |
| `environment` | Environment, climate, sustainability, zero-waste, emissions, or eco-friendly framing |
| `taste` | Taste, flavor, deliciousness, cravings, taste tests, or negative taste reactions |
| `animal_ethics` | Animal welfare, animal rights, cruelty-free, slaughter, compassion, or ethics framing |
| `recipe_cooking` | Recipes, cooking, baking, ingredients, meals, dishes, or homemade preparation |
| `convenience_easy` | Ease, speed, beginner-friendliness, meal prep, one-pot/no-bake, or time-saving framing |
| `fitness_weight` | Fitness, sport, bodybuilding, muscle, calories/macros, transformation, or weight loss |
| `challenge_transition` | Trying veganism, going vegan, beginner guides, first-time videos, or vegan challenges |
| `product_review_alternatives` | Plant-based substitutes, dairy/meat alternatives, brand/product reviews, hauls, or taste tests |
| `budget_shopping` | Budget, price, grocery shopping, supermarkets, discount retailers, or affordability |
| `restaurant_travel` | Restaurants, cafes, vegan options while traveling, city guides, vlogs, or eating out |

## Geographic and Time Scope

The project covers 11 European countries:

- Austria
- Belgium
- Denmark
- France
- Germany
- Italy
- Netherlands
- Poland
- Romania
- Spain
- United Kingdom

The YouTube file contains videos from 2017-2020. The main notebook filters to 2018-2020 because the cleaned sales dataset covers 2018-2020.

For the sales relationship analysis, YouTube narrative measures are aggregated to the country-year level and merged with country-year total sales value. If a country-year has no usable non-zero `Value EUR`, it is listed in the notebook as unmatched and excluded from the sales relationship and OLS analysis.

## Key Files

```text
data/
  Clean/
    ALL_11countries_2017_2020_aspect_sentiment_narrative_coded.csv
    narrative_coding_codebook.csv
    plant_based_food_sales_data.csv
  Raw/
    youtube_plant_based_*.csv
    youtube_plant_based_*.json

notebooks/
  0627 Group I.ipynb
  0627 Group I.py
  0624 Group I.py

src/
  analysis_helpers.py
  youtube_plant_based_uk_extractor.py
```

## How to Run

1. Open `notebooks/0627 Group I.ipynb`.
2. Run the notebook from top to bottom.
3. The notebook reads cleaned datasets from `data/Clean`.
4. The YouTube API extraction is not required for the current analysis.

Recommended Python packages:

- pandas
- numpy
- matplotlib
- statsmodels
- requests

## Notes

- The YouTube narrative workflow uses the cleaned aspect-sentiment CSV directly.
- Taste is analyzed as one combined `taste` narrative based on `taste_positive` and `taste_negative`.
- The notebook emphasizes descriptive patterns and exploratory OLS regressions at the country-year level. Regression coefficients should be interpreted as associations, not causal effects.

## References

- YouTube Data API Documentation: https://developers.google.com/youtube/v3
- European plant-based foods sales data: https://zenodo.org/records/6411841
