# Video-Analysis--DataScience-Group

# AI Agents & Video Platform Behaviors (YouTube case study)

## 👥 Team Members

- Name (@github_handle) - Role/Responsibility
- Name (@github_handle) - Role/Responsibility
- HUANG Shuo (@huangshuojason) - Role/Responsibility


## ❓ Research Question & 🎯 Hypothesis

> State your central research question clearly and concisely

- Hypothesis 1
- Hypothesis 2
- Hypothesis 3

## 📁 Data Sources

| Source | Description | URL |
|--------|-------------|-----|
| World Bank | Brief description | [World Bank Open Data](https://data.worldbank.org/) |
| IMF | Brief description | [IMF Data Portal](https://www.imf.org/en/Data) |

### Data Sources Details

#### D.1 World Bank  
**Variables:** e.g., NY.GDP.MKTP.CD, SE.PRM.CMPT.ZS

**Granularity:** e.g., Annual data by Country

#### D.2 IMF  
**Variables:** e.g., Consumer Price Index, Interest Rates

**Granularity:** e.g., Quarterly data by Region

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
| CONG Tianxiang| Description of contributions |
| Euimin Keum   | Description of contributions |
| HUANG Shuo| Description of contributions |

## 🔗 References
- Link to methodology references
