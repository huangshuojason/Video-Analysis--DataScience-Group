# %% [markdown]
# Project Summary and Research Questions
# 
# This project examines how plant-based food markets and YouTube narratives developed across selected European countries from 2018 to 2020. It combines country-year-product sales data with YouTube video text data to compare market size, product-category composition, narrative prevalence, and the relationship between online narratives and plant-based food sales. Since sales value and volume are strongly correlated, the later analysis focuses mainly on Value EUR as the key market indicator.
# 
# - Main RQ: How are YouTube narratives about plant-based foods associated with plant-based food sales patterns across selected European countries from 2018 to 2020?
# - Sub-RQ1: How do plant-based food sales values vary across countries, years, and product groups?
# - Sub-RQ2: How does the product-category composition of plant-based food sales differ across countries and over time?
# - Sub-RQ3: Which YouTube narratives are most frequently mentioned across countries and years?
# - Sub-RQ4: To what extent are YouTube narrative mentions associated with total plant-based food sales value at the country-year level?

# %%
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis_helpers import (
    add_log_total_value,
    add_narrative_features,
    add_sales_log_columns,
    combine_youtube_csvs,
    create_narrative_share_table,
    create_share_table,
    create_wide_value_table,
    fit_ols,
    merge_narrative_sales,
    narrative_mention_columns,
    prepare_sales_data,
    prepare_youtube_text,
    print_basic_info,
    print_sales_overview,
    summarize_narratives,
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
# The following analysis is based on video information related to plant-based foods collected using the YouTube API. Since the YouTube API requires a personal API key, and repeated execution may trigger quota limits, the API extraction code is not executed in the following notebook. Instead, the extracted CSV dataset is provided in the data/Raw directory. The original Python source code used for API extraction can be found in the src directory.

# %%
folder = Path("../data/Raw")
df_all_youtube, csv_files = combine_youtube_csvs(folder)
print("Number of CSV files found:", len(csv_files))
for file in csv_files:
    print(file.name)
output_path = folder / "youtube_plant_based_all_countries.csv"
df_all_youtube.to_csv(output_path, index=False, encoding="utf-8-sig")
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
df_youtube = prepare_youtube_text(df_youtube)

# %%
narrative_keywords = {
    "health": [
        # English
        "health", "healthy", "healthier", "nutrition", "nutritious", "nutrient",
        "protein", "high protein", "rich in protein", "diet", "dietary",
        "wellbeing", "well-being", "wellness", "balanced diet",
        "low fat", "low calorie", "low sugar", "sugar free", "cholesterol free",
        "natural", "organic", "clean label", "fiber", "fibre", "vitamin", "mineral",

        # German - Austria
        "gesund", "gesunde", "gesunder", "gesundheit", "gesundheitsbewusst",
        "ernährung", "nährstoff", "nährstoffe", "nährwert", "protein",
        "eiweiß", "eiweiss", "eiweißreich", "eiweissreich",
        "diät", "ausgewogene ernährung", "wohlbefinden",
        "fettarm", "kalorienarm", "zuckerfrei", "ballaststoffe",
        "natürlich", "bio", "biologisch", "vitamine", "mineralstoffe",

        # French - France / Belgium
        "santé", "sante", "sain", "saine", "plus sain", "nutrition",
        "nutritif", "nutritive", "nutriments", "protéine", "proteine",
        "protéines", "proteines", "riche en protéines", "riche en proteines",
        "régime", "regime", "bien-être", "bien etre", "équilibré", "equilibre",
        "faible en gras", "faible en calories", "sans sucre",
        "sans cholestérol", "sans cholesterol", "naturel", "naturelle",
        "bio", "biologique", "fibres", "vitamines", "minéraux", "mineraux",

        # Dutch - Netherlands / Belgium
        "gezond", "gezonde", "gezonder", "gezondheid", "voeding",
        "voedzaam", "voedingswaarde", "voedingsstoffen",
        "proteïne", "proteine", "eiwit", "eiwitten", "eiwitrijk",
        "dieet", "welzijn", "gebalanceerd dieet",
        "vetarm", "caloriearm", "suikervrij", "cholesterolvrij",
        "natuurlijk", "biologisch", "vezels", "vitaminen", "mineralen",

        # Danish - Denmark
        "sund", "sunde", "sundere", "sundhed", "ernæring",
        "næringsrig", "næringsstoffer", "næringsværdi",
        "protein", "proteiner", "proteinrig",
        "kost", "diæt", "velvære", "balanceret kost",
        "fedtfattig", "kaloriefattig", "sukkerfri", "kolesterolfri",
        "naturlig", "økologisk", "okologisk", "fibre", "vitaminer", "mineraler",

        # Italian - Italy
        "salute", "sano", "sana", "più sano", "piu sano",
        "nutrizione", "nutriente", "nutrienti", "valore nutrizionale",
        "proteina", "proteine", "ricco di proteine",
        "dieta", "benessere", "dieta equilibrata",
        "pochi grassi", "basso contenuto di grassi",
        "poche calorie", "senza zucchero", "senza colesterolo",
        "naturale", "biologico", "bio", "fibre", "vitamine", "minerali",

        # Romanian - Romania
        "sănătate", "sanatate", "sănătos", "sanatos", "sănătoasă", "sanatoasa",
        "mai sănătos", "mai sanatos", "nutriție", "nutritie",
        "nutritiv", "nutrienți", "nutrienti", "valoare nutritivă", "valoare nutritiva",
        "proteină", "proteina", "proteine", "bogat în proteine", "bogat in proteine",
        "dietă", "dieta", "bunăstare", "bunastare", "dietă echilibrată",
        "fără zahăr", "fara zahar", "sărac în grăsimi", "sarac in grasimi",
        "natural", "organic", "bio", "fibre", "vitamine", "minerale",

        # Spanish - Spain
        "salud", "saludable", "más saludable", "mas saludable",
        "nutrición", "nutricion", "nutritivo", "nutritiva", "nutrientes",
        "valor nutricional", "proteína", "proteina", "proteínas", "proteinas",
        "rico en proteínas", "rico en proteinas",
        "dieta", "bienestar", "dieta equilibrada",
        "bajo en grasa", "bajo en calorías", "bajo en calorias",
        "sin azúcar", "sin azucar", "sin colesterol",
        "natural", "orgánico", "organico", "ecológico", "ecologico",
        "fibra", "vitaminas", "minerales"
    ],

    "environment": [
        # English
        "environment", "environmental", "eco", "ecological", "green",
        "sustainable", "sustainability", "climate", "climate change",
        "carbon", "carbon footprint", "co2", "co2e", "emissions",
        "greenhouse gas", "greenhouse gases", "ghg",
        "biodiversity", "water use", "land use", "resource use",
        "planet", "planet-friendly", "earth", "low impact",
        "low carbon", "zero waste", "renewable", "recyclable",

        # German - Austria
        "umwelt", "umweltfreundlich", "umweltbewusst", "ökologisch", "okologisch",
        "öko", "oko", "grün", "gruen", "nachhaltig", "nachhaltigkeit",
        "klima", "klimawandel", "klimaschutz",
        "kohlenstoff", "co2", "co2-fußabdruck", "co2 fussabdruck",
        "emissionen", "treibhausgas", "treibhausgase",
        "biodiversität", "biodiversitaet", "artenvielfalt",
        "wasserverbrauch", "landnutzung", "ressourcenverbrauch",
        "planet", "erde", "klimafreundlich", "kohlenstoffarm",

        # French - France / Belgium
        "environnement", "environnemental", "écologique", "ecologique",
        "écolo", "ecolo", "vert", "verte", "durable", "durabilité", "durabilite",
        "climat", "changement climatique", "réchauffement climatique",
        "rechauffement climatique", "carbone", "empreinte carbone",
        "co2", "émissions", "emissions", "gaz à effet de serre", "gaz a effet de serre",
        "biodiversité", "biodiversite", "utilisation de l'eau", "consommation d'eau",
        "utilisation des terres", "usage des terres", "ressources",
        "planète", "planete", "terre", "faible impact", "bas carbone",

        # Dutch - Netherlands / Belgium
        "milieu", "milieuvriendelijk", "milieubewust", "ecologisch",
        "eco", "groen", "duurzaam", "duurzaamheid",
        "klimaat", "klimaatverandering", "klimaatvriendelijk",
        "koolstof", "co2", "co2-voetafdruk", "carbon footprint",
        "uitstoot", "emissies", "broeikasgas", "broeikasgassen",
        "biodiversiteit", "waterverbruik", "landgebruik",
        "grondstoffengebruik", "hulpbronnen", "planeet", "aarde",
        "lage impact", "koolstofarm",

        # Danish - Denmark
        "miljø", "miljo", "miljøvenlig", "miljovenlig", "miljøbevidst",
        "økologisk", "okologisk", "grøn", "gron",
        "bæredygtig", "baeredygtig", "bæredygtighed", "baeredygtighed",
        "klima", "klimaforandringer", "klimavenlig",
        "kulstof", "co2", "co2-aftryk", "carbon footprint",
        "udledning", "emissioner", "drivhusgas", "drivhusgasser",
        "biodiversitet", "vandforbrug", "arealanvendelse", "jordforbrug",
        "ressourceforbrug", "planet", "jorden", "lav miljøpåvirkning",
        "lav miljopavirkning", "lavt klimaaftryk",

        # Italian - Italy
        "ambiente", "ambientale", "ecologico", "eco", "verde",
        "sostenibile", "sostenibilità", "sostenibilita",
        "clima", "cambiamento climatico", "riscaldamento globale",
        "carbonio", "impronta di carbonio", "co2",
        "emissioni", "gas serra", "gas a effetto serra",
        "biodiversità", "biodiversita", "uso dell'acqua", "consumo d'acqua",
        "uso del suolo", "uso della terra", "risorse",
        "pianeta", "terra", "basso impatto", "basse emissioni",

        # Romanian - Romania
        "mediu", "de mediu", "ecologic", "eco", "verde",
        "sustenabil", "sustenabilitate", "durabil", "durabilitate",
        "climă", "clima", "schimbări climatice", "schimbari climatice",
        "încălzire globală", "incalzire globala",
        "carbon", "amprentă de carbon", "amprenta de carbon", "co2",
        "emisii", "gaze cu efect de seră", "gaze cu efect de sera",
        "biodiversitate", "consum de apă", "consum de apa",
        "utilizarea terenurilor", "folosirea terenurilor",
        "resurse", "planetă", "planeta", "pământ", "pamant",
        "impact redus", "emisii reduse",

        # Spanish - Spain
        "medio ambiente", "ambiental", "ecológico", "ecologico",
        "eco", "verde", "sostenible", "sostenibilidad",
        "clima", "cambio climático", "cambio climatico", "calentamiento global",
        "carbono", "huella de carbono", "co2",
        "emisiones", "gases de efecto invernadero",
        "biodiversidad", "uso del agua", "consumo de agua",
        "uso de la tierra", "uso del suelo", "recursos",
        "planeta", "tierra", "bajo impacto", "bajas emisiones",
        "bajo en carbono"
    ],

    "animal_welfare": [
        # English
        "animal welfare", "animal", "animals", "cruelty", "cruelty free",
        "animal-friendly", "animal friendly", "ethical", "ethically made",
        "no slaughter", "slaughter-free", "slaughter", "cows", "pigs", "chickens",
        "livestock", "factory farming", "animal suffering", "animal rights",

        # German - Austria
        "tierwohl", "tierschutz", "tierfreundlich", "tierleid",
        "ohne tierleid", "grausamkeitsfrei", "tiere", "tier",
        "ethisch", "schlachtung", "ohne schlachtung", "schlachtfrei",
        "kühe", "kuehe", "schweine", "hühner", "huehner",
        "nutztier", "nutztiere", "massentierhaltung", "tierrechte",

        # French - France / Belgium
        "bien-être animal", "bien etre animal", "protection animale",
        "animaux", "animal", "sans cruauté", "sans cruaute",
        "cruauté", "cruaute", "respect des animaux",
        "éthique", "ethique", "abattage", "sans abattage",
        "vaches", "porcs", "cochons", "poulets",
        "élevage industriel", "elevage industriel", "souffrance animale",
        "droits des animaux",

        # Dutch - Netherlands / Belgium
        "dierenwelzijn", "dierenbescherming", "diervriendelijk",
        "dierenleed", "zonder dierenleed", "wreedheidvrij",
        "dieren", "dier", "ethisch", "slacht", "slachten",
        "zonder slacht", "slachtvrij", "koeien", "varkens", "kippen",
        "vee", "veehouderij", "bio-industrie", "dierenrechten",

        # Danish - Denmark
        "dyrevelfærd", "dyrevelfaerd", "dyrebeskyttelse",
        "dyrevenlig", "dyrelidelse", "uden dyrelidelse",
        "grusomhedsfri", "dyr", "etisk", "slagtning",
        "uden slagtning", "slagtefri", "køer", "koer", "grise", "svin", "kyllinger",
        "husdyr", "fabrikslandbrug", "industrielt landbrug", "dyrerettigheder",

        # Italian - Italy
        "benessere animale", "protezione degli animali",
        "animali", "animale", "senza crudeltà", "senza crudelta",
        "crudele", "crudeltà", "crudelta", "amico degli animali",
        "etico", "etica", "macellazione", "senza macellazione",
        "mucche", "vacche", "maiali", "polli",
        "allevamento intensivo", "sofferenza animale", "diritti degli animali",

        # Romanian - Romania
        "bunăstarea animalelor", "bunastarea animalelor",
        "protecția animalelor", "protectia animalelor",
        "animale", "animal", "fără cruzime", "fara cruzime",
        "cruzime", "prietenos cu animalele", "etic", "etică", "etica",
        "sacrificare", "fără sacrificare", "fara sacrificare",
        "vaci", "porci", "pui", "găini", "gaini",
        "creștere intensivă", "crestere intensiva",
        "suferința animalelor", "suferinta animalelor",
        "drepturile animalelor",

        # Spanish - Spain
        "bienestar animal", "protección animal", "proteccion animal",
        "animales", "animal", "sin crueldad", "libre de crueldad",
        "crueldad", "amigable con los animales",
        "ético", "etico", "ética", "etica",
        "sacrificio", "sin sacrificio", "matadero",
        "vacas", "cerdos", "pollos", "ganado",
        "ganadería industrial", "ganaderia industrial",
        "sufrimiento animal", "derechos de los animales"
    ],

    "hedonism": [
        # English
        "taste", "tasty", "delicious", "flavour", "flavor",
        "flavourful", "flavorful", "yummy", "enjoy", "enjoyment",
        "pleasure", "pleasurable", "comfort food", "indulgent",
        "craving", "satisfying", "juicy", "crispy", "tender",
        "mouthfeel", "texture", "aroma", "smell", "savory", "savoury",
        "umami", "gourmet", "treat",

        # German - Austria
        "geschmack", "lecker", "köstlich", "koestlich",
        "schmackhaft", "genuss", "genießen", "geniessen",
        "vergnügen", "vergnuegen", "wohlfühlessen", "wohlfuehlessen",
        "komfortessen", "verwöhnend", "verwoehnend",
        "heißhunger", "heisshunger", "befriedigend",
        "saftig", "knusprig", "zart", "textur", "mundgefühl", "mundgefuehl",
        "aroma", "geruch", "herzhaft", "umami", "feinschmecker",

        # French - France / Belgium
        "goût", "gout", "savoureux", "savoureuse", "délicieux", "delicieux",
        "délicieuse", "delicieuse", "plaisir", "se faire plaisir",
        "gourmand", "gourmande", "gourmandise", "réconfortant", "reconfortant",
        "nourriture réconfortante", "nourriture reconfortante",
        "envie", "satisfaisant", "satisfaisante",
        "juteux", "juteuse", "croustillant", "croustillante",
        "tendre", "texture", "arôme", "arome", "odeur",
        "umami", "gastronomique",

        # Dutch - Netherlands / Belgium
        "smaak", "lekker", "heerlijk", "smakelijk",
        "genieten", "genot", "plezier", "comfortfood",
        "troosteten", "verwennerij", "verleidelijk",
        "trek", "craving", "bevredigend", "sappig",
        "knapperig", "mals", "textuur", "mondgevoel",
        "aroma", "geur", "hartig", "umami", "gourmet",

        # Danish - Denmark
        "smag", "velsmagende", "lækker", "laekker",
        "lækkert", "laekkert", "nyde", "nyder", "nydelse",
        "fornøjelse", "fornojelse", "komfortmad",
        "trøstemad", "troestemad", "forkælelse", "forkaelelse",
        "craving", "lyst", "tilfredsstillende",
        "saftig", "sprød", "sprod", "mør", "moer",
        "tekstur", "mundfølelse", "mundfolelse",
        "aroma", "duft", "umami", "gourmet",

        # Italian - Italy
        "gusto", "sapore", "gustoso", "gustosa",
        "delizioso", "deliziosa", "buono", "buona",
        "piacere", "godere", "godimento",
        "comfort food", "cibo di conforto",
        "indulgente", "sfizioso", "sfiziosa",
        "voglia", "soddisfacente", "succoso", "succosa",
        "croccante", "tenero", "tenera",
        "consistenza", "texture", "sensazione in bocca",
        "aroma", "profumo", "saporito", "umami", "gourmet",

        # Romanian - Romania
        "gust", "gustos", "gustoasă", "gustoasa",
        "delicios", "delicioasă", "delicioasa",
        "plăcere", "placere", "a savura", "savurare",
        "mâncare de confort", "mancare de confort",
        "răsfăț", "rasfat", "poftă", "pofta",
        "satisfăcător", "satisfacator",
        "suculent", "crocant", "fraged",
        "textură", "textura", "senzație în gură", "senzatie in gura",
        "aromă", "aroma", "miros", "savuros", "umami", "gourmet",

        # Spanish - Spain
        "sabor", "sabroso", "sabrosa", "rico", "rica",
        "delicioso", "deliciosa", "placer", "disfrutar",
        "disfrute", "comida reconfortante",
        "capricho", "indulgente", "antojo",
        "satisfactorio", "satisfactoria",
        "jugoso", "jugosa", "crujiente", "tierno", "tierna",
        "textura", "sensación en boca", "sensacion en boca",
        "aroma", "olor", "umami", "gourmet"
    ]
}

# %%
df_youtube = add_narrative_features(df_youtube, narrative_keywords)
df_youtube.head()

# %%
narrative_by_year = summarize_narratives(df_youtube, "Year", narrative_keywords).set_index("Year")
narrative_by_country_year = summarize_narratives(df_youtube, ["country", "Year"], narrative_keywords)
narrative_by_year

# %%
narrative_by_year.plot(kind='line',marker='o',figsize=(9, 5))
plt.title('Number of Videos Mentioning Each Narrative by Year')
plt.xlabel('Year')
plt.ylabel('Number of Videos')
plt.xticks(narrative_by_year.index)
plt.legend(title='Narrative', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# The figure shows that the number of YouTube videos mentioning plant-based food narratives increased across all four themes from 2018 to 2020. Environmental and hedonistic narratives showed the strongest growth and became the most frequently mentioned themes by 2020, while health-related narratives also increased steadily. In contrast, animal welfare remained the least mentioned narrative throughout the period, despite some growth. This suggests that YouTube discussions around plant-based foods became more active over time, with environmental benefits and enjoyment-related appeals gaining particular visibility.

# %%
narrative_cols = narrative_mention_columns(narrative_keywords)
df_plot, countries, years = create_narrative_share_table(narrative_by_country_year, narrative_cols)
fig, axes = plt.subplots(1, len(years), figsize=(18, 5), sharey=True)
for i, year in enumerate(years):
    ax = axes[i]
    data = (
        df_plot[df_plot['Year'] == year]
        .set_index('country')
        .reindex(countries)[narrative_cols]
    )
    data.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        legend=False)
    if i > 0:
        prev_data = (
            df_plot[df_plot['Year'] == years[i - 1]]
            .set_index('country')
            .reindex(countries)[narrative_cols]
        )
        change = data - prev_data
        for country_i, country in enumerate(data.index):
            bottom = 0
            for narrative in narrative_cols:
                value = data.loc[country, narrative]
                if value > 8:
                    arrow = '↑' if change.loc[country, narrative] > 0 else '↓'
                    ax.text(country_i, bottom + value / 2, arrow, ha='center', va='center')
                bottom += value
    ax.set_title(year)
    ax.set_xlabel('Country')
    ax.set_ylabel('Share (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 100)
plt.legend(title='Narrative', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle('Narrative Share by Country and Year')
plt.tight_layout()
plt.show()

# %% [markdown]
# The figure shows the narrative share of YouTube videos across countries from 2018 to 2020. Overall, health and environmental narratives account for a large proportion in most countries, while animal welfare takes a relatively smaller share. Hedonism-related narratives also remain important across the three years, suggesting that taste, enjoyment, and daily appeal are frequently used in plant-based food discussions. The arrows show that narrative shares fluctuate by country and year, but there is no single narrative that consistently dominates all markets.
# These results suggest that plant-based food communication on YouTube is multi-dimensional. Health and environmental messages remain important, but hedonism-related narratives are also widely present. This implies that communication strategies should not rely only on sustainability or ethical arguments. Combining health, environmental benefits, and enjoyable eating experiences may be more effective in attracting mainstream consumers.

# %%
narrative_sales_country_year = merge_narrative_sales(narrative_by_country_year, df_wide_value)
narrative_sales_country_year.head()

# %%
model_raw = fit_ols(narrative_sales_country_year, narrative_cols, 'Total Value EUR')
narrative_sales_country_year = add_log_total_value(narrative_sales_country_year)
model_log = fit_ols(narrative_sales_country_year, narrative_cols, 'log_total_value')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(model_raw.fittedvalues, model_raw.resid)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Residuals: Raw Model')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[1].scatter(model_log.fittedvalues, model_log.resid)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Residuals: Log Model')
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Residuals')
plt.tight_layout()
plt.show()

# %% [markdown]
# The residual plots compare the raw and log-transformed models. The raw model shows large residuals because total sales values vary greatly across countries. After applying the log transformation, the residuals are on a smaller and more stable scale, reducing the influence of very large market sizes. Therefore, the log model is more appropriate for the regression analysis.

# %%

model_log = fit_ols(narrative_sales_country_year, narrative_cols, 'log_total_value')
print(model_log.summary())

# %%
import seaborn as sns
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for i, narrative in enumerate(narrative_cols):
    sns.regplot(
        data=narrative_sales_country_year,
        x=narrative,
        y='log_total_value',
        ax=axes[i])
    axes[i].set_title(narrative)
    axes[i].set_xlabel('Number of Videos')
    axes[i].set_ylabel('Log Total Value EUR')
plt.tight_layout()
plt.show()

# %% [markdown]
# The scatter plots suggest that all four YouTube narratives show a broadly positive relationship with log total plant-based food sales when examined separately. In particular, countries and years with more videos mentioning health, environment, animal welfare, or hedonism-related narratives tend to have higher sales values. However, the OLS regression provides a more nuanced result after controlling for the four narratives simultaneously. The model explains a moderate share of variation in plant-based food sales, with an R-squared value of 0.555, and the overall model is statistically significant. Among the four narratives, health-related mentions show a positive and statistically significant association with log total sales. This suggests that health narratives may be more closely linked to plant-based food market performance during the study period. In contrast, environmental mentions show a negative and statistically significant coefficient after controlling for the other narratives. This does not necessarily mean that environmental narratives reduce sales; rather, it may indicate that environmental narratives overlap with other narratives or are more visible in markets where sales are not growing as strongly. Animal welfare and hedonism-related mentions have positive coefficients, but they are not statistically significant in the regression model. These results imply that YouTube narratives are associated with plant-based food sales, but their effects are not uniform across themes. Health-related communication appears to be the most consistently linked with higher sales, suggesting that health may be an important consumer-facing message in plant-based food markets. At the same time, the difference between the scatter plots and regression results suggests that simple visual correlations may overstate the role of some narratives. For business and communication strategy, this means that narrative framing should not only focus on increasing the number of videos, but also consider which themes are more strongly associated with market outcomes. For policy or sustainability communication, environmental narratives may still be important, but they may need to be combined with more directly consumer-relevant messages, such as health, taste, or daily-use benefits, to translate attention into market demand.

# %%
narrative_corr = narrative_sales_country_year[narrative_cols].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(narrative_corr,annot=True,cmap='coolwarm',vmin=-1,vmax=1)
plt.title('Correlation Heatmap of Narrative Variables')
plt.tight_layout()
plt.show()

# %% [markdown]
# The heatmap shows positive correlations among all narrative variables. The strongest relationship is between health_mentioned and environment_mentioned at 0.76, followed by environment_mentioned with animal_welfare_mentioned at 0.69 and hedonism_mentioned at 0.68.
# 
# The narratives often appear together in the same country-year context, suggesting potential multicollinearity in the regression model. This may explain why some coefficients, such as environment_mentioned, differ from the simple scatterplot trend after controlling for other narratives.


