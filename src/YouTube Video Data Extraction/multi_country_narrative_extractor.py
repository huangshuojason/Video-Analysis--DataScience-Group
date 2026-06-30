"""
Multi-country YouTube Narrative Extractor
==========================================

Goal
----
For every country that appears in `plant_based_food_sales_data.csv`, pull the
top-N most-viewed YouTube videos about plant-based / vegan food, extract their
narrative signals (health / environment / animal-welfare / taste / price /
convenience / novelty / cultural-fit), and produce country-level summaries
that can be joined against the sales data to study the relationship between
public narrative and vegan food consumption.

Pipeline
--------
1. Read the sales CSV → unique country list.
2. For each country, run several multilingual YouTube searches
   (plant-based / vegan / meat-alt / milk-alt) in that country's language(s)
   with regionCode set to the country code.
3. Fetch full video metadata + statistics, rank by viewCount,
   keep the top --top-n per country (default 100).
4. Run narrative tagging on title + description + (optional) transcript.
5. Export:
       videos_all_countries.csv     – one row per video, all countries
       per_country/<country>.csv    – one file per country (top-N)
       narrative_by_country.csv     – tag totals per country
       narrative_share_by_country.csv – tags normalised to share (0–1)
       country_vs_sales.csv         – narrative shares joined to sales totals
       report.md                    – human-readable summary

Setup
-----
Same deps as youtube_narrative_extractor.py:
    pip install google-api-python-client youtube-transcript-api vaderSentiment \
                pandas tqdm python-dateutil

Set your key — either edit YOUTUBE_API_KEY at the top of this file, or:
    export YOUTUBE_API_KEY="AIza..."

Run
---
    python multi_country_narrative_extractor.py
    python multi_country_narrative_extractor.py --top-n 50 --no-transcripts
    python multi_country_narrative_extractor.py --countries Germany France
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =============================================================================
# 0. PASTE YOUR API KEY HERE (between the quotes), or use the env var.
# =============================================================================
YOUTUBE_API_KEY = ""   # e.g. "AIzaSyB1234567890abcdefghijklmnop"


# =============================================================================
# 1. COUNTRY → (regionCode, languages, search queries)
# Multilingual queries widen recall; English queries help for cross-border videos.
# =============================================================================
COUNTRY_CONFIG: dict[str, dict] = {
    "Austria":        {"region": "AT", "lang": "de",
                       "queries": ["pflanzlich vegan Lebensmittel",
                                   "vegane Ernährung Österreich",
                                   "Fleischersatz Hafermilch",
                                   "plant based food Austria"]},
    "Belgium":        {"region": "BE", "lang": "nl",
                       "queries": ["plantaardig voedsel veganistisch",
                                   "vegan eten België",
                                   "vleesvervangers haver melk",
                                   "alimentation végétale Belgique",
                                   "plant based food Belgium"]},
    "Denmark":        {"region": "DK", "lang": "da",
                       "queries": ["plantebaseret mad veganer",
                                   "vegansk Danmark",
                                   "kødalternativ havremælk",
                                   "plant based food Denmark"]},
    "France":         {"region": "FR", "lang": "fr",
                       "queries": ["alimentation végétale vegan",
                                   "régime végétalien France",
                                   "substitut de viande lait d'avoine",
                                   "plant based food France"]},
    "Germany":        {"region": "DE", "lang": "de",
                       "queries": ["pflanzliche Ernährung vegan",
                                   "vegane Lebensmittel Deutschland",
                                   "Fleischersatz Hafermilch",
                                   "plant based food Germany"]},
    "Italy":          {"region": "IT", "lang": "it",
                       "queries": ["alimentazione vegetale vegana",
                                   "dieta vegana Italia",
                                   "sostituto della carne latte d'avena",
                                   "plant based food Italy"]},
    "Netherlands":    {"region": "NL", "lang": "nl",
                       "queries": ["plantaardig eten veganistisch",
                                   "vegan Nederland",
                                   "vleesvervanger havermelk",
                                   "plant based food Netherlands"]},
    "Poland":         {"region": "PL", "lang": "pl",
                       "queries": ["roślinne jedzenie wegańskie",
                                   "wegańska dieta Polska",
                                   "zamiennik mięsa mleko owsiane",
                                   "plant based food Poland"]},
    "Romania":        {"region": "RO", "lang": "ro",
                       "queries": ["mâncare vegană pe bază de plante",
                                   "dietă vegană România",
                                   "înlocuitor de carne lapte de ovăz",
                                   "plant based food Romania"]},
    "Spain":          {"region": "ES", "lang": "es",
                       "queries": ["alimentación vegetal vegana",
                                   "dieta vegana España",
                                   "sustituto de carne leche de avena",
                                   "plant based food Spain"]},
    "United Kingdom": {"region": "GB", "lang": "en",
                       "queries": ["plant based food UK",
                                   "vegan diet United Kingdom",
                                   "meat alternative oat milk",
                                   "vegan burger England"]},
}


# =============================================================================
# 2. NARRATIVE TAG DICTIONARY  (extend per study)
# Each tag has keywords in many languages.
# =============================================================================
NARRATIVE_TAGS: dict[str, list[str]] = {
    "health": [
        # EN/DE/FR/IT/ES/NL/PL/RO/DA
        "healthy", "health", "nutrition", "protein", "low fat", "cholesterol",
        "gesund", "Gesundheit", "Eiweiß", "Ernährung",
        "santé", "sain", "nutrition", "protéine",
        "salute", "salutare", "nutrizione", "proteina",
        "salud", "saludable", "nutrición", "proteína",
        "gezond", "gezondheid", "voeding", "eiwit",
        "zdrowy", "zdrowie", "białko", "odżywianie",
        "sănătate", "sănătos", "proteine", "nutriție",
        "sundhed", "sund", "protein", "ernæring",
    ],
    "environment": [
        "environment", "climate", "carbon", "sustainable", "co2", "planet",
        "Umwelt", "Klima", "nachhaltig",
        "environnement", "climat", "durable",
        "ambiente", "clima", "sostenibile",
        "medio ambiente", "clima", "sostenible",
        "milieu", "klimaat", "duurzaam",
        "środowisko", "klimat", "zrównoważony",
        "mediu", "climă", "durabil",
        "miljø", "klima", "bæredygtig",
    ],
    "animal_welfare": [
        "animal welfare", "cruelty", "ethical", "animal rights",
        "Tierschutz", "Tierwohl", "ethisch",
        "bien-être animal", "cruauté", "éthique",
        "benessere animale", "crudeltà", "etico",
        "bienestar animal", "crueldad", "ético",
        "dierenwelzijn", "wreedheid", "ethisch",
        "dobrostan zwierząt", "okrucieństwo", "etyczny",
        "bunăstarea animalelor", "etic",
        "dyrevelfærd", "etisk",
    ],
    "taste": [
        "taste", "tasty", "flavor", "delicious", "texture",
        "Geschmack", "lecker", "schmeckt",
        "goût", "savoureux", "délicieux",
        "sapore", "gustoso", "delizioso",
        "sabor", "sabroso", "delicioso",
        "smaak", "lekker", "heerlijk",
        "smak", "smaczny", "pyszny",
        "gust", "delicios",
        "smag", "lækker",
    ],
    "price": [
        "price", "expensive", "cheap", "value", "affordable",
        "Preis", "teuer", "günstig",
        "prix", "cher", "abordable",
        "prezzo", "caro", "economico",
        "precio", "caro", "barato", "asequible",
        "prijs", "duur", "goedkoop",
        "cena", "drogi", "tani",
        "preț", "scump", "ieftin",
        "pris", "dyrt", "billigt",
    ],
    "convenience": [
        "convenient", "easy", "quick", "ready meal",
        "bequem", "einfach", "schnell",
        "pratique", "facile", "rapide",
        "comodo", "facile", "veloce",
        "cómodo", "fácil", "rápido",
        "gemakkelijk", "snel",
        "wygodny", "łatwy", "szybki",
        "convenabil", "ușor", "rapid",
        "nem", "hurtig",
    ],
    "novelty": [
        "new", "trend", "first time", "try", "review",
        "neu", "Trend", "probieren",
        "nouveau", "tendance", "essayer",
        "nuovo", "tendenza", "provare",
        "nuevo", "tendencia", "probar",
        "nieuw", "trend", "proberen",
        "nowy", "trend", "spróbować",
        "nou", "trend", "încercare",
        "ny", "trend", "smag",
    ],
    "cultural_fit": [
        "tradition", "culture", "local",
        "Tradition", "Kultur", "lokal",
        "tradition", "culture", "local",
        "tradizione", "cultura", "locale",
        "tradición", "cultura", "local",
        "traditie", "cultuur", "lokaal",
        "tradycja", "kultura", "lokalny",
        "tradiție", "cultură", "local",
        "tradition", "kultur", "lokal",
    ],
}


# =============================================================================
# 3. DATA MODEL
# =============================================================================
@dataclass
class VideoRow:
    country: str
    video_id: str
    url: str
    title: str
    description: str
    channel_title: str
    published_at: str
    duration_seconds: int
    view_count: int
    like_count: int | None
    comment_count: int | None
    default_language: str | None
    tags_str: str
    transcript_excerpt: str
    summary: str
    sentiment: float | None
    # one column per narrative tag → narr_health, narr_environment, ...


# =============================================================================
# 4. YOUTUBE WRAPPER
# =============================================================================
def _is_quota_error(e: HttpError) -> bool:
    # YouTube returns quota/rate exhaustion as HTTP 403 OR 429
    # (reason rateLimitExceeded / quotaExceeded / dailyLimitExceeded).
    s = str(e).lower()
    status = getattr(getattr(e, "resp", None), "status", None)
    return status in (403, 429) and (
        "quota" in s or "ratelimit" in s or "dailylimit" in s)


class YouTubeClient:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.idx = 0
        self._build()

    def _build(self):
        self.api = build("youtube", "v3", developerKey=self.keys[self.idx],
                         cache_discovery=False)

    def _exec(self, resource: str, **params):
        """Run resource.list(**params).execute(), rotating keys on quota.
        Raises HttpError(quota) only when ALL keys are exhausted."""
        while True:
            try:
                return getattr(self.api, resource)().list(**params).execute()
            except HttpError as e:
                if _is_quota_error(e) and self.idx < len(self.keys) - 1:
                    self.idx += 1
                    print(f"  [quota] key#{self.idx} exhausted -> rotating to "
                          f"key#{self.idx + 1}/{len(self.keys)}", file=sys.stderr)
                    self._build()
                    continue
                raise

    def search_ids(self, query: str, region: str, lang: str | None = None,
                   max_results: int = 50,
                   published_after: str | None = None,
                   published_before: str | None = None) -> list[str]:
        ids, page = [], None
        while len(ids) < max_results:
            try:
                params = dict(
                    q=query, part="id", type="video",
                    maxResults=min(50, max_results - len(ids)),
                    regionCode=region,
                    order="viewCount", pageToken=page,
                )
                if lang:
                    params["relevanceLanguage"] = lang
                if published_after:
                    params["publishedAfter"] = published_after
                if published_before:
                    params["publishedBefore"] = published_before
                resp = self._exec("search", **params)
            except HttpError as e:
                print(f"  [search] {query!r}: {e}", file=sys.stderr)
                return ids
            ids.extend(item["id"]["videoId"] for item in resp.get("items", []))
            page = resp.get("nextPageToken")
            if not page:
                break
        return ids

    def video_details(self, ids: list[str]) -> list[dict]:
        out = []
        for chunk in _chunks(ids, 50):
            resp = self._exec("videos", id=",".join(chunk),
                              part="snippet,statistics,contentDetails")
            out.extend(resp.get("items", []))
        return out


# =============================================================================
# 5. TRANSCRIPT + SIGNALS
# =============================================================================
_YT_API = YouTubeTranscriptApi()   # 1.x: instance methods .fetch() / .list()

def fetch_transcript(video_id: str, langs: tuple[str, ...]) -> str | None:
    try:
        try:
            fetched = _YT_API.fetch(video_id, languages=list(langs))
        except NoTranscriptFound:
            tl = _YT_API.list(video_id)
            picked = next(iter(tl))
            try:
                picked = picked.translate(langs[0])
            except Exception:
                pass
            fetched = picked.fetch()
        # FetchedTranscript is iterable; each snippet has a .text attribute
        return " ".join(s.text for s in fetched if getattr(s, "text", "")).strip()
    except Exception:
        return None


_SENT = re.compile(r"(?<=[。．\.!?！？])\s+")

def extractive_summary(text: str, n: int = 4) -> str:
    if not text:
        return ""
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if len(sents) <= n:
        return " ".join(sents)
    words = re.findall(r"\w+", text.lower())
    freq = Counter(w for w in words if len(w) > 2)
    scored = sorted(((sum(freq.get(w, 0) for w in re.findall(r"\w+", s.lower()))
                      / (len(s.split()) + 1), s) for s in sents),
                    key=lambda x: -x[0])
    keep = {s for _, s in scored[:n]}
    return " ".join(s for s in sents if s in keep)


_VADER = SentimentIntensityAnalyzer()

def sentiment(text: str) -> float | None:
    return _VADER.polarity_scores(text)["compound"] if text else None


def tag_narratives(text: str) -> dict[str, int]:
    if not text:
        return {k: 0 for k in NARRATIVE_TAGS}
    low = text.lower()
    return {tag: sum(low.count(k.lower()) for k in kws)
            for tag, kws in NARRATIVE_TAGS.items()}


# =============================================================================
# 5b. QUERY GENERATOR
# Goal: per country, broadly capture vegan / plant-based FOOD videos.
# A video qualifies if it matches ANY query (English generic + country's own
# name + native-language broad terms). The consumer-behaviour phrasings from
# HUANG Shuo's spec returned ~0 on YouTube, so we widen to general vegan-food
# terms while keeping the productive "supermarket" angle.
# {country} = English name, {local_name} = country's own-language name.
# =============================================================================
LOCAL_NAMES = {
    "Austria": "Österreich", "Belgium": "België", "Denmark": "Danmark",
    "France": "France", "Germany": "Deutschland", "Italy": "Italia",
    "Netherlands": "Nederland", "Poland": "Polska", "Romania": "România",
    "Spain": "España", "United Kingdom": "United Kingdom",
}

# English + local-name generic templates (high recall, on-topic)
BROAD_TEMPLATES = [
    "vegan food {country}",
    "plant based food {country}",
    "vegan {country} supermarket",
    "meat alternatives {country}",
    "vegan {local_name}",
    "plant based food {local_name}",
]

# Short / single-word native-language terms (HIGH recall — the long phrases in
# the old COUNTRY_CONFIG under-collected small countries, e.g. Denmark 71 vs 234).
BROAD_NATIVE: dict[str, list[str]] = {
    "Austria":        ["vegan", "pflanzlich", "vegane Ernährung",
                       "Fleischersatz", "Hafermilch"],
    "Belgium":        ["veganistisch", "plantaardig", "vegan eten",
                       "végétalien", "alimentation végétale"],
    "Denmark":        ["vegansk", "plantebaseret", "vegansk mad",
                       "kødalternativ", "havremælk"],
    "France":         ["végétalien", "végétal", "vegan",
                       "substitut de viande", "lait d'avoine"],
    "Germany":        ["vegan", "pflanzlich", "vegane Ernährung",
                       "Fleischersatz", "Hafermilch"],
    "Italy":          ["vegano", "vegetale", "cibo vegano",
                       "sostituto della carne", "latte d'avena"],
    "Netherlands":    ["veganistisch", "plantaardig", "vegan eten",
                       "vleesvervanger", "havermelk"],
    "Poland":         ["wegański", "roślinny", "wegańskie jedzenie",
                       "zamiennik mięsa", "mleko owsiane"],
    "Romania":        ["vegan", "vegetal", "mâncare vegană",
                       "înlocuitor de carne", "lapte de ovăz"],
    "Spain":          ["vegano", "vegetal", "comida vegana",
                       "sustituto de carne", "leche de avena"],
    "United Kingdom": ["vegan", "plant based", "vegan food",
                       "meat alternative", "oat milk"],
}


def build_queries(country: str) -> list[str]:
    """Generic English/local templates + broad native-language terms."""
    local = LOCAL_NAMES.get(country, country)
    qs = [t.format(country=country, local_name=local) for t in BROAD_TEMPLATES]
    qs += BROAD_NATIVE.get(country, [])
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q); out.append(q)
    return out


# =============================================================================
# 6. ORCHESTRATION  — search whole window once, bucket by year, top-N per year
# =============================================================================
def collect_country(country: str, cfg: dict, yt: YouTubeClient,
                    top_n: int, fetch_tr: bool,
                    years: list[int]) -> list[dict]:
    region = cfg["region"]
    queries = build_queries(country)
    y0, y1 = min(years), max(years)
    pa = f"{y0}-01-01T00:00:00Z"
    pb = f"{y1 + 1}-01-01T00:00:00Z"
    print(f"\n→ {country}  ({region})  {len(queries)} queries over {y0}-{y1}")

    # one viewCount-sorted pass per query over the whole window (~2 pages each)
    seen: dict[str, None] = {}
    for q in queries:
        for vid in yt.search_ids(q, region, None, 100, pa, pb):
            seen[vid] = None
    details = yt.video_details(list(seen))
    print(f"  {len(seen)} unique candidates; bucketing by year…")

    # bucket by publish year, take top-N per year by views
    by_year: dict[int, list] = {y: [] for y in years}
    for it in details:
        yr = int((it["snippet"].get("publishedAt", "0000")[:4]) or 0)
        if yr in by_year:
            by_year[yr].append(it)
    rows: list[dict] = []
    for y in years:
        lst = sorted(by_year[y],
                     key=lambda it: _int(it.get("statistics", {}).get("viewCount")) or 0,
                     reverse=True)[:top_n]
        print(f"    {y}: {len(by_year[y])} candidates -> kept {len(lst)}")
        rows.extend(_rows_from_details(lst, country, y, fetch_tr))
    return rows


def _rows_from_details(details: list[dict], country: str, year: int,
                       fetch_tr: bool) -> list[dict]:
    rows = []
    for it in tqdm(details, desc=f"  {country} {year}", leave=False):
        snip = it["snippet"]
        stats = it.get("statistics", {})
        cd = it.get("contentDetails", {})
        transcript = fetch_transcript(it["id"], ("en",)) if fetch_tr else None
        narrative_text = " ".join(filter(None, [
            snip.get("title", ""), snip.get("description", ""), transcript or ""
        ]))
        tags = tag_narratives(narrative_text)
        row = {
            "country": country,
            "year": year,
            "video_id": it["id"],
            "url": f"https://www.youtube.com/watch?v={it['id']}",
            "title": snip.get("title", ""),
            "description": (snip.get("description", "") or "")[:1000],
            "channel_title": snip.get("channelTitle", ""),
            "published_at": snip.get("publishedAt", ""),
            "duration_seconds": _iso_dur(cd.get("duration", "")),
            "view_count": _int(stats.get("viewCount")) or 0,
            "like_count": _int(stats.get("likeCount")),
            "comment_count": _int(stats.get("commentCount")),
            "default_language": snip.get("defaultLanguage"),
            "tags_str": "; ".join(snip.get("tags") or []),
            "transcript_excerpt": (transcript or "")[:3000],
            "summary": extractive_summary(transcript or snip.get("description", "")),
            "sentiment": sentiment(narrative_text),
        }
        for tag, val in tags.items():
            row[f"narr_{tag}"] = val
        rows.append(row)
        time.sleep(0.05)
    return rows


# =============================================================================
# 7. EXPORT + JOIN-TO-SALES
# =============================================================================
def export(all_rows: list[dict], outdir: Path, sales_csv: Path | None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "per_country").mkdir(exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "videos_all_countries.csv", index=False)

    for country, g in df.groupby("country"):
        slug = re.sub(r"\W+", "_", country).strip("_")
        g.to_csv(outdir / "per_country" / f"{slug}.csv", index=False)

    # narrative totals + share per country
    tag_cols = [c for c in df.columns if c.startswith("narr_")]
    totals = df.groupby("country")[tag_cols].sum()
    totals.to_csv(outdir / "narrative_by_country.csv")
    shares = totals.div(totals.sum(axis=1).replace(0, 1), axis=0)
    shares.to_csv(outdir / "narrative_share_by_country.csv")

    # weighted-by-views variant (each video weighted by viewCount)
    weighted = (df[tag_cols].multiply(df["view_count"], axis=0)
                  .groupby(df["country"]).sum())
    weighted_share = weighted.div(weighted.sum(axis=1).replace(0, 1), axis=0)
    weighted_share.to_csv(outdir / "narrative_share_view_weighted.csv")

    # join to sales
    if sales_csv and sales_csv.exists():
        sales = pd.read_csv(sales_csv, low_memory=False)
        # Aggregate sales per country across all years
        sales_country = (sales.groupby("Country")[["Value EUR", "Volume kg/l"]]
                              .sum().rename(columns={
                                  "Value EUR": "sales_value_eur_total",
                                  "Volume kg/l": "sales_volume_kgL_total",
                              }))
        # Also keep latest-year (2020) numbers
        last_year = sales[sales["Year"] == sales["Year"].max()]
        last = (last_year.groupby("Country")[["Value EUR", "Volume kg/l"]]
                          .sum().rename(columns={
                              "Value EUR": "sales_value_eur_2020",
                              "Volume kg/l": "sales_volume_kgL_2020",
                          }))
        joined = shares.join(sales_country, how="inner").join(last, how="inner")
        joined.to_csv(outdir / "country_vs_sales.csv")

        # correlation between each narrative share and sales value
        corrs = []
        for tag in tag_cols:
            for tgt in ["sales_value_eur_total", "sales_value_eur_2020",
                        "sales_volume_kgL_total"]:
                if tag in joined.columns and tgt in joined.columns:
                    r = joined[tag].corr(joined[tgt])
                    corrs.append({"narrative": tag, "target": tgt,
                                  "pearson_r": r})
        pd.DataFrame(corrs).to_csv(outdir / "narrative_sales_correlations.csv",
                                   index=False)

    write_report(df, outdir)
    print(f"\n✓ exported to {outdir.resolve()}")


def write_report(df: pd.DataFrame, outdir: Path) -> None:
    tag_cols = [c for c in df.columns if c.startswith("narr_")]
    shares = (df.groupby("country")[tag_cols].sum()
                .pipe(lambda x: x.div(x.sum(axis=1).replace(0, 1), axis=0)))
    lines = ["# Narrative shares by country", ""]
    lines.append(f"Videos analysed: **{len(df)}** across **{df['country'].nunique()}** countries.")
    lines.append("")
    lines.append(shares.round(3).to_markdown())
    lines.append("")
    lines.append("## Top video per country (by views)")
    for country, g in df.groupby("country"):
        top = g.sort_values("view_count", ascending=False).iloc[0]
        lines.append(f"- **{country}** — [{top['title']}]({top['url']}) — "
                     f"{int(top['view_count']):,} views — {top['published_at'][:10]}")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 8. UTILS
# =============================================================================
def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def _int(x):
    try: return int(x)
    except (TypeError, ValueError): return None

_DUR = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
def _iso_dur(s):
    m = _DUR.fullmatch(s or "")
    if not m: return 0
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return h*3600 + mi*60 + se


# =============================================================================
# 9. CLI
# =============================================================================
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sales-csv", default=str(Path(__file__).resolve().parent.parent /
                   "uploads" / "plant_based_food_sales_data.csv"),
                   help="Path to the plant_based_food_sales_data.csv file")
    p.add_argument("--top-n", type=int, default=100,
                   help="Top-N most-viewed videos per country (default 100)")
    p.add_argument("--countries", nargs="*", default=None,
                   help="Subset of countries (default: all in the CSV)")
    p.add_argument("--no-transcripts", action="store_true")
    p.add_argument("--years", default="2018-2020",
                   help="Year range, inclusive, e.g. 2018-2020 or 2016-2020. "
                        "Fetches top-N per country PER YEAR.")
    p.add_argument("--output-dir", default="output_multi_country")
    args = p.parse_args(argv)

    a, _, b = args.years.partition("-")
    years = list(range(int(a), int(b or a) + 1))

    key = os.environ.get("YOUTUBE_API_KEY") or YOUTUBE_API_KEY
    if not key:
        sys.exit("Set YOUTUBE_API_KEY env var, or paste your key into the file.")
    keys = [k.strip() for k in key.split(",") if k.strip()]  # comma-sep rotation
    print(f"API keys loaded: {len(keys)} (rotate on quota)")

    sales_path = Path(args.sales_csv)
    if not sales_path.exists():
        print(f"[warn] sales CSV not found at {sales_path}; running without it.",
              file=sys.stderr)

    # Determine country list
    if sales_path.exists():
        df_sales = pd.read_csv(sales_path, low_memory=False)
        countries = sorted(df_sales["Country"].dropna().unique().tolist())
    else:
        countries = list(COUNTRY_CONFIG)
    if args.countries:
        countries = [c for c in countries if c in args.countries]

    missing = [c for c in countries if c not in COUNTRY_CONFIG]
    if missing:
        print(f"[warn] no query config for: {missing}. Add them to "
              f"COUNTRY_CONFIG.", file=sys.stderr)
    countries = [c for c in countries if c in COUNTRY_CONFIG]
    print(f"Countries to process: {countries}")
    print(f"Years (top-{args.top_n} each): {years}")

    yt = YouTubeClient(keys)
    all_rows = []
    for country in countries:
        all_rows.extend(collect_country(country, COUNTRY_CONFIG[country],
                                        yt, args.top_n,
                                        not args.no_transcripts,
                                        years))

    outdir = Path(args.output_dir)
    export(all_rows, outdir, sales_path if sales_path.exists() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
