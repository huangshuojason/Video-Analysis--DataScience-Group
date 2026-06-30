"""
YouTube Narrative & Consumer-Behavior Extractor
================================================

Goal
----
Pull rich, structured data from YouTube videos on ANY topic so that you can
study how consumer narratives shift over time. The default configuration
targets "plant-based burgers in Japan" (プラントベース バーガー 日本), but you
can switch topic / language / region / time window with a single config block.

What gets extracted per video
-----------------------------
1.  videoId, URL
2.  title, description, tags, category
3.  channelId, channelTitle, channelSubscriberCount, channelCountry
4.  publishedAt (upload date), durationSeconds
5.  viewCount, likeCount, commentCount, favoriteCount
6.  defaultLanguage / defaultAudioLanguage
7.  thumbnail URL
8.  transcript (if captions are available, any language; auto-translated to EN
    when the requested language is missing)
9.  transcript-based summary (extractive, no external LLM required; optional
    LLM hook included)
10. top-N comments with author, like count, publish date
11. simple narrative signals:
        - keyword frequency
        - sentiment (VADER)
        - named entities (brands, places) via spaCy if installed
        - "narrative tags" (health / environment / animal-welfare / taste /
          price / convenience / novelty / cultural-fit) via keyword rules
12. per-video and per-channel aggregations
13. CSV + JSON exports + a Markdown report

Why this works for "narrative shift" research
---------------------------------------------
By collecting publishedAt + transcript + comments + narrative tags, you can
group videos into time buckets (quarter / year) and watch how the dominant
"reasons people care" change — e.g. early videos may be 80 % "novelty / taste",
later videos may shift toward "health" or "environment". The same pipeline
works for EV adoption, AI tools, Korean skincare, anything.

Setup
-----
pip install --upgrade google-api-python-client youtube-transcript-api \
            vaderSentiment pandas tqdm python-dateutil
# optional, for entities:
pip install spacy && python -m spacy download xx_ent_wiki_sm

Then set an API key:
    export YOUTUBE_API_KEY=""   # from Google Cloud Console
                                       # enable "YouTube Data API v3"

Run
---
python youtube_narrative_extractor.py
    # uses the default plant-based-burger-Japan config

# or override:
python youtube_narrative_extractor.py \
    --query "電気自動車 普及" --region JP --language ja \
    --published-after 2020-01-01 --max-videos 100

Outputs (in ./output/<run_slug>/)
---------------------------------
videos.csv          one row per video, all fields above
videos.json         same data, nested
comments.csv        one row per comment
narrative_tags.csv  long-format video × tag matrix
report.md           human-readable summary with time-bucketed narrative shift
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --- third-party (lazy / optional imports below) -----------------------------
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    sys.exit("Missing dep: pip install google-api-python-client")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
    )
except ImportError:
    sys.exit("Missing dep: pip install youtube-transcript-api")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    sys.exit("Missing dep: pip install vaderSentiment")

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dep: pip install pandas")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):  # graceful fallback
        return x

from dateutil import parser as dtparser


# =============================================================================
# 0. PASTE YOUR API KEY HERE  (between the quotes)
# Get it from https://console.cloud.google.com/  ->  YouTube Data API v3
# =============================================================================
YOUTUBE_API_KEY = ""   # e.g. "AIzaSyB1234567890abcdefghijklmnop"


# =============================================================================
# 1. CONFIG  – change this block to study any topic / region / window
# =============================================================================
@dataclass
class ExtractionConfig:
    query: str = "プラントベース バーガー 日本"   # search query
    extra_queries: list[str] = field(default_factory=lambda: [
        # synonyms / English fallbacks broaden recall
        "plant based burger Japan",
        "代替肉 バーガー",
        "ヴィーガン バーガー 日本",
        "vegan burger Tokyo",
    ])
    region_code: str = "JP"          # ISO 3166-1 alpha-2
    relevance_language: str = "ja"   # search bias
    published_after: str = "2018-01-01"
    published_before: str | None = None
    max_videos_per_query: int = 50   # YouTube caps per call at 50; we paginate
    max_total_videos: int = 150
    max_comments_per_video: int = 50
    fetch_transcripts: bool = True
    transcript_languages: tuple[str, ...] = ("ja", "en")
    output_dir: Path = Path("output")
    api_key_env: str = "YOUTUBE_API_KEY"


# =============================================================================
# 2. NARRATIVE TAG DICTIONARY
# Edit / extend per study. Each tag maps to keywords (multi-lingual).
# =============================================================================
NARRATIVE_TAGS: dict[str, list[str]] = {
    "health":        ["健康", "ヘルシー", "コレステロール", "low fat", "healthy",
                      "nutrition", "protein", "栄養"],
    "environment":   ["環境", "サステナブル", "気候", "co2", "carbon",
                      "climate", "sustainable", "地球"],
    "animal_welfare":["動物", "アニマルウェルフェア", "cruelty", "animal welfare",
                      "倫理", "ethical"],
    "taste":         ["美味しい", "うまい", "まずい", "tasty", "flavor", "taste",
                      "delicious", "食感", "texture"],
    "price":         ["値段", "高い", "安い", "price", "expensive", "cheap",
                      "コスパ", "value"],
    "convenience":   ["便利", "手軽", "convenient", "easy", "quick"],
    "novelty":       ["新しい", "話題", "trend", "new", "初めて", "first time",
                      "初挑戦"],
    "cultural_fit":  ["日本", "和", "japanese", "和風", "culture", "文化",
                      "受け入れ"],
}


# =============================================================================
# 3. DATA MODEL
# =============================================================================
@dataclass
class VideoRecord:
    video_id: str
    url: str
    title: str
    description: str
    tags: list[str]
    category_id: str
    channel_id: str
    channel_title: str
    channel_subscribers: int | None
    channel_country: str | None
    published_at: str
    duration_seconds: int
    view_count: int
    like_count: int | None
    comment_count: int | None
    default_language: str | None
    default_audio_language: str | None
    thumbnail_url: str
    transcript: str | None
    transcript_language: str | None
    summary: str
    sentiment_compound: float | None
    narrative_tags: dict[str, int]
    top_keywords: list[tuple[str, int]]
    entities: list[str]
    comments: list[dict]


# =============================================================================
# 4. YOUTUBE API CLIENT WRAPPER
# =============================================================================
class YouTubeClient:
    def __init__(self, api_key: str):
        self.api = build("youtube", "v3", developerKey=api_key,
                         cache_discovery=False)

    # --- search ---------------------------------------------------------------
    def search_video_ids(self, cfg: ExtractionConfig) -> list[str]:
        """Run cfg.query (+ extras) and collect unique videoIds up to the cap."""
        seen: dict[str, None] = {}
        queries = [cfg.query, *cfg.extra_queries]
        for q in queries:
            page_token = None
            collected_for_q = 0
            while True:
                req = self.api.search().list(
                    q=q,
                    part="id",
                    type="video",
                    maxResults=min(50, cfg.max_videos_per_query
                                   - collected_for_q),
                    regionCode=cfg.region_code,
                    relevanceLanguage=cfg.relevance_language,
                    publishedAfter=_iso(cfg.published_after),
                    publishedBefore=_iso(cfg.published_before),
                    pageToken=page_token,
                    order="relevance",
                )
                try:
                    resp = req.execute()
                except HttpError as e:
                    print(f"[search] {q!r}: {e}", file=sys.stderr)
                    break

                for item in resp.get("items", []):
                    vid = item["id"]["videoId"]
                    if vid not in seen:
                        seen[vid] = None
                        collected_for_q += 1
                        if len(seen) >= cfg.max_total_videos:
                            return list(seen)

                page_token = resp.get("nextPageToken")
                if not page_token or collected_for_q >= cfg.max_videos_per_query:
                    break
        return list(seen)

    # --- video details (batched 50 at a time) ---------------------------------
    def fetch_video_details(self, video_ids: list[str]) -> list[dict]:
        out = []
        for chunk in _chunks(video_ids, 50):
            resp = self.api.videos().list(
                id=",".join(chunk),
                part="snippet,statistics,contentDetails,topicDetails",
            ).execute()
            out.extend(resp.get("items", []))
        return out

    # --- channel details (for subscriber count) -------------------------------
    def fetch_channel_details(self, channel_ids: list[str]) -> dict[str, dict]:
        out = {}
        for chunk in _chunks(list(set(channel_ids)), 50):
            resp = self.api.channels().list(
                id=",".join(chunk),
                part="snippet,statistics",
            ).execute()
            for item in resp.get("items", []):
                out[item["id"]] = item
        return out

    # --- comments -------------------------------------------------------------
    def fetch_top_comments(self, video_id: str, n: int) -> list[dict]:
        if n <= 0:
            return []
        try:
            resp = self.api.commentThreads().list(
                videoId=video_id,
                part="snippet",
                maxResults=min(100, n),
                order="relevance",
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            # Most common: comments disabled (403)
            return []

        comments = []
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": top.get("authorDisplayName"),
                "text": top.get("textDisplay"),
                "like_count": top.get("likeCount"),
                "published_at": top.get("publishedAt"),
            })
            if len(comments) >= n:
                break
        return comments


# =============================================================================
# 5. TRANSCRIPT + SUMMARY
# =============================================================================
def fetch_transcript(video_id: str,
                     languages: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Return (text, language_code) or (None, None) if unavailable."""
    try:
        # Try requested languages first, then any auto-translated
        try:
            tr = YouTubeTranscriptApi.get_transcript(
                video_id, languages=list(languages))
            lang = languages[0]
        except NoTranscriptFound:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            # pick the first available, then translate to first preferred lang
            picked = next(iter(transcripts))
            try:
                picked = picked.translate(languages[0])
                lang = languages[0]
            except Exception:
                lang = picked.language_code
            tr = picked.fetch()
        text = " ".join(seg["text"] for seg in tr if seg.get("text"))
        return text.strip(), lang
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None, None
    except Exception as e:
        # Network / parsing edge cases
        return None, None


_SENT_SPLIT = re.compile(r"(?<=[。．\.!?！？])\s+")

def extractive_summary(text: str, max_sentences: int = 5) -> str:
    """A no-dependency extractive summary: top-N sentences by keyword overlap.
    Replace with an LLM call if you need higher quality.
    """
    if not text:
        return ""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # crude term-frequency scoring
    words = re.findall(r"\w+", text.lower())
    freqs = Counter(w for w in words if len(w) > 2)
    scored = []
    for s in sentences:
        ws = re.findall(r"\w+", s.lower())
        score = sum(freqs.get(w, 0) for w in ws) / (len(ws) + 1)
        scored.append((score, s))
    top = sorted(scored, key=lambda x: -x[0])[:max_sentences]
    # restore original order
    top_set = {s for _, s in top}
    return " ".join(s for s in sentences if s in top_set)


# Optional: plug an LLM here if you want abstractive summaries.
def llm_summary(text: str) -> str | None:
    """Stub. Wire to Claude/OpenAI as needed. Returns None by default."""
    return None


# =============================================================================
# 6. NARRATIVE / SENTIMENT / ENTITIES
# =============================================================================
_VADER = SentimentIntensityAnalyzer()

def sentiment_score(text: str) -> float | None:
    if not text:
        return None
    # VADER is English-tuned; for JA you'd swap in oseti or asari. Still a
    # useful directional signal on the EN portions.
    return _VADER.polarity_scores(text)["compound"]


def tag_narratives(text: str) -> dict[str, int]:
    if not text:
        return {k: 0 for k in NARRATIVE_TAGS}
    low = text.lower()
    out = {}
    for tag, kws in NARRATIVE_TAGS.items():
        out[tag] = sum(low.count(kw.lower()) for kw in kws)
    return out


def top_keywords(text: str, n: int = 15) -> list[tuple[str, int]]:
    if not text:
        return []
    STOP = set("the a an and or of to in on for is are was were be been being "
               "this that with as it its by from at i you we they he she them "
               "です ます した して する こと もの ない なる いる ある ため".split())
    words = re.findall(r"[A-Za-z぀-ヿ一-鿿]{2,}", text)
    counts = Counter(w.lower() for w in words if w.lower() not in STOP)
    return counts.most_common(n)


def extract_entities(text: str) -> list[str]:
    if not text:
        return []
    try:
        import spacy  # optional
        try:
            nlp = extract_entities._nlp  # type: ignore[attr-defined]
        except AttributeError:
            nlp = spacy.load("xx_ent_wiki_sm")
            extract_entities._nlp = nlp  # type: ignore[attr-defined]
        doc = nlp(text[:100_000])
        return list({ent.text for ent in doc.ents
                     if ent.label_ in {"ORG", "GPE", "LOC", "PRODUCT", "PERSON"}})
    except Exception:
        return []


# =============================================================================
# 7. ORCHESTRATION
# =============================================================================
def build_records(cfg: ExtractionConfig, yt: YouTubeClient) -> list[VideoRecord]:
    print(f"[1/4] searching: {cfg.query!r}  (+ {len(cfg.extra_queries)} synonyms)")
    video_ids = yt.search_video_ids(cfg)
    print(f"      found {len(video_ids)} unique video ids")

    print("[2/4] fetching video metadata")
    items = yt.fetch_video_details(video_ids)

    print("[3/4] fetching channel metadata")
    channel_ids = [it["snippet"]["channelId"] for it in items]
    channels = yt.fetch_channel_details(channel_ids)

    print("[4/4] enriching: transcripts, summaries, comments, narrative tags")
    records: list[VideoRecord] = []
    for it in tqdm(items):
        vid = it["id"]
        snip = it["snippet"]
        stats = it.get("statistics", {})
        cd = it.get("contentDetails", {})
        ch = channels.get(snip["channelId"], {})
        ch_stats = ch.get("statistics", {})
        ch_snip = ch.get("snippet", {})

        transcript, tlang = (None, None)
        if cfg.fetch_transcripts:
            transcript, tlang = fetch_transcript(vid, cfg.transcript_languages)

        # Source for narrative analysis: transcript ▸ description
        narrative_text = transcript or snip.get("description", "")
        summary = (llm_summary(narrative_text)
                   or extractive_summary(narrative_text))

        comments = yt.fetch_top_comments(vid, cfg.max_comments_per_video)

        records.append(VideoRecord(
            video_id=vid,
            url=f"https://www.youtube.com/watch?v={vid}",
            title=snip.get("title", ""),
            description=snip.get("description", ""),
            tags=snip.get("tags", []) or [],
            category_id=snip.get("categoryId", ""),
            channel_id=snip["channelId"],
            channel_title=snip.get("channelTitle", ""),
            channel_subscribers=_int(ch_stats.get("subscriberCount")),
            channel_country=ch_snip.get("country"),
            published_at=snip.get("publishedAt", ""),
            duration_seconds=_iso8601_duration_to_seconds(cd.get("duration", "")),
            view_count=_int(stats.get("viewCount")) or 0,
            like_count=_int(stats.get("likeCount")),
            comment_count=_int(stats.get("commentCount")),
            default_language=snip.get("defaultLanguage"),
            default_audio_language=snip.get("defaultAudioLanguage"),
            thumbnail_url=(snip.get("thumbnails", {})
                               .get("high", {}).get("url", "")),
            transcript=transcript,
            transcript_language=tlang,
            summary=summary,
            sentiment_compound=sentiment_score(narrative_text),
            narrative_tags=tag_narratives(narrative_text),
            top_keywords=top_keywords(narrative_text),
            entities=extract_entities(narrative_text),
            comments=comments,
        ))
        time.sleep(0.05)  # be polite
    return records


# =============================================================================
# 8. EXPORT + REPORT
# =============================================================================
def export(records: list[VideoRecord], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # JSON
    (outdir / "videos.json").write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV (flatten narrative tags)
    flat = []
    for r in records:
        row = asdict(r)
        for tag, val in row.pop("narrative_tags").items():
            row[f"tag_{tag}"] = val
        row["top_keywords"] = "; ".join(f"{w}:{c}" for w, c in row["top_keywords"])
        row["entities"] = "; ".join(row["entities"])
        row["tags"] = "; ".join(row["tags"])
        row["comments_n"] = len(row.pop("comments"))
        # truncate transcript so the CSV stays openable
        if row.get("transcript"):
            row["transcript"] = row["transcript"][:5000]
        flat.append(row)
    pd.DataFrame(flat).to_csv(outdir / "videos.csv", index=False)

    # comments.csv
    crows = []
    for r in records:
        for c in r.comments:
            crows.append({"video_id": r.video_id, **c})
    pd.DataFrame(crows).to_csv(outdir / "comments.csv", index=False)

    # narrative_tags.csv (long)
    trows = []
    for r in records:
        for tag, v in r.narrative_tags.items():
            trows.append({
                "video_id": r.video_id,
                "published_at": r.published_at,
                "tag": tag,
                "count": v,
            })
    pd.DataFrame(trows).to_csv(outdir / "narrative_tags.csv", index=False)

    write_report(records, outdir / "report.md")
    print(f"\n✓ exported to {outdir.resolve()}")


def write_report(records: list[VideoRecord], path: Path) -> None:
    df = pd.DataFrame([{
        "published_at": r.published_at,
        "views": r.view_count,
        **{f"tag_{k}": v for k, v in r.narrative_tags.items()},
    } for r in records])
    if df.empty:
        path.write_text("No videos collected.", encoding="utf-8")
        return
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["year"] = df["published_at"].dt.year

    yearly = df.groupby("year")[[c for c in df.columns
                                 if c.startswith("tag_")]].sum()
    # normalize per year so we see narrative SHARE, not absolute volume
    yearly_share = yearly.div(yearly.sum(axis=1).replace(0, 1), axis=0)

    lines = ["# Narrative-shift report", ""]
    lines.append(f"- Videos analysed: **{len(records)}**")
    lines.append(f"- Date range: {df['published_at'].min().date()} → "
                 f"{df['published_at'].max().date()}")
    lines.append(f"- Total views: **{int(df['views'].sum()):,}**")
    lines.append("")
    lines.append("## Narrative share by year")
    lines.append("")
    lines.append(yearly_share.round(2).to_markdown())
    lines.append("")
    lines.append("## Top 10 videos by views")
    top = sorted(records, key=lambda r: -r.view_count)[:10]
    for r in top:
        lines.append(f"- [{r.title}]({r.url}) — {r.view_count:,} views — "
                     f"{r.published_at[:10]} — *{r.channel_title}*")
        if r.summary:
            lines.append(f"    > {r.summary[:300]}{'…' if len(r.summary)>300 else ''}")
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 9. UTILITIES
# =============================================================================
def _chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def _int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def _iso(date_str: str | None) -> str | None:
    if not date_str:
        return None
    return dtparser.parse(date_str).astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

_DUR_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
def _iso8601_duration_to_seconds(d: str) -> int:
    m = _DUR_RE.fullmatch(d or "")
    if not m:
        return 0
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


# =============================================================================
# 10. CLI
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    cfg = ExtractionConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--query", default=cfg.query)
    p.add_argument("--region", default=cfg.region_code)
    p.add_argument("--language", default=cfg.relevance_language)
    p.add_argument("--published-after", default=cfg.published_after)
    p.add_argument("--published-before", default=cfg.published_before)
    p.add_argument("--max-videos", type=int, default=cfg.max_total_videos)
    p.add_argument("--max-comments", type=int, default=cfg.max_comments_per_video)
    p.add_argument("--no-transcripts", action="store_true")
    p.add_argument("--output-dir", default=str(cfg.output_dir))
    args = p.parse_args(argv)

    cfg.query = args.query
    cfg.region_code = args.region
    cfg.relevance_language = args.language
    cfg.published_after = args.published_after
    cfg.published_before = args.published_before
    cfg.max_total_videos = args.max_videos
    cfg.max_comments_per_video = args.max_comments
    cfg.fetch_transcripts = not args.no_transcripts
    slug = re.sub(r"\W+", "_", cfg.query)[:60].strip("_") or "run"
    cfg.output_dir = Path(args.output_dir) / slug

    # Prefer the env var; otherwise fall back to the hardcoded key at the top.
    api_key = os.environ.get(cfg.api_key_env) or YOUTUBE_API_KEY
    if not api_key:
        sys.exit(f"Set {cfg.api_key_env} env var, or paste your key into "
                 f"YOUTUBE_API_KEY at the top of this file.")

    yt = YouTubeClient(api_key)
    records = build_records(cfg, yt)
    export(records, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
