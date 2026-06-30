"""
Fetch YouTube transcripts (captions) for the 4170 collected videos.

YouTube IP-blocks rapid scraping, so this script:
  - rate-limits (random delay between requests),
  - backs off on block (sleep + retry), stops cleanly if persistently blocked,
  - saves incrementally to transcripts.jsonl and RESUMES (skips done ids).

Transcript is fetched in the country's language first, then English, then any
available track (records which language). No API key needed (not the Data API).

Usage:
    python fetch_transcripts.py            # all videos, resumable
    python fetch_transcripts.py --limit 50 # test on first 50
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

BASE = Path("/Users/congtx/Documents/youtube_analysis")
SRC = BASE / "video_data_2017_2020" / "ALL_11countries_2017_2020.csv"
OUT = BASE / "transcripts.jsonl"

COUNTRY_LANG = {
    "Austria": "de", "Belgium": "nl", "Denmark": "da", "France": "fr",
    "Germany": "de", "Italy": "it", "Netherlands": "nl", "Poland": "pl",
    "Romania": "ro", "Spain": "es", "United Kingdom": "en",
}

api = YouTubeTranscriptApi()


def fetch_one(vid: str, lang: str):
    """Return (text, lang_code) or (None, None). Raises on IP block."""
    prefs = [lang, "en"] if lang != "en" else ["en"]
    try:
        f = api.fetch(vid, languages=prefs)
        return " ".join(s.text for s in f if getattr(s, "text", "")).strip(), prefs[0]
    except NoTranscriptFound:
        tl = api.list(vid)
        picked = next(iter(tl))
        f = picked.fetch()
        return " ".join(s.text for s in f if getattr(s, "text", "")).strip(), \
            picked.language_code


def is_block(e: Exception) -> bool:
    n = type(e).__name__.lower()
    s = str(e).lower()
    return "ipblock" in n or "requestblock" in n or "blocked" in s or \
        "too many request" in s or "429" in s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-delay", type=float, default=1.5)
    p.add_argument("--max-delay", type=float, default=3.5)
    args = p.parse_args()

    df = pd.read_csv(SRC)[["video_id", "country"]].drop_duplicates("video_id")
    if args.limit:
        df = df.head(args.limit)

    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["video_id"])
            except Exception:
                pass
    print(f"total {len(df)} videos, {len(done)} already done, "
          f"{len(df) - len(df[df.video_id.isin(done)])} to go")

    hit = miss = 0
    with OUT.open("a", encoding="utf-8") as out:
        for i, (vid, country) in enumerate(zip(df.video_id, df.country), 1):
            if vid in done:
                continue
            lang = COUNTRY_LANG.get(country, "en")
            for attempt in range(4):
                try:
                    text, tlang = fetch_one(vid, lang)
                    rec = {"video_id": vid, "country": country,
                           "lang": tlang if text else None,
                           "n_chars": len(text) if text else 0,
                           "text": text or ""}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    if text:
                        hit += 1
                    else:
                        miss += 1
                    break
                except Exception as e:
                    if is_block(e):
                        # stop fast on block; the outer loop handles the long
                        # cooldown before resuming.
                        wait = 30
                        print(f"  [BLOCKED] at {vid} (attempt {attempt+1}) "
                              f"-> sleep {wait}s", file=sys.stderr)
                        time.sleep(wait)
                        if attempt == 1:
                            print(f"!!! Blocked. Stopping at #{i}. "
                                  f"Resume later.", file=sys.stderr)
                            print(f"progress: hit={hit} miss={miss}")
                            return 2
                        continue
                    # no transcript / unavailable -> record miss
                    rec = {"video_id": vid, "country": country, "lang": None,
                           "n_chars": 0, "text": ""}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    miss += 1
                    break
            if i % 25 == 0:
                print(f"  {i}/{len(df)}  hit={hit} miss={miss}")
            time.sleep(random.uniform(args.min_delay, args.max_delay))

    print(f"DONE. hit={hit} miss={miss} (hit rate {hit/(hit+miss+1e-9)*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
