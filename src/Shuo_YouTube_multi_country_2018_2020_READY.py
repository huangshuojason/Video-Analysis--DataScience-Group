"""
Run this file directly in Visual Studio / VS Code.

Task:
Extract YouTube videos about plant-based food in multiple European countries,
for videos published from 2018-01-01 to 2020-12-31.

Before running:
1. Make sure requirements are installed:
   python -m pip install -r requirements.txt
2. Paste your YouTube Data API key into:
   youtube_api_key.txt

Important:
The YouTube Data API does not provide the publisher's IP address.
This script outputs country, search_region_code, search_language,
channel_country, default_language, and default_audio_language instead.
"""

import csv
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import requests
except ModuleNotFoundError:
    print("Missing package: requests")
    print("Run this command first:")
    print("python -m pip install -r requirements.txt")
    raise SystemExit(1)


PAGES_PER_QUERY = 2
OUTPUT_DIR = "output_2018_2020_multi_country"

PUBLISHED_AFTER = "2018-01-01T00:00:00Z"
PUBLISHED_BEFORE = "2021-01-01T00:00:00Z"

COUNTRIES = [
    {"country": "Austria", "region_code": "AT", "language": "de", "local_name": "Austria Osterreich"},
    {"country": "Belgium", "region_code": "BE", "language": "nl", "local_name": "Belgium Belgie Belgique"},
    {"country": "Denmark", "region_code": "DK", "language": "da", "local_name": "Denmark Danmark"},
    {"country": "France", "region_code": "FR", "language": "fr", "local_name": "France"},
    {"country": "Italy", "region_code": "IT", "language": "it", "local_name": "Italy Italia"},
    {"country": "Netherlands", "region_code": "NL", "language": "nl", "local_name": "Netherlands Nederland"},
    {"country": "Romania", "region_code": "RO", "language": "ro", "local_name": "Romania"},
    {"country": "Spain", "region_code": "ES", "language": "es", "local_name": "Spain Espana"},
    {"country": "United Kingdom", "region_code": "GB", "language": "en", "local_name": "United Kingdom UK Britain"},
]

QUERY_TEMPLATES = [
    "plant based food {country} consumer behaviour",
    "plant based meat {country} consumers",
    "vegan food {country} consumer trends",
    "plant based diet {country} supermarket",
    "meat alternatives {country} shoppers",
    "vegan products {country} consumers",
    "plant based food {local_name} sustainability",
    "vegan {local_name} supermarket",
]

NARRATIVE_KEYWORDS = {
    "health": ["health", "healthy", "nutrition", "protein", "diet", "wellbeing"],
    "environment": ["climate", "carbon", "sustainability", "sustainable", "planet", "emissions"],
    "animal_welfare": ["animal", "welfare", "cruelty", "ethical", "ethics"],
    "price_value": ["price", "cost", "cheap", "expensive", "value", "inflation"],
    "taste_convenience": ["taste", "flavour", "flavor", "delicious", "easy", "convenient"],
    "identity_lifestyle": ["vegan", "vegetarian", "flexitarian", "lifestyle", "identity"],
    "innovation_market": ["startup", "brand", "market", "supermarket", "retail", "product"],
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "i", "in", "is", "it", "its", "of", "on", "or", "our", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "with", "you", "your",
    "about", "also", "but", "can", "do", "if", "into", "more", "new", "not",
    "so", "than", "then", "there", "these", "those", "what", "when", "which",
    "who", "will", "would", "video", "youtube",
}


def project_root():
    return Path(__file__).resolve().parent.parent


def load_api_key():
    folder = Path(__file__).resolve().parent
    possible_files = [
        folder / "youtube_api_key.txt",
        folder / "youtube_api_key.txt.txt",
        folder.parent / "youtube_api_key.txt",
        folder.parent / "youtube_api_key.txt.txt",
    ]

    for key_file in possible_files:
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip()
            if api_key and "PASTE" not in api_key.upper():
                print(f"Using API key from: {key_file}")
                return api_key

    print("No YouTube Data API key found.")
    print("Please paste your key into youtube_api_key.txt.")
    api_key = input("Or paste YouTube Data API key here: ").strip()
    if not api_key:
        raise SystemExit("Stopped: no API key.")
    return api_key


def youtube_get(endpoint, params, api_key):
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}"
    params = dict(params)
    params["key"] = api_key
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_videos(api_key, queries, country_config):
    video_ids = []
    seen = set()

    for query in queries:
        page_token = None
        for _ in range(PAGES_PER_QUERY):
            params = {
                "part": "snippet",
                "type": "video",
                "q": query,
                "maxResults": 50,
                "safeSearch": "none",
                "order": "relevance",
                "regionCode": country_config["region_code"],
                "relevanceLanguage": country_config["language"],
                "publishedAfter": PUBLISHED_AFTER,
                "publishedBefore": PUBLISHED_BEFORE,
            }
            if page_token:
                params["pageToken"] = page_token

            data = youtube_get("search", params, api_key)
            for item in data.get("items", []):
                video_id = item.get("id", {}).get("videoId")
                if video_id and video_id not in seen:
                    seen.add(video_id)
                    video_ids.append(video_id)

            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.1)

    return video_ids


def fetch_video_details(api_key, video_ids):
    details = []
    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start:start + 50]
        if not chunk:
            continue
        data = youtube_get(
            "videos",
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(chunk),
                "maxResults": 50,
            },
            api_key,
        )
        details.extend(data.get("items", []))
        time.sleep(0.1)
    return details


def fetch_channel_details(api_key, channel_ids):
    channels = {}
    unique_channel_ids = sorted({channel_id for channel_id in channel_ids if channel_id})
    for start in range(0, len(unique_channel_ids), 50):
        chunk = unique_channel_ids[start:start + 50]
        if not chunk:
            continue
        data = youtube_get(
            "channels",
            {
                "part": "snippet,statistics,brandingSettings",
                "id": ",".join(chunk),
                "maxResults": 50,
            },
            api_key,
        )
        for item in data.get("items", []):
            channels[item.get("id")] = item
        time.sleep(0.1)
    return channels


def clean_text(text):
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def summarize_text(text, max_sentences=3):
    text = clean_text(text)
    if not text:
        return ""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) > 25
    ]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = re.findall(r"[a-zA-Z']+", text.lower())
    word_counts = Counter(word for word in words if word not in STOPWORDS and len(word) > 2)
    if not word_counts:
        return " ".join(sentences[:max_sentences])

    scored = []
    for index, sentence in enumerate(sentences):
        sentence_words = re.findall(r"[a-zA-Z']+", sentence.lower())
        score = sum(word_counts.get(word, 0) for word in sentence_words)
        score = score / max(len(sentence_words), 1)
        if index < 2:
            score += 0.15
        scored.append((score, index, sentence))

    top = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(sentence for _, _, sentence in sorted(top, key=lambda row: row[1]))


def classify_narratives(text):
    lower = (text or "").lower()
    hits = {}
    for label, keywords in NARRATIVE_KEYWORDS.items():
        count = sum(lower.count(keyword) for keyword in keywords)
        if count:
            hits[label] = count
    return sorted(hits, key=hits.get, reverse=True)


def normalise_video(item, search_row, channel):
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    video_id = item.get("id", "")
    title = clean_text(snippet.get("title", ""))
    description = clean_text(snippet.get("description", ""))
    published_at = snippet.get("publishedAt", "")
    upload_date = published_at[:10] if published_at else ""
    summary = summarize_text(description, max_sentences=3)
    narratives = classify_narratives(" ".join([title, description, summary]))
    channel_country = channel.get("brandingSettings", {}).get("channel", {}).get("country", "")

    return {
        "country": search_row["country"],
        "search_region_code": search_row["search_region_code"],
        "search_language": search_row["search_language"],
        "channel_country": channel_country,
        "default_language": snippet.get("defaultLanguage", ""),
        "default_audio_language": snippet.get("defaultAudioLanguage", ""),
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "upload_date": upload_date,
        "channel": snippet.get("channelTitle", ""),
        "channel_id": snippet.get("channelId", ""),
        "summary": summary,
        "narratives": "; ".join(narratives),
        "view_count": statistics.get("viewCount", ""),
        "like_count": statistics.get("likeCount", ""),
        "comment_count": statistics.get("commentCount", ""),
        "description": description,
    }


def save_outputs(rows):
    output_dir = project_root() / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"youtube_plant_based_multi_country_{stamp}.csv"
    json_path = output_dir / f"youtube_plant_based_multi_country_{stamp}.json"

    fieldnames = [
        "country",
        "search_region_code",
        "search_language",
        "channel_country",
        "default_language",
        "default_audio_language",
        "video_id",
        "url",
        "title",
        "upload_date",
        "channel",
        "channel_id",
        "summary",
        "narratives",
        "view_count",
        "like_count",
        "comment_count",
        "description",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    return csv_path, json_path


def main():
    api_key = load_api_key()
    print("Searching YouTube videos from 2018-01-01 to 2020-12-31...")

    search_rows = []
    seen_country_video_pairs = set()

    for country_config in COUNTRIES:
        country = country_config["country"]
        queries = [
            template.format(
                country=country,
                local_name=country_config["local_name"],
            )
            for template in QUERY_TEMPLATES
        ]

        print(f"\nSearching {country}...")
        video_ids = search_videos(api_key, queries, country_config)
        print(f"Found {len(video_ids)} unique video ids for {country}.")

        for video_id in video_ids:
            pair = (country, video_id)
            if pair in seen_country_video_pairs:
                continue
            seen_country_video_pairs.add(pair)
            search_rows.append({
                "country": country,
                "search_region_code": country_config["region_code"],
                "search_language": country_config["language"],
                "video_id": video_id,
            })

    all_video_ids = sorted({row["video_id"] for row in search_rows})
    print(f"\nFound {len(all_video_ids)} unique video ids across all countries.")

    details = fetch_video_details(api_key, all_video_ids)
    details_by_id = {item.get("id"): item for item in details}

    channel_ids = [
        item.get("snippet", {}).get("channelId")
        for item in details
        if item.get("snippet", {}).get("channelId")
    ]
    channels_by_id = fetch_channel_details(api_key, channel_ids)

    rows = []
    for index, search_row in enumerate(search_rows, start=1):
        item = details_by_id.get(search_row["video_id"])
        if not item:
            continue

        channel_id = item.get("snippet", {}).get("channelId", "")
        channel = channels_by_id.get(channel_id, {})
        row = normalise_video(item, search_row, channel)
        rows.append(row)
        print(f"[{index}/{len(search_rows)}] {row['country']} - {row['upload_date']} - {row['title'][:80]}")

    csv_path, json_path = save_outputs(rows)

    print()
    print("Done.")
    print("Date range: 2018-01-01 to 2020-12-31")
    print(f"Rows saved: {len(rows)}")
    print(f"CSV : {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()
