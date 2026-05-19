"""
Run this file directly in Visual Studio / VS Code.

Task:
Extract YouTube videos about UK plant-based food,
only for videos published from 2017-01-01 to 2020-12-31.

Before running:
1. Make sure requirements are installed:
   python -m pip install -r requirements.txt
2. Paste your YouTube Data API key into:
   youtube_api_key.txt
"""

from pathlib import Path


# Search settings
PAGES_PER_QUERY = 2
OUTPUT_DIR = "output_2017_2020"

# Date range:
# YouTube API uses publishedAfter inclusive-ish and publishedBefore before this date.
# So 2021-01-01T00:00:00Z includes everything up to 2020-12-31.
PUBLISHED_AFTER = "2017-01-01T00:00:00Z"
PUBLISHED_BEFORE = "2021-01-01T00:00:00Z"

# Keep this True for the first run. It is faster and more stable.
# Change to False later if you also want to try subtitle/transcript summaries.
SKIP_TRANSCRIPT = True

# False means keep stricter UK + plant-based related results.
KEEP_BROAD_RESULTS = True

QUERIES = [
    "plant based food UK consumer behaviour",
    "plant based meat UK consumers",
    "vegan food UK consumer trends",
    "plant based diet UK supermarket",
    "plant based food Britain narrative",
    "meat alternatives UK shoppers",
    "vegan products UK consumers",
    "plant based food United Kingdom sustainability",
]


def load_api_key():
    folder = Path(__file__).parent
    possible_files = [
        folder / "youtube_api_key.txt",
        folder / "youtube_api_key.txt.txt",
    ]

    for key_file in possible_files:
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip()
            if api_key and "PASTE" not in api_key.upper():
                print(f"Using API key from: {key_file.name}")
                return api_key

    print("No YouTube Data API key found.")
    print("Please paste your key into youtube_api_key.txt.")
    api_key = input("Or paste YouTube Data API key here: ").strip()
    if not api_key:
        raise SystemExit("Stopped: no API key.")
    return api_key


def main():
    try:
        from youtube_plant_based_uk_extractor import (
            fetch_video_details,
            is_relevant,
            normalise_video,
            save_outputs,
            search_videos,
        )
    except ModuleNotFoundError as error:
        print(f"Missing Python package: {error.name}")
        print("Run this command first:")
        print("python -m pip install -r requirements.txt")
        raise SystemExit(1)

    api_key = load_api_key()

    print("Searching YouTube videos from 2017-01-01 to 2020-12-31...")
    video_ids = search_videos(
        api_key=api_key,
        queries=QUERIES,
        max_pages_per_query=PAGES_PER_QUERY,
        published_after=PUBLISHED_AFTER,
        published_before=PUBLISHED_BEFORE,
    )
    print(f"Found {len(video_ids)} unique video ids.")

    details = fetch_video_details(api_key, video_ids)
    if not KEEP_BROAD_RESULTS:
        details = [item for item in details if is_relevant(item)]
    print(f"Keeping {len(details)} relevant UK plant-based food videos.")

    rows = []
    for index, item in enumerate(details, start=1):
        row = normalise_video(item, include_transcript=not SKIP_TRANSCRIPT)
        rows.append(row)
        print(f"[{index}/{len(details)}] {row['upload_date']} - {row['title'][:80]}")

    csv_path, json_path = save_outputs(rows, Path(OUTPUT_DIR))

    print()
    print("Done.")
    print("Date range: 2017-01-01 to 2020-12-31")
    print(f"CSV : {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()
