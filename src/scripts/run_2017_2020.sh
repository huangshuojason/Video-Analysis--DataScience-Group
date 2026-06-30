#!/bin/bash
# 每国每年 top100 vegan/plant-based 视频(宽词,整窗口分桶),时间窗 2017-2020。
# 逐国独立保存;撞配额上限自动停止,次日 re-run 续跑(已完成的国家会跳过)。
# KEY 从环境变量 YOUTUBE_API_KEY 读取。
set -u
SCRIPT="/Users/congtx/Documents/youtube_analysis/scripts/multi_country_narrative_extractor.py"
BASE="/Users/congtx/Documents/youtube_analysis/video_data_2017_2020"
mkdir -p "$BASE"

COUNTRIES=(Austria Belgium Denmark France Germany Italy Netherlands Poland Romania Spain "United Kingdom")

for c in "${COUNTRIES[@]}"; do
  if [ -f "$BASE/$c/videos_all_countries.csv" ]; then
    echo "SKIP $c (already done)"
    continue
  fi
  log="$BASE/${c// /_}.log"
  echo ">>> $c  $(date '+%Y-%m-%d %H:%M:%S')"
  python3 -W ignore "$SCRIPT" --countries "$c" --top-n 100 --no-transcripts \
    --years 2017-2020 \
    --sales-csv "/tmp/__no_sales_join__.csv" \
    --output-dir "$BASE/$c" > "$log" 2>&1

  # Only stop when ALL keys are exhausted. Benign key rotation prints
  # "[quota] ... rotating" (none of these strings); genuine exhaustion makes
  # search_ids print the full HttpError (rateLimitExceeded / Quota exceeded).
  if grep -qiE "quotaexceeded|ratelimitexceeded|dailylimitexceeded|quota exceeded" "$log"; then
    echo "!!! ALL KEYS EXHAUSTED at $c — partial deleted, stopping. Re-run to resume."
    echo "$c" > "$BASE/_stopped_at.txt"
    rm -rf "$BASE/$c"
    break
  fi
  if [ -f "$BASE/$c/videos_all_countries.csv" ]; then
    n=$(python3 -c "import pandas as pd;print(len(pd.read_csv('$BASE/$c/videos_all_countries.csv')))" 2>/dev/null || echo "?")
    echo "    $c done: $n videos"
  else
    echo "    $c produced no output (see $log)"
  fi
done
echo "=== RUN FINISHED $(date '+%Y-%m-%d %H:%M:%S') ==="
