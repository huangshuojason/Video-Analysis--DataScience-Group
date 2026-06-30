#!/bin/bash
# 逐国抓取 2016-2020 plant-based 视频(含字幕),撞到配额上限自动停止。
# KEY 从环境变量 YOUTUBE_API_KEY 读取,不写进文件。
set -u
SCRIPT="/Users/congtx/Documents/youtube_analysis/scripts/multi_country_narrative_extractor.py"
BASE="/Users/congtx/Documents/youtube_analysis/video_data_2016_2020"
mkdir -p "$BASE"

COUNTRIES=(Austria Belgium Denmark France Germany Italy Netherlands Poland Romania Spain "United Kingdom")

for c in "${COUNTRIES[@]}"; do
  # 已完成则跳过(便于明天续跑)
  if [ -f "$BASE/$c/videos_all_countries.csv" ]; then
    echo "SKIP $c (already done)"
    continue
  fi
  log="$BASE/${c// /_}.log"
  echo ">>> $c  $(date +%H:%M:%S)"
  python3 -W ignore "$SCRIPT" --countries "$c" --top-n 100 --no-transcripts \
    --published-after 2016-01-01T00:00:00Z \
    --published-before 2021-01-01T00:00:00Z \
    --sales-csv "/tmp/__no_sales_join__.csv" \
    --output-dir "$BASE/$c" > "$log" 2>&1

  if grep -qi "quota\|quotaExceeded\|dailyLimitExceeded" "$log"; then
    echo "!!! QUOTA HIT at $c — stopping. Resume tomorrow."
    echo "$c" > "$BASE/_stopped_at.txt"
    rm -rf "$BASE/$c"   # 该国不完整,删除以便明天重跑
    break
  fi
  if [ -f "$BASE/$c/videos_all_countries.csv" ]; then
    n=$(python3 -c "import pandas as pd;print(len(pd.read_csv('$BASE/$c/videos_all_countries.csv')))" 2>/dev/null || echo "?")
    echo "    $c done: $n videos"
  else
    echo "    $c produced no output (see $log)"
  fi
done
echo "=== RUN FINISHED $(date +%H:%M:%S) ==="
