#!/bin/bash
# 自动间断抓字幕:抓到被封就停,等冷却后自动续,直到全部处理完。
# 用法: bash fetch_transcripts_loop.sh            # 默认冷却
#       bash fetch_transcripts_loop.sh 1200 3000  # 自定义 base/long 冷却(秒)
set -u
DIR="/Users/congtx/Documents/youtube_analysis"
SCRIPT="$DIR/scripts/fetch_transcripts.py"
OUT="$DIR/transcripts.jsonl"
BASE_COOLDOWN="${1:-1200}"   # 有进展时的冷却(默认 20 分钟)
LONG_COOLDOWN="${2:-3000}"   # 仍被封(无进展)时的冷却(默认 50 分钟)

TOTAL=$(python3 -c "import pandas as pd;print(pd.read_csv('$DIR/video_data_2017_2020/ALL_11countries_2017_2020.csv').video_id.nunique())" 2>/dev/null)
echo "目标唯一视频数: $TOTAL"

while true; do
  before=$( [ -f "$OUT" ] && wc -l < "$OUT" | tr -d ' ' || echo 0 )
  if [ "${before:-0}" -ge "${TOTAL:-2720}" ]; then
    echo "=== 全部完成: $before/$TOTAL  $(date '+%H:%M:%S') ==="; break
  fi
  echo ">>> 周期开始 $(date '+%m-%d %H:%M:%S') — 已完成 $before/$TOTAL"
  python3 -W ignore "$SCRIPT" 2>&1 | grep -v "FutureWarning\|warnings.warn"
  after=$( [ -f "$OUT" ] && wc -l < "$OUT" | tr -d ' ' || echo 0 )

  if [ "${after:-0}" -ge "${TOTAL:-2720}" ]; then
    echo "=== 全部完成: $after/$TOTAL  $(date '+%H:%M:%S') ==="; break
  fi
  if [ "${after:-0}" -gt "${before:-0}" ]; then
    echo "    本周期新增 $((after-before)) 条(共 $after)。冷却 ${BASE_COOLDOWN}s"
    sleep "$BASE_COOLDOWN"
  else
    echo "    无进展(仍被封,$after)。延长冷却 ${LONG_COOLDOWN}s"
    sleep "$LONG_COOLDOWN"
  fi
done
