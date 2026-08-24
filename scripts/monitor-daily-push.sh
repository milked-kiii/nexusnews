#!/bin/bash
# Monitor today's 11:00 CST (03:00 UTC) schedule run on GitHub Actions.
# Polls every 45s, exits when a schedule run created after 03:00 UTC today
# reaches a completed state. Prints the result.
set -u

REPO_DIR="/Users/milked/Desktop/Nexusnews"
TOKEN=$(git -C "$REPO_DIR" remote get-url origin | sed -E 's|https://[^:]+:([^@]+)@.*|\1|')
if [ -z "$TOKEN" ]; then
  echo "ERROR: could not extract GitHub token from remote URL"
  exit 1
fi

# UTC date boundary for today's 03:00 trigger
TODAY_UTC=$(date -u +%Y-%m-%d)
START_MARKER="${TODAY_UTC}T03:00:00Z"

MAX_ATTEMPTS=160   # 160 * 45s = 2 hours
for ((i=1; i<=MAX_ATTEMPTS; i++)); do
  RUNS=$(curl -s -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/milked-kiii/nexusnews/actions/runs?per_page=20" 2>/dev/null)

  # Find today's schedule runs (created >= 03:00 UTC, event=schedule)
  FOUND=$(echo "$RUNS" | python3 -c "
import sys, json
from datetime import datetime
marker = '$START_MARKER'
runs = json.load(sys.stdin).get('workflow_runs', [])
today_sched = [r for r in runs if r.get('event') == 'schedule' and r.get('created_at', '') >= marker]
if not today_sched:
    print('WAITING')
else:
    r = today_sched[0]
    print(f\"{r['status']}|{r.get('conclusion') or 'none'}|{r['created_at']}|{r.get('display_title','')}\")
" 2>/dev/null)

  case "$FOUND" in
    WAITING)
      echo "[$(date '+%H:%M:%S')] attempt $i/$MAX_ATTEMPTS: 今天 11:00 的 schedule run 还没出现，继续等…"
      ;;
    completed\|*)
      echo "[$(date '+%H:%M:%S')] ✅ schedule run 已完成"
      echo "$FOUND"
      exit 0
      ;;
    in_progress\|*)
      echo "[$(date '+%H:%M:%S')] attempt $i/$MAX_ATTEMPTS: run 执行中 (created=$FOUND)"
      ;;
    *)
      echo "[$(date '+%H:%M:%S')] attempt $i/$MAX_ATTEMPTS: 无法解析 ($FOUND)，继续等…"
      ;;
  esac
  sleep 45
done

echo "❌ 2 小时后仍未等到今天 11:00 的 schedule 触发（可能被 GitHub 延迟更久或跳过）"
exit 1
