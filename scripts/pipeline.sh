#!/usr/bin/env bash
# Pipeline: fetch -> normalize -> diff -> review -> render -> commit -> push.
# Safe to run from a systemd timer. Idempotent; review/push are non-fatal.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prevent concurrent runs (systemd timer + manual invocation would race on git).
LOCK_FD=9
LOCK_FILE="$ROOT/.pipeline.lock"
exec 9>"$LOCK_FILE"
if ! flock -n "$LOCK_FD"; then
  echo "[pipeline] another instance is running, exiting" >&2
  exit 0
fi

PY="$ROOT/venv/bin/python"
LOG_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log() { printf '[%s] %s\n' "$LOG_TS" "$*"; }

log "fetch"
"$PY" scripts/fetch_doc.py

log "normalize"
"$PY" scripts/normalize_doc.py --all

SOURCES="$("$PY" -c '
import json
cfg=json.load(open("config/source_docs.json"))
for s in cfg.get("sources",[]):
  if s.get("enabled"): print(s["source_id"])
')"

log "diff"
for sid in $SOURCES; do
  if [ "$(find normalized -type f -name "*.normalized.json" -path "*/$sid/*" 2>/dev/null | wc -l)" -ge 2 ]; then
    "$PY" scripts/diff_snapshots.py --latest "$sid" || true
  fi
done

# Review every delta that doesn't yet have a review (catches up if prior tick failed).
# Non-fatal: a review failure must not block commit/push.
log "review"
for sid in $SOURCES; do
  while IFS= read -r d; do
    ts="$(basename "$d" .delta.json)"
    r="reviews/${ts:0:4}/${ts:5:2}/${ts:8:2}/$sid/${ts}.review.json"
    if [ ! -f "$r" ]; then
      if ! "$PY" scripts/review_agent.py "$d"; then
        log "review failed for $d (non-fatal)"
      fi
    fi
  done < <(find deltas -type f -name "*.delta.json" -path "*/$sid/*" 2>/dev/null | sort)
done

log "render"
"$PY" scripts/render_ledger.py >/dev/null

# Commit + push if anything new is staged. Derived dirs (normalized/, site/) are gitignored.
log "git"
if [ -d .git ]; then
  git add snapshots deltas reviews 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -m "ledger: $LOG_TS" >/dev/null
    log "committed: $(git rev-parse --short HEAD)"
    if git remote get-url origin >/dev/null 2>&1; then
      if git push origin main >/dev/null 2>&1; then
        log "pushed to origin/main"
      else
        log "git push failed (non-fatal)"
      fi
    fi
  else
    log "nothing to commit"
  fi
else
  log "no .git — skipping"
fi
