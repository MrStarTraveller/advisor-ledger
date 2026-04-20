#!/usr/bin/env python3
"""
Experimental: per-delta deduplication via bitdeer Kimi K2.5. For each delta,
ask the model which deleted paragraph is actually a rephrasing of which newly
inserted paragraph. Used by the `deduped.html` experimental view to collapse
noisy edit chains into a single "revised" entry.

This is advisory and separate from review_agent — one run produces one
dedup artifact, even if the underlying diff has zero rephrasings.

Output: dedup/YYYY/MM/DD/<source_id>/<ts>.dedup.json
{
  "source_id", "delta_ts", "reviewed_at_utc", "model",
  "pairs": [{"ghost_hash": "...", "insert_hash": "...", "note": "..."}],
  "finish_reason", "completion_tokens",
  "skipped_reason": null | "no ghosts" | "no inserts" | "too many"
}
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DELTAS_DIR = ROOT / "deltas"
DEDUP_DIR = ROOT / "dedup"
ENV_PATH = ROOT / "secrets" / "review_api.env"
MAX_TOKENS = 8192
REQUEST_TIMEOUT = 180
MAX_PAIR_PRODUCT = 200  # skip dedup when ghost*insert combinations exceed this

SYSTEM_PROMPT = """You identify rephrasings in a single edit to a community Chinese document. Given a list of DELETED paragraphs (ghosts) and a list of INSERTED paragraphs from one edit, find any ghost-insert pairs that are clearly the SAME statement being rephrased — typo fix, wording adjustment, clarification, translation, minor rewrite. Be conservative: only pair when the two say essentially the same thing. Do NOT pair unrelated statements that merely share topic or target.

Return ONLY JSON in this exact shape:
{"pairs": [{"ghost_index": <int>, "insert_index": <int>, "note": "<why, <=40 chars>"}]}

Indices are 0-based positions in the lists below. If nothing matches, return {"pairs": []}. No other text."""


def load_env(path: Path) -> dict:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def collect_ghosts_inserts(delta: dict) -> tuple[list[dict], list[dict]]:
    """Pull all deleted + inserted paragraph objects out of a delta, dedup by content_hash."""
    ghosts: dict[str, dict] = {}
    inserts: dict[str, dict] = {}
    for op in delta["operations"]:
        if op["op"] == "delete":
            for p in op["paragraphs"]:
                ghosts.setdefault(p["content_hash"], p)
        elif op["op"] == "insert":
            for p in op["paragraphs"]:
                inserts.setdefault(p["content_hash"], p)
        elif op["op"] == "replace":
            for p in op["from_paragraphs"]:
                ghosts.setdefault(p["content_hash"], p)
            for p in op["to_paragraphs"]:
                inserts.setdefault(p["content_hash"], p)
    return list(ghosts.values()), list(inserts.values())


def call_chat(url: str, model: str, api_key: str, system: str, user: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "advisor-ledger/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def dedup_delta(delta_path: Path) -> tuple[Path, int, str | None]:
    env = load_env(ENV_PATH)
    delta = json.loads(delta_path.read_text(encoding="utf-8"))
    ghosts, inserts = collect_ghosts_inserts(delta)
    ts = delta["to"]["captured_at_utc"]
    out_path = (
        DEDUP_DIR
        / ts[:4]
        / ts[5:7]
        / ts[8:10]
        / delta["source_id"]
        / f"{ts}.dedup.json"
    )

    skipped_reason: str | None = None
    pairs: list[dict] = []
    finish_reason: str | None = None
    tokens: int | None = None

    if not ghosts:
        skipped_reason = "no ghosts"
    elif not inserts:
        skipped_reason = "no inserts"
    elif len(ghosts) * len(inserts) > MAX_PAIR_PRODUCT:
        skipped_reason = "too many"
    else:
        ghost_text = "\n".join(f"[{i}] {g['text']}" for i, g in enumerate(ghosts))
        insert_text = "\n".join(f"[{i}] {p['text']}" for i, p in enumerate(inserts))
        user_msg = f"DELETED (ghosts):\n{ghost_text}\n\nINSERTED:\n{insert_text}"
        try:
            resp = call_chat(
                env["REVIEW_API_URL"],
                env["REVIEW_API_MODEL"],
                env["REVIEW_API_KEY"],
                SYSTEM_PROMPT,
                user_msg,
            )
            choice = resp["choices"][0]
            finish_reason = choice.get("finish_reason")
            tokens = resp.get("usage", {}).get("completion_tokens")
            content = choice["message"].get("content") or ""
            parsed = extract_json(content)
            if parsed and isinstance(parsed.get("pairs"), list):
                for p in parsed["pairs"]:
                    try:
                        gi, ii = int(p["ghost_index"]), int(p["insert_index"])
                        if 0 <= gi < len(ghosts) and 0 <= ii < len(inserts):
                            pairs.append(
                                {
                                    "ghost_hash": ghosts[gi]["content_hash"],
                                    "insert_hash": inserts[ii]["content_hash"],
                                    "note": (p.get("note") or "")[:80],
                                    "ghost_text": ghosts[gi]["text"],
                                    "insert_text": inserts[ii]["text"],
                                }
                            )
                    except (KeyError, TypeError, ValueError):
                        continue
            else:
                skipped_reason = f"parse_failed (finish={finish_reason}, tok={tokens})"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            skipped_reason = f"transport_error: {e!r}"
        except Exception as e:  # noqa: BLE001
            skipped_reason = f"unexpected: {e!r}"

    out = {
        "source_id": delta["source_id"],
        "delta_ts": ts,
        "reviewed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": env.get("REVIEW_API_MODEL"),
        "n_ghosts": len(ghosts),
        "n_inserts": len(inserts),
        "pairs": pairs,
        "finish_reason": finish_reason,
        "completion_tokens": tokens,
        "skipped_reason": skipped_reason,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path, len(pairs), skipped_reason


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("delta_path", nargs="?")
    ap.add_argument("--latest", metavar="SOURCE_ID")
    args = ap.parse_args()

    if args.latest:
        deltas = sorted(DELTAS_DIR.rglob(f"*/{args.latest}/*.delta.json"))
        if not deltas:
            print(f"no deltas for {args.latest}", file=sys.stderr)
            return 0
        delta_path = deltas[-1]
    elif args.delta_path:
        delta_path = Path(args.delta_path).resolve()
    else:
        ap.error("provide --latest SOURCE_ID or a delta path")

    out, n_pairs, skip = dedup_delta(delta_path)
    tag = f"skipped={skip}" if skip else f"pairs={n_pairs}"
    print(f"[ok] {delta_path.name}: {tag} -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
