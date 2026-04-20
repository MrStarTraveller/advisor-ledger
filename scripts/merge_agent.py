#!/usr/bin/env python3
"""
Poll GitHub Issues (created by utterances from comments on docs/*.html) for
new comments, ask bitdeer Kimi K2.5 whether each comment should be merged
into MIRROR.md, and if so append the proposed patch under the right section,
commit + push, and reply to the comment with "已并入 <sha>".

Idempotent: each processed comment is recorded in `merges/<issue#>-<comment_id>.json`.
Non-fatal: any failure (gh not auth'd, API down, Kimi transient error) is
logged; the comment stays eligible for retry on the next tick.

Auth: `gh` CLI must be logged in as the bot identity. If not, this script
exits cleanly with a warning and processes nothing.

Kimi decision shape:
  {"action": "merge" | "skip" | "needs_clarification",
   "section_heading": "## XXX" | null,      # full heading line incl. # marks
   "patch_markdown": "- ...\n- ...",         # multi-line MD block to append
   "reply": "<polite reply to commenter>",
   "reason": "<internal>"}

Safety rails applied on top of Kimi's output:
  - never modify anything outside MIRROR.md
  - if section_heading doesn't exist, create it at end of file
  - `patch_markdown` is inserted verbatim after the target section
    (before the next same-or-higher heading), so the doc doesn't get
    reshuffled by the agent
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIRROR_PATH = ROOT / "MIRROR.md"
MERGES_DIR = ROOT / "merges"
ENV_PATH = ROOT / "secrets" / "review_api.env"
REPO = "the-hidden-fish/advisor-ledger"

MAX_TOKENS = 8192
REQUEST_TIMEOUT = 180
MIRROR_MAX_CHARS_FOR_PROMPT = 20000  # truncate MIRROR.md tail if huge

SYSTEM_PROMPT = """You moderate comments on a semi-public community document about PhD advisors' practices (学术黑榜). The document's Markdown mirror is `MIRROR.md`, organized by country and school/professor as headings. For each new comment, decide whether to merge it into MIRROR.md.

Merge ONLY if ALL of the following hold:
- It is a concrete observation about a specific advisor (behavior, incident, policy).
- No real names of students, lab members, or other private individuals (professors' full names are OK — they are the subject).
- No phone numbers, addresses, email addresses, or ID numbers.
- Not a pure personal insult without supporting behavior (profanity-only, "stupid/ugly/disgusting" without context).
- Not already substantively covered in the existing document (check for duplicates by meaning, not exact wording).
- Not a thanks / question / emoji / off-topic / meta-discussion about the ledger itself.

If the comment is borderline (interesting but underspecified — e.g. mentions a school but not an advisor), choose `needs_clarification` and ask what's missing.

Pick a `section_heading` that exists in MIRROR.md (copy the whole heading line including its `#` marks). Only if no appropriate heading exists should you propose a new one (e.g. `## <university>` or `### <prof.> <name>`).

`patch_markdown` must be self-contained Markdown to APPEND under that section. Prefer bullet form (`- ...`). Keep it to a few lines. Do NOT include the heading itself in `patch_markdown`.

`reply` is posted publicly under the commenter's comment in the Issue. Be polite, concise (≤2 sentences Chinese), and explain the decision briefly.

Return ONLY JSON:
{"action": "merge"|"skip"|"needs_clarification",
 "section_heading": "## ..." | null,
 "patch_markdown": "..." | null,
 "reply": "...",
 "reason": "..."}
No other text."""


def load_env(path: Path) -> dict:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def gh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh"] + cmd,
        capture_output=True,
        text=True,
        check=check,
    )


def gh_auth_ok() -> bool:
    try:
        r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def list_open_issues() -> list[dict]:
    r = gh(
        [
            "api",
            f"repos/{REPO}/issues",
            "-X", "GET",
            "-f", "state=open",
            "-f", "labels=comments",
            "-f", "per_page=100",
        ],
        check=False,
    )
    if r.returncode != 0:
        print(f"gh list issues failed: {r.stderr.strip()}", file=sys.stderr)
        return []
    try:
        return [i for i in json.loads(r.stdout) if "pull_request" not in i]
    except json.JSONDecodeError:
        return []


def list_comments(issue_number: int) -> list[dict]:
    r = gh(
        [
            "api",
            f"repos/{REPO}/issues/{issue_number}/comments",
            "-X", "GET",
            "-f", "per_page=100",
        ],
        check=False,
    )
    if r.returncode != 0:
        return []
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return []


def gh_login_user() -> str:
    r = subprocess.run(["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True)
    return (r.stdout or "").strip()


def post_reply(issue_number: int, body: str) -> bool:
    # `gh issue comment` is the friendliest interface
    r = subprocess.run(
        ["gh", "issue", "comment", str(issue_number), "--repo", REPO, "--body", body],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"post reply failed: {r.stderr.strip()}", file=sys.stderr)
        return False
    return True


def merge_record_path(issue_number: int, comment_id: int) -> Path:
    return MERGES_DIR / f"{issue_number}-{comment_id}.json"


def save_merge_record(issue_number: int, comment_id: int, data: dict) -> None:
    MERGES_DIR.mkdir(exist_ok=True)
    merge_record_path(issue_number, comment_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def call_kimi(user_msg: str) -> tuple[dict | None, str | None]:
    env = load_env(ENV_PATH)
    body = json.dumps(
        {
            "model": env["REVIEW_API_MODEL"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        env["REVIEW_API_URL"],
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {env['REVIEW_API_KEY']}",
            "User-Agent": "advisor-ledger/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            r = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return None, f"transport_error: {e!r}"
    except Exception as e:  # noqa: BLE001
        return None, f"unexpected: {e!r}"
    content = r["choices"][0]["message"].get("content") or ""
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return None, f"no_json (finish={r['choices'][0].get('finish_reason')})"
    try:
        return json.loads(content[start : end + 1]), None
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e}"


def apply_patch_to_mirror(section_heading: str, patch_markdown: str) -> bool:
    """Append patch_markdown under the given heading. Create the heading at
    end of file if not present. Returns True if MIRROR.md was modified."""
    md = MIRROR_PATH.read_text(encoding="utf-8")
    heading = section_heading.strip()
    if not heading.startswith("#"):
        return False
    # Find heading line exactly
    lines = md.split("\n")
    idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == heading:
            idx = i
            break
    if idx is None:
        # append new heading at end of file
        new_md = md.rstrip() + f"\n\n{heading}\n{patch_markdown.rstrip()}\n"
        MIRROR_PATH.write_text(new_md, encoding="utf-8")
        return True
    # find the end of this section: next line whose heading level is <= current
    level = len(heading) - len(heading.lstrip("#"))
    end_idx = len(lines)
    for j in range(idx + 1, len(lines)):
        ln = lines[j]
        m = re.match(r"^(#{1,6})\s", ln)
        if m and len(m.group(1)) <= level:
            end_idx = j
            break
    # insertion: after section content, but before the next heading (end_idx)
    insert_point = end_idx
    # trim trailing blanks of the section
    while insert_point > idx + 1 and lines[insert_point - 1].strip() == "":
        insert_point -= 1
    block = patch_markdown.rstrip().split("\n")
    new_lines = lines[:insert_point] + [""] + block + [""] + lines[insert_point:]
    MIRROR_PATH.write_text("\n".join(new_lines), encoding="utf-8")
    return True


def git_commit_and_push(message: str) -> str | None:
    """Commit MIRROR.md + merges/ and push. Returns short sha on success."""
    try:
        subprocess.run(
            ["git", "add", "MIRROR.md", "merges"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        # nothing to commit?
        r = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ROOT,
        )
        if r.returncode == 0:
            return None
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=ROOT,
            capture_output=True,
            check=False,  # non-fatal; next tick retries
        )
        return sha
    except subprocess.CalledProcessError as e:
        print(f"git op failed: {e.stderr.decode() if e.stderr else e}", file=sys.stderr)
        return None


def process_comment(issue: dict, comment: dict, bot_login: str, mirror_md: str) -> dict:
    issue_number = issue["number"]
    comment_id = comment["id"]
    record_path = merge_record_path(issue_number, comment_id)
    if record_path.exists():
        return {"action": "already_processed"}

    # Skip our own replies (to avoid loops), but NOT comments the Worker relayed
    # on behalf of anonymous users — those are posted with the same PAT but are
    # user content. The Worker prefixes them with "_(匿名"; our own replies are
    # prefixed with "(bot) ". Use that as the discriminator.
    body = (comment.get("body") or "").lstrip()
    is_own_reply = comment["user"]["login"] == bot_login and body.startswith("(bot)")
    if is_own_reply:
        save_merge_record(
            issue_number,
            comment_id,
            {
                "action": "self_skip",
                "issue_number": issue_number,
                "comment_id": comment_id,
                "processed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        return {"action": "self_skip"}

    mirror_excerpt = mirror_md
    if len(mirror_excerpt) > MIRROR_MAX_CHARS_FOR_PROMPT:
        mirror_excerpt = mirror_excerpt[:MIRROR_MAX_CHARS_FOR_PROMPT] + "\n\n[... TRUNCATED ...]\n"

    user_msg = (
        f"=== CURRENT MIRROR.md (possibly truncated) ===\n"
        f"{mirror_excerpt}\n"
        f"=== NEW COMMENT (issue #{issue_number}, by @{comment['user']['login']}) ===\n"
        f"{comment['body']}"
    )
    decision, err = call_kimi(user_msg)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if err or decision is None:
        # don't persist — let it retry next tick
        print(f"kimi failed for issue #{issue_number} comment {comment_id}: {err}", file=sys.stderr)
        return {"action": "kimi_failed", "error": err}

    action = decision.get("action")
    reply = (decision.get("reply") or "").strip()
    record: dict = {
        "issue_number": issue_number,
        "comment_id": comment_id,
        "comment_author": comment["user"]["login"],
        "processed_at_utc": now,
        "kimi_decision": decision,
    }

    if action == "merge":
        heading = (decision.get("section_heading") or "").strip()
        patch = (decision.get("patch_markdown") or "").strip()
        if not heading or not patch:
            record["action"] = "malformed_merge"
            save_merge_record(issue_number, comment_id, record)
            post_reply(issue_number, f"(bot) 未并入:Kimi 返回的结构缺失 section/patch。\n<!-- decision: {json.dumps(decision, ensure_ascii=False)} -->")
            return record
        if not apply_patch_to_mirror(heading, patch):
            record["action"] = "apply_failed"
            save_merge_record(issue_number, comment_id, record)
            post_reply(issue_number, "(bot) 未并入:patch 落盘失败。")
            return record
        sha = git_commit_and_push(
            f"merge: issue #{issue_number} comment {comment_id}\n\nvia Kimi; section={heading}"
        )
        record["action"] = "merged"
        record["sha"] = sha
        save_merge_record(issue_number, comment_id, record)
        msg = f"(bot) 已并入 [`{sha}`](https://github.com/{REPO}/commit/{sha})。\n\n> {reply}" if sha else f"(bot) 已并入(但 push 暂时失败)。\n\n> {reply}"
        post_reply(issue_number, msg)
        return record

    if action == "skip":
        record["action"] = "skipped"
        save_merge_record(issue_number, comment_id, record)
        post_reply(issue_number, f"(bot) 未并入:{reply}")
        return record

    if action == "needs_clarification":
        record["action"] = "clarification_requested"
        save_merge_record(issue_number, comment_id, record)
        post_reply(issue_number, f"(bot) 需要补充信息:{reply}")
        return record

    record["action"] = "unknown"
    save_merge_record(issue_number, comment_id, record)
    post_reply(issue_number, f"(bot) 审查返回了未知的 action={action!r}。")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print decisions, don't commit or reply")
    args = ap.parse_args()

    if not gh_auth_ok():
        print("gh not authenticated; skipping merge pass", file=sys.stderr)
        return 0
    if not MIRROR_PATH.exists():
        print("MIRROR.md missing; run bootstrap_mirror.py first", file=sys.stderr)
        return 0

    bot_login = gh_login_user()
    if not bot_login:
        print("could not resolve gh login user", file=sys.stderr)
        return 0

    issues = list_open_issues()
    mirror_md = MIRROR_PATH.read_text(encoding="utf-8")
    processed = 0
    merged = 0
    for issue in issues:
        comments = list_comments(issue["number"])
        for c in comments:
            if merge_record_path(issue["number"], c["id"]).exists():
                continue
            if args.dry_run:
                print(f"[dry-run] issue #{issue['number']} comment {c['id']} by @{c['user']['login']}")
                continue
            result = process_comment(issue, c, bot_login, mirror_md)
            processed += 1
            if result.get("action") == "merged":
                merged += 1
                # reload in case of subsequent comments in same pass
                mirror_md = MIRROR_PATH.read_text(encoding="utf-8")
    print(f"[merge_agent] processed={processed} merged={merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
