#!/usr/bin/env python3
"""
Render the ledger for each source as a single HTML page:
- current (latest snapshot) paragraphs shown in order
- paragraphs that were ever deleted are preserved in-place with strike-through
  and a "deleted at <ts>" badge
- paragraphs added after the first snapshot are highlighted with "added at <ts>"

Output: site/<source_id>.html  (also site/index.html pointing at the first source)

Anchoring: each delete/replace-from op is recorded with the content_hash of the
paragraph immediately preceding it in the pre-delete snapshot. At render time,
the ghost is emitted right after the first live occurrence of that anchor hash.
When the anchor was itself later deleted, the ghost lands in an "orphaned
deletions" section at the bottom (still preserved, just not in-place).
"""

from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "normalized"
DELTAS_DIR = ROOT / "deltas"
REVIEWS_DIR = ROOT / "reviews"
DEDUP_DIR = ROOT / "dedup"
SITE_DIR = ROOT / "docs"
RELAY_CONFIG_PATH = ROOT / "config" / "relay.json"

BLANK_HASH = "0" * 16


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def list_normalized(source_id: str) -> list[Path]:
    return sorted(NORMALIZED_DIR.rglob(f"*/{source_id}/*.normalized.json"))


def list_deltas(source_id: str) -> list[Path]:
    return sorted(DELTAS_DIR.rglob(f"*/{source_id}/*.delta.json"))


def list_reviews(source_id: str) -> list[Path]:
    return sorted(REVIEWS_DIR.rglob(f"*/{source_id}/*.review.json"))


def list_dedups(source_id: str) -> list[Path]:
    return sorted(DEDUP_DIR.rglob(f"*/{source_id}/*.dedup.json"))


def dedup_index(source_id: str, deltas: list[dict]):
    """Build two maps from dedup artifacts + positional replace-op pairs:
      - insert_to_ghosts: insert_hash -> [(ghost_text, ghost_style, delta_ts, note, via), ...]
      - ghost_keys_consumed: set of (delta_ts, ghost_hash) that are paired
    """
    insert_to_ghosts: dict[str, list[dict]] = defaultdict(list)
    ghost_keys_consumed: set[tuple[str, str]] = set()

    # LLM-judged pairs
    for p in list_dedups(source_id):
        d = load_json(p)
        ts = d.get("delta_ts")
        for pair in d.get("pairs", []) or []:
            gh = pair.get("ghost_hash")
            ih = pair.get("insert_hash")
            if not gh or not ih:
                continue
            insert_to_ghosts[ih].append(
                {
                    "ghost_hash": gh,
                    "ghost_text": pair.get("ghost_text", ""),
                    "delta_ts": ts,
                    "note": pair.get("note", ""),
                    "via": "llm",
                }
            )
            ghost_keys_consumed.add((ts, gh))

    # Positional replace-op pairs (deterministic; free dedup signal)
    for d in deltas:
        ts = d["to"]["captured_at_utc"]
        for op in d["operations"]:
            if op["op"] != "replace":
                continue
            froms = op.get("from_paragraphs", [])
            tos = op.get("to_paragraphs", [])
            # pair 1-to-1 up to the shorter length
            for fp, tp in zip(froms, tos):
                if fp["content_hash"] == BLANK_HASH or tp["content_hash"] == BLANK_HASH:
                    continue
                if (ts, fp["content_hash"]) in ghost_keys_consumed:
                    continue  # already paired by LLM
                insert_to_ghosts[tp["content_hash"]].append(
                    {
                        "ghost_hash": fp["content_hash"],
                        "ghost_text": fp["text"],
                        "delta_ts": ts,
                        "note": "positional replace",
                        "via": "positional",
                    }
                )
                ghost_keys_consumed.add((ts, fp["content_hash"]))

    return insert_to_ghosts, ghost_keys_consumed


def suspicious_by_ts(source_id: str) -> dict[str, list[dict]]:
    """Return {delta_ts: [suspicious_deletion concerns]} for the source."""
    out: dict[str, list[dict]] = defaultdict(list)
    for p in list_reviews(source_id):
        r = load_json(p)
        for c in r.get("concerns", []) or []:
            if c.get("type") == "suspicious_deletion":
                out[r["delta_ts"]].append(c)
    return out


def concern_matches_text(concern: dict, text: str) -> bool:
    """Fuzzy match: share a contiguous 6-char window, or one is substring of the other."""
    excerpt = (concern.get("excerpt") or "").replace("...", "").replace("…", "").strip()
    if not excerpt or not text:
        return False
    if excerpt in text or text in excerpt:
        return True
    # window-based fuzzy match to tolerate paraphrase/truncation
    for s, l in ((excerpt, text), (text, excerpt)):
        if len(s) < 6:
            continue
        for i in range(len(s) - 5):
            if s[i : i + 6] in l:
                return True
    return False


def attach_suspicious(
    ghosts_head: list[dict],
    ghosts_by_anchor: dict[str, list[dict]],
    by_ts: dict[str, list[dict]],
) -> None:
    """Mutate ghost dicts in-place: add 'suspicious_concerns' where matched.
    Unmatched concerns fall back to every ghost in the same delta (conservative)."""
    all_ghosts = list(ghosts_head) + [g for lst in ghosts_by_anchor.values() for g in lst]
    by_ts_ghosts: dict[str, list[dict]] = defaultdict(list)
    for g in all_ghosts:
        by_ts_ghosts[g["deleted_at"]].append(g)
    for ts, concerns in by_ts.items():
        ghosts_here = by_ts_ghosts.get(ts, [])
        if not ghosts_here:
            continue
        unmatched: list[dict] = []
        for c in concerns:
            matched = False
            for g in ghosts_here:
                if concern_matches_text(c, g["text"]):
                    g.setdefault("suspicious_concerns", []).append(c)
                    matched = True
            if not matched:
                unmatched.append(c)
        if unmatched:
            for g in ghosts_here:
                # only add fallback concerns where no specific match already exists
                if "suspicious_concerns" not in g:
                    g["suspicious_concerns"] = list(unmatched)


def discover_source_ids() -> list[str]:
    return sorted({p.parent.name for p in NORMALIZED_DIR.rglob("*.normalized.json")})


def load_relay_config() -> dict | None:
    if not RELAY_CONFIG_PATH.exists():
        return None
    try:
        cfg = json.loads(RELAY_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not cfg.get("worker_url") or not cfg.get("turnstile_site_key"):
        return None
    if "YOUR_SUBDOMAIN" in cfg["worker_url"] or "AAAAAAAAAAAAAAAAAAAAAA" in cfg["turnstile_site_key"]:
        return None  # still on example values
    return cfg


def anon_form_html(cfg: dict | None) -> str:
    if not cfg:
        return ""  # quietly omit the anon section until relay is configured
    worker_url = html.escape(cfg["worker_url"])
    site_key = html.escape(cfg["turnstile_site_key"])
    return f"""
 <div class="anon-form">
  <h3>匿名发言(无需登录)</h3>
  <form id="anon-form">
   <textarea name="body" rows="4" required minlength="3" maxlength="4000" placeholder="留下你想说的……评论会以匿名身份进入上方的 Issue 线程"></textarea>
   <div class="cf-turnstile" data-sitekey="{site_key}" data-theme="auto"></div>
   <button type="submit">发送</button>
   <div id="anon-status" role="status" aria-live="polite"></div>
  </form>
  <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
  <script>
  document.getElementById('anon-form').addEventListener('submit', async (e) => {{
    e.preventDefault();
    const f = e.target;
    const body = f.body.value.trim();
    const token = f.querySelector('[name=cf-turnstile-response]')?.value;
    const status = document.getElementById('anon-status');
    const btn = f.querySelector('button');
    if (!body) {{ status.textContent = '请写点什么'; return; }}
    if (!token) {{ status.textContent = '请先完成人机验证'; return; }}
    btn.disabled = true; status.textContent = '提交中...';
    try {{
      const r = await fetch({json.dumps(cfg["worker_url"])}, {{
        method: 'POST',
        headers: {{'content-type': 'application/json'}},
        body: JSON.stringify({{pathname: location.pathname, comment: body, token}}),
      }});
      if (r.ok) {{
        const d = await r.json();
        status.innerHTML = '已提交 (<a href="' + d.url + '" target="_blank">查看</a>)。Kimi 约 2 分钟内扫描。';
        f.body.value = '';
        if (window.turnstile) window.turnstile.reset();
      }} else {{
        status.textContent = '提交失败 (' + r.status + '): ' + (await r.text());
      }}
    }} catch (err) {{ status.textContent = '网络错误: ' + err; }}
    btn.disabled = false;
  }});
  </script>
 </div>
"""


def is_mass_deletion(d: dict) -> bool:
    """Honor the delta's own flag; for older deltas missing it, recompute."""
    s = d.get("summary", {}) or {}
    v = s.get("mass_deletion_suspected")
    if v is not None:
        return bool(v)
    deleted = s.get("deleted_paragraphs", 0)
    from_pc = max(d.get("from", {}).get("paragraph_count", 0), 1)
    return deleted >= 10 and (deleted / from_pc) >= 0.15


def build_ghosts(norms: list[dict], deltas: list[dict]):
    """Return (ghosts_head, ghosts_by_anchor) for this source."""
    norm_by_ts = {n["captured_at_utc"]: n for n in norms}
    ghosts_head: list[dict] = []
    ghosts_by_anchor: dict[str, list[dict]] = defaultdict(list)

    for d in deltas:
        from_ts = d["from"]["captured_at_utc"]
        if from_ts not in norm_by_ts:
            continue
        from_paras = norm_by_ts[from_ts]["paragraphs"]
        to_ts = d["to"]["captured_at_utc"]
        mass_del = is_mass_deletion(d)
        for op in d["operations"]:
            if op["op"] == "delete":
                pos = op["at_from"]
                ghost_list = op["paragraphs"]
            elif op["op"] == "replace":
                pos = op["at_from"]
                ghost_list = op["from_paragraphs"]
            else:
                continue
            anchor = from_paras[pos - 1]["content_hash"] if pos > 0 else None
            for g in ghost_list:
                if g["content_hash"] == BLANK_HASH:
                    continue  # skip blank-line ghosts (too noisy)
                rec = {**g, "deleted_at": to_ts, "mass_deletion": mass_del}
                if anchor is None:
                    ghosts_head.append(rec)
                else:
                    ghosts_by_anchor[anchor].append(rec)
    return ghosts_head, ghosts_by_anchor


def first_seen_map(norms: list[dict]) -> dict[str, str]:
    fs: dict[str, str] = {}
    for n in norms:
        ts = n["captured_at_utc"]
        for p in n["paragraphs"]:
            h = p["content_hash"]
            if h == BLANK_HASH:
                continue
            if h not in fs:
                fs[h] = ts
    return fs


def esc(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")


def render_live(p: dict, added_at: str | None, revisions: list[dict] | None = None) -> str:
    revisions = revisions or []
    classes = ["p", "live"]
    if added_at:
        classes.append("added")
    if revisions:
        classes.append("revised")
    badges: list[str] = []
    if added_at:
        badges.append(f'<span class="badge added">+ {html.escape(added_at)}</span>')
    if revisions:
        badges.append(f'<span class="badge revised">✎ 改写 ({len(revisions)})</span>')
    style = html.escape(p.get("style", "NORMAL_TEXT"))
    text = esc(p["text"]) or "&nbsp;"
    detail_html = ""
    if revisions:
        items = "".join(
            f'<li><span class="ts">{html.escape(r.get("delta_ts",""))}</span> '
            f'<span class="via">({html.escape(r.get("via",""))})</span><br>'
            f'<span class="prev">{esc(r.get("ghost_text",""))}</span>'
            f'{"<br><em>" + html.escape(r.get("note","")) + "</em>" if r.get("note") else ""}'
            f'</li>'
            for r in revisions
        )
        detail_html = (
            f'<details class="rev-detail"><summary>前版本 ({len(revisions)})</summary>'
            f'<ul>{items}</ul></details>'
        )
    return (
        f'<div class="{" ".join(classes)}" data-style="{style}">'
        f'{"".join(badges)}<div class="text">{text}</div>{detail_html}</div>'
    )


def render_ghost(g: dict) -> str:
    style = html.escape(g.get("style", "NORMAL_TEXT"))
    text = esc(g["text"])
    concerns = g.get("suspicious_concerns") or []
    mass = bool(g.get("mass_deletion"))
    classes = ["p", "ghost"]
    if mass:
        classes.append("mass-deletion")
    if concerns:
        classes.append("suspicious")
    badges = [f'<span class="badge deleted">− {html.escape(g["deleted_at"])}</span>']
    if mass:
        badges.append('<span class="badge mass">⛔ 批量删除</span>')
    if concerns:
        badges.append('<span class="badge flag">⚠ 可疑删除</span>')
    detail_html = ""
    if concerns:
        items = "".join(
            f'<li>{html.escape(c.get("detail", ""))}</li>' for c in concerns
        )
        detail_html = (
            f'<details class="flag-detail"><summary>AI 审查意见 ({len(concerns)})</summary>'
            f'<ul>{items}</ul></details>'
        )
    return (
        f'<div class="{" ".join(classes)}" data-style="{style}">'
        f'{"".join(badges)}'
        f'<div class="text">{text}</div>'
        f'{detail_html}'
        f'</div>'
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:920px;margin:2em auto;padding:0 1em;color:#222;line-height:1.5;}}
 h1{{font-size:1.5em;margin:.2em 0;}}
 .meta{{color:#666;font-size:.85em;margin-bottom:1.5em;border-bottom:1px solid #ddd;padding-bottom:1em;}}
 .legend span{{margin-right:1em;}}
 .p{{margin:.25em 0;padding:.3em .5em;border-left:3px solid transparent;}}
 .p .text{{white-space:pre-wrap;word-wrap:break-word;}}
 .p[data-style=HEADING_1] .text{{font-size:1.3em;font-weight:600;}}
 .p[data-style=HEADING_2] .text{{font-size:1.15em;font-weight:600;}}
 .p[data-style=HEADING_3] .text{{font-size:1.05em;font-weight:600;}}
 .p.ghost{{background:#fff4f4;border-left-color:#c33;}}
 .p.ghost .text{{color:#a33;text-decoration:line-through;}}
 .p.ghost.suspicious{{background:#fff6d6;border-left-color:#d4a017;box-shadow:inset 3px 0 0 #d4a017, 0 0 0 1px #d4a017;}}
 .p.ghost.suspicious .text{{color:#7a5c00;}}
 .p.ghost.mass-deletion{{background:#ffe5e5;border-left-color:#a00;box-shadow:inset 4px 0 0 #a00, 0 0 0 1px #a00;}}
 .p.ghost.mass-deletion .text{{color:#800;font-weight:500;}}
 .p.ghost.mass-deletion.suspicious{{background:#ffd9d9;}}
 .p.live.revised{{background:#f5f1ff;border-left-color:#7a5af5;}}
 .badge.revised{{background:#7a5af5;}}
 .rev-detail{{margin-top:.3em;font-size:.8em;color:#555;}}
 .rev-detail summary{{cursor:pointer;color:#5a3ad5;}}
 .rev-detail ul{{list-style:none;margin:.3em 0;padding:0;border-left:2px solid #cbbcf5;padding-left:.7em;}}
 .rev-detail li{{margin:.35em 0;}}
 .rev-detail .ts{{font-family:ui-monospace,monospace;font-size:.85em;color:#888;}}
 .rev-detail .via{{color:#aaa;font-size:.8em;}}
 .rev-detail .prev{{color:#a33;text-decoration:line-through;}}
 .view-nav{{font-size:.85em;margin:.5em 0;}}
 .view-nav a{{color:#555;text-decoration:none;padding:.2em .6em;border-radius:3px;background:#eee;margin-right:.3em;}}
 .view-nav a.current{{background:#333;color:#fff;}}
 .view-nav .exp{{color:#7a5af5;font-weight:600;font-size:.9em;}}
 .comments{{margin-top:3em;padding-top:1.5em;border-top:2px solid #ddd;}}
 .comments h2{{font-size:1.1em;color:#333;margin:.2em 0;}}
 .comments-hint{{font-size:.85em;color:#666;margin-bottom:1em;line-height:1.6;}}
 .comments-hint a{{color:#0366d6;}}
 .anon-form{{margin-top:2em;padding:1em;background:#f8fafb;border:1px solid #e1e4e8;border-radius:5px;}}
 .anon-form h3{{margin:.1em 0 .5em;font-size:1em;color:#333;}}
 .anon-form textarea{{width:100%;font:inherit;padding:.6em;border:1px solid #ccc;border-radius:4px;resize:vertical;box-sizing:border-box;}}
 .anon-form .cf-turnstile{{margin:.8em 0;}}
 .anon-form button{{background:#2a8;color:#fff;border:0;padding:.55em 1.3em;border-radius:4px;font:inherit;cursor:pointer;}}
 .anon-form button:disabled{{background:#aaa;cursor:wait;}}
 .anon-form #anon-status{{margin-top:.5em;font-size:.85em;color:#555;}}
 .anon-form #anon-status a{{color:#0366d6;}}
 .badge.mass{{background:#a00;}}
 .attack-banner{{background:#a00;color:#fff;padding:.7em 1em;border-radius:4px;margin:1em 0 1.5em;font-size:.95em;}}
 .attack-banner strong{{font-weight:600;}}
 .attack-banner code{{background:rgba(255,255,255,.2);padding:.05em .35em;border-radius:2px;}}
 .p.added{{background:#f0fff4;border-left-color:#2a8;}}
 .badge{{display:inline-block;font-size:.7em;padding:.05em .45em;border-radius:3px;margin-right:.6em;font-family:ui-monospace,monospace;vertical-align:middle;text-decoration:none;color:#fff;}}
 .badge.added{{background:#2a8;}}
 .badge.deleted{{background:#c33;}}
 .badge.flag{{background:#d4a017;}}
 .flag-detail{{margin-top:.3em;font-size:.8em;color:#555;text-decoration:none;}}
 .flag-detail summary{{cursor:pointer;color:#9a7a00;}}
 .flag-detail ul{{margin:.3em 0 .2em 1.4em;padding:0;}}
 .flag-detail li{{margin:.15em 0;}}
 h2.section{{margin-top:3em;font-size:1.1em;color:#666;border-top:1px dashed #ccc;padding-top:1em;}}
</style></head><body>
<h1>{title}</h1>
<div class="view-nav">
 <a href="index.html"{faithful_current}>原始视图</a>
 <a href="deduped.html"{deduped_current}>去重视图 <span class="exp">实验性</span></a>
</div>
<div class="meta">
 source: <code>{source_id}</code> · snapshots: {n_snapshots} · range: {earliest_ts} → {latest_ts}<br>
 live paragraphs: {n_live} · deleted (preserved): {n_ghosts} · added since start: {n_added} · 可疑删除: {n_suspicious} · 批量删除: {n_mass}<br>
 <span class="legend"><span class="badge added">+ ts</span>added after first snapshot</span>
 <span class="legend"><span class="badge deleted">− ts</span>deleted (kept with strike-through)</span>
 <span class="legend"><span class="badge flag">⚠</span>AI 标注为可疑删除</span>
 <span class="legend"><span class="badge mass">⛔</span>单次删除超 15% / ≥10 段(疑似批量删除)</span>
</div>
<main>
{body}
</main>
<section class="comments">
 <h2>评论区</h2>
 <p class="comments-hint">两种发言方式都进入同一个 GitHub Issue 线程,Kimi 定期扫描、符合规范的会自动并入 <a href="https://github.com/the-hidden-fish/advisor-ledger/blob/main/MIRROR.md">MIRROR.md</a>,并在评论下回复 "已并入 &lt;commit-sha&gt;"。<br>如果原 Google Doc 被下架,MIRROR.md 就是下一个源。</p>
{anon_form_html}
 <h3 style="margin-top:2em;font-size:1em;color:#333;">登录 GitHub 发言</h3>
 <script src="https://utteranc.es/client.js"
         repo="the-hidden-fish/advisor-ledger"
         issue-term="pathname"
         label="comments"
         theme="preferred-color-scheme"
         crossorigin="anonymous"
         async>
 </script>
</section>
</body></html>
"""


def render_source(source_id: str, mode: str = "faithful") -> str | None:
    norm_paths = list_normalized(source_id)
    if not norm_paths:
        return None
    norms = [load_json(p) for p in norm_paths]
    deltas = [load_json(p) for p in list_deltas(source_id)]

    latest = norms[-1]
    earliest_ts = norms[0]["captured_at_utc"]
    fs = first_seen_map(norms)
    ghosts_head, ghosts_by_anchor = build_ghosts(norms, deltas)
    attach_suspicious(ghosts_head, ghosts_by_anchor, suspicious_by_ts(source_id))

    # For the deduped view: fold paired ghosts into their replacement live paragraphs.
    insert_to_ghosts, ghost_consumed_keys = (
        dedup_index(source_id, deltas) if mode == "deduped" else ({}, set())
    )

    mass_events = [
        {
            "ts": d["to"]["captured_at_utc"],
            "deleted": d["summary"].get("deleted_paragraphs", 0),
            "ratio": d["summary"].get("deletion_ratio"),
        }
        for d in deltas
        if is_mass_deletion(d)
    ]

    parts: list[str] = []
    if mass_events:
        latest_evt = mass_events[-1]
        ratio_txt = (
            f"{latest_evt['ratio']*100:.0f}%" if latest_evt.get("ratio") is not None else "?"
        )
        parts.append(
            f'<div class="attack-banner">'
            f'⛔ <strong>检测到 {len(mass_events)} 次疑似批量删除事件</strong> '
            f'(最近一次 <code>{html.escape(latest_evt["ts"])}</code>,'
            f'删除 {latest_evt["deleted"]} 段 ≈ {ratio_txt})。'
            f'所有被删内容已在下方以红色高亮保留,不会丢失。'
            f'</div>'
        )
    n_added = 0

    def ghost_suppressed(g: dict) -> bool:
        return mode == "deduped" and (g["deleted_at"], g["content_hash"]) in ghost_consumed_keys

    for g in ghosts_head:
        if ghost_suppressed(g):
            continue
        parts.append(render_ghost(g))

    emitted_anchor: set[str] = set()
    for p in latest["paragraphs"]:
        h = p["content_hash"]
        added_at = None
        if h != BLANK_HASH and fs.get(h) and fs[h] != earliest_ts:
            added_at = fs[h]
            n_added += 1
        revisions = insert_to_ghosts.get(h, []) if mode == "deduped" else []
        parts.append(render_live(p, added_at, revisions))
        if h in ghosts_by_anchor and h not in emitted_anchor:
            emitted_anchor.add(h)
            for g in ghosts_by_anchor[h]:
                if ghost_suppressed(g):
                    continue
                parts.append(render_ghost(g))

    orphaned = [
        g
        for h, lst in ghosts_by_anchor.items()
        if h not in emitted_anchor
        for g in lst
        if not ghost_suppressed(g)
    ]
    if orphaned:
        parts.append('<h2 class="section">Orphaned deletions (anchor also gone)</h2>')
        for g in orphaned:
            parts.append(render_ghost(g))

    all_ghosts = ghosts_head + [x for lst in ghosts_by_anchor.values() for x in lst]
    n_ghosts = len(all_ghosts)
    n_suspicious = sum(1 for g in all_ghosts if g.get("suspicious_concerns"))
    n_mass = sum(1 for g in all_ghosts if g.get("mass_deletion"))
    return HTML_TEMPLATE.format(
        title=html.escape(latest.get("title") or source_id),
        source_id=html.escape(source_id),
        n_snapshots=len(norms),
        earliest_ts=html.escape(earliest_ts),
        latest_ts=html.escape(latest["captured_at_utc"]),
        n_live=len(latest["paragraphs"]),
        n_ghosts=n_ghosts,
        n_added=n_added,
        n_suspicious=n_suspicious,
        n_mass=n_mass,
        faithful_current=' class="current"' if mode == "faithful" else "",
        deduped_current=' class="current"' if mode == "deduped" else "",
        body="\n".join(parts),
        anon_form_html=anon_form_html(load_relay_config()),
    )


def main() -> int:
    source_ids = discover_source_ids()
    if not source_ids:
        print("no normalized snapshots yet", file=sys.stderr)
        return 1
    SITE_DIR.mkdir(exist_ok=True)
    first = source_ids[0]
    for sid in source_ids:
        for mode, fname in (("faithful", f"{sid}.html"), ("deduped", f"{sid}.deduped.html")):
            out_html = render_source(sid, mode=mode)
            if out_html is None:
                continue
            (SITE_DIR / fname).write_text(out_html, encoding="utf-8")
            print(f"[ok] {mode} -> {(SITE_DIR / fname).relative_to(ROOT)}")
    # index.html and deduped.html for the first source (what Pages serves at /)
    (SITE_DIR / "index.html").write_text(
        (SITE_DIR / f"{first}.html").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (SITE_DIR / "deduped.html").write_text(
        (SITE_DIR / f"{first}.deduped.html").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"[ok] -> docs/index.html, docs/deduped.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
