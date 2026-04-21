#!/usr/bin/env python3
"""Render a Google Docs API `documents.get` JSON as a fully self-contained HTML
page that preserves every visible style: colors, fonts, weights, sizes, bold /
italic / underline / strikethrough, text & paragraph shading, alignment, line
spacing, space-above / space-below, indentation, headings, bullet/numbered
lists with nesting, rich-link chips, and embedded images.

Usage:
    python3 scripts/render_gdoc_faithful.py <input.json> <output.html>
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# --- small helpers --------------------------------------------------------


def pt_of(dim: dict | None) -> float:
    if not dim:
        return 0.0
    return float(dim.get("magnitude", 0) or 0)


def rgb_css(color_holder: dict | None) -> str | None:
    """Accepts either {"color":{"rgbColor":{...}}} or {"rgbColor":{...}}.
    Returns a css color string, or None if the color is absent / fully-default.
    Note: an rgbColor with no channels means pure black, which is a meaningful
    explicit color (e.g. text foreground), so we still emit rgb(0,0,0)."""
    if not color_holder:
        return None
    inner = color_holder.get("color", color_holder)
    rgb = inner.get("rgbColor") if inner else None
    if rgb is None:
        return None
    r = round(float(rgb.get("red", 0) or 0) * 255)
    g = round(float(rgb.get("green", 0) or 0) * 255)
    b = round(float(rgb.get("blue", 0) or 0) * 255)
    return f"rgb({r},{g},{b})"


def join_style(parts: list[str]) -> str:
    return ";".join(p for p in parts if p)


# --- text-run rendering ---------------------------------------------------


def text_style_css(ts: dict) -> str:
    p: list[str] = []
    if ts.get("bold"):
        p.append("font-weight:bold")
    if ts.get("italic"):
        p.append("font-style:italic")
    decos = []
    if ts.get("underline"):
        decos.append("underline")
    if ts.get("strikethrough"):
        decos.append("line-through")
    if decos:
        p.append("text-decoration:" + " ".join(decos))
    if ts.get("smallCaps"):
        p.append("font-variant:small-caps")
    fg = rgb_css(ts.get("foregroundColor"))
    if fg is not None:
        p.append(f"color:{fg}")
    bg = rgb_css(ts.get("backgroundColor"))
    if bg is not None:
        p.append(f"background-color:{bg}")
    fs = ts.get("fontSize")
    if fs and "magnitude" in fs:
        p.append(f"font-size:{fs['magnitude']}pt")
    wff = ts.get("weightedFontFamily")
    if wff:
        fam = wff.get("fontFamily")
        if fam:
            p.append(f"font-family:'{fam}',sans-serif")
        w = wff.get("weight")
        # weightedFontFamily.weight is authoritative unless bold:true already set it
        if w and not ts.get("bold"):
            p.append(f"font-weight:{w}")
    bo = ts.get("baselineOffset")
    if bo == "SUBSCRIPT":
        p.append("vertical-align:sub")
        p.append("font-size:75%")
    elif bo == "SUPERSCRIPT":
        p.append("vertical-align:super")
        p.append("font-size:75%")
    return join_style(p)


def link_href(link: dict) -> str:
    if url := link.get("url"):
        return url
    if hid := link.get("headingId"):
        return f"#{hid}"
    if bid := link.get("bookmarkId"):
        return f"#{bid}"
    if tid := link.get("tabId"):
        return f"#tab-{tid}"
    return "#"


def render_text_content(content: str, style_css: str, link: dict | None) -> str:
    if not content:
        return ""
    # Google Docs terminates every paragraph with a trailing "\n" in the text run.
    # Strip one trailing newline (the paragraph break is already structural);
    # preserve any internal newlines as <br>.
    body = content[:-1] if content.endswith("\n") else content
    if body == "":
        return ""
    escaped = html.escape(body).replace("\n", "<br>")
    attrs = f' style="{style_css}"' if style_css else ""
    if link:
        href = html.escape(link_href(link), quote=True)
        return f'<a href="{href}"{attrs}>{escaped}</a>'
    if style_css:
        return f"<span{attrs}>{escaped}</span>"
    return escaped


def render_text_run(tr: dict) -> str:
    ts = tr.get("textStyle", {}) or {}
    return render_text_content(
        tr.get("content", ""),
        text_style_css(ts),
        ts.get("link"),
    )


def render_inline_object(el: dict, inline_objects: dict) -> str:
    oid = el["inlineObjectElement"].get("inlineObjectId")
    obj = inline_objects.get(oid, {})
    emb = obj.get("inlineObjectProperties", {}).get("embeddedObject", {}) or {}
    img = emb.get("imageProperties") or {}
    uri = img.get("contentUri", "")
    if not uri:
        return ""
    size = emb.get("size") or {}
    w = pt_of(size.get("width"))
    h = pt_of(size.get("height"))
    m_t = pt_of(emb.get("marginTop"))
    m_b = pt_of(emb.get("marginBottom"))
    m_l = pt_of(emb.get("marginLeft"))
    m_r = pt_of(emb.get("marginRight"))
    style_parts = []
    if w:
        style_parts.append(f"width:{w}pt")
    if h:
        style_parts.append(f"height:{h}pt")
    if any((m_t, m_b, m_l, m_r)):
        style_parts.append(f"margin:{m_t}pt {m_r}pt {m_b}pt {m_l}pt")
    style_parts.append("max-width:100%")
    alt = html.escape(emb.get("title") or emb.get("description") or "", quote=True)
    return (
        f'<img src="{html.escape(uri, quote=True)}" alt="{alt}" '
        f'style="{join_style(style_parts)}" loading="lazy" referrerpolicy="no-referrer">'
    )


def render_rich_link(el: dict) -> str:
    rl = el["richLink"]
    props = rl.get("richLinkProperties", {}) or {}
    uri = props.get("uri") or "#"
    title = props.get("title") or uri
    ts_css = text_style_css(rl.get("textStyle", {}) or {})
    attrs = f' style="{ts_css}"' if ts_css else ""
    return (
        f'<a class="rich-link" href="{html.escape(uri, quote=True)}"{attrs}>'
        f"{html.escape(title)}</a>"
    )


# --- bullet / list numbering ---------------------------------------------


def _alpha(n: int, upper: bool) -> str:
    if n <= 0:
        n = 1
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr((ord("A") if upper else ord("a")) + r) + s
    return s


def _roman(n: int, upper: bool) -> str:
    pairs = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    s = ""
    if n <= 0:
        return "i" if not upper else "I"
    for v, r in pairs:
        while n >= v:
            s += r
            n -= v
    return s if upper else s.lower()


def _format_num(n: int, gtype: str) -> str:
    if gtype == "DECIMAL":
        return str(n)
    if gtype == "ZERO_DECIMAL":
        return f"{n:02d}"
    if gtype == "UPPER_ALPHA":
        return _alpha(n, True)
    if gtype == "ALPHA":
        return _alpha(n, False)
    if gtype == "UPPER_ROMAN":
        return _roman(n, True)
    if gtype == "ROMAN":
        return _roman(n, False)
    return str(n)


def resolve_glyph(
    list_def: dict,
    nesting_level: int,
    counters: dict[int, int],
) -> tuple[str, dict]:
    """Return (glyph_text, nestingLevel_def) for the bullet at nesting_level.
    counters is per-list (caller manages reset on level-up / listId change)."""
    nls = (list_def.get("listProperties") or {}).get("nestingLevels") or []
    if not nls:
        return "•", {}
    idx = min(nesting_level, len(nls) - 1)
    nl = nls[idx]
    if "glyphSymbol" in nl:
        return nl["glyphSymbol"], nl
    gtype_own = nl.get("glyphType") or "DECIMAL"
    if gtype_own == "GLYPH_TYPE_UNSPECIFIED":
        return nl.get("glyphSymbol", "•"), nl
    fmt = nl.get("glyphFormat") or "%0."
    out = fmt
    # replace %N for each level up to and including this one
    for i in range(idx + 1):
        n_i = counters.get(i, nls[i].get("startNumber", 1))
        t_i = (nls[i].get("glyphType") or "DECIMAL").upper()
        if t_i == "GLYPH_TYPE_UNSPECIFIED":
            t_i = "DECIMAL"
        out = out.replace(f"%{i}", _format_num(n_i, t_i))
    return out, nl


# --- paragraph rendering --------------------------------------------------


def paragraph_style_css(ps: dict, ns_defaults: dict) -> str:
    """Merge named-style defaults + paragraph-local overrides into CSS."""
    def pick(key):
        if key in ps:
            return ps[key]
        return (ns_defaults.get("paragraphStyle") or {}).get(key)

    p: list[str] = []
    # text alignment
    a = pick("alignment")
    if a == "CENTER":
        p.append("text-align:center")
    elif a == "END":
        p.append("text-align:right")
    elif a == "JUSTIFIED":
        p.append("text-align:justify")
    elif a == "START":
        p.append("text-align:left")
    # direction
    d = pick("direction")
    if d == "RIGHT_TO_LEFT":
        p.append("direction:rtl")
    # line spacing — Docs stores percentage (115 => 1.15)
    ls = pick("lineSpacing")
    if isinstance(ls, (int, float)) and ls > 0:
        p.append(f"line-height:{ls / 100:.3f}")
    # space above/below
    sa = pt_of(pick("spaceAbove"))
    if sa:
        p.append(f"margin-top:{sa}pt")
    sb = pt_of(pick("spaceBelow"))
    if sb:
        p.append(f"margin-bottom:{sb}pt")
    # paragraph shading (background)
    sh = pick("shading") or {}
    if sh:
        bg = rgb_css(sh.get("backgroundColor"))
        if bg:
            p.append(f"background-color:{bg}")
    return join_style(p)


def indent_css(ps: dict, is_bullet: bool) -> str:
    """Return CSS for indentStart / indentEnd / indentFirstLine.
    For bullet paragraphs we use hanging-indent: padding-left = indentStart,
    text-indent = indentFirstLine - indentStart (usually negative), so the
    bullet glyph sits in the outdented first line."""
    i_s = pt_of(ps.get("indentStart"))
    i_e = pt_of(ps.get("indentEnd"))
    i_f = pt_of(ps.get("indentFirstLine"))
    parts = []
    if is_bullet:
        if i_s:
            parts.append(f"padding-left:{i_s}pt")
        # text-indent is relative to content box, so include negative hang
        parts.append(f"text-indent:{i_f - i_s}pt")
    else:
        if i_s:
            parts.append(f"padding-left:{i_s}pt")
        if i_f:
            parts.append(f"text-indent:{i_f}pt")
    if i_e:
        parts.append(f"padding-right:{i_e}pt")
    return join_style(parts)


def heading_tag(named: str) -> str:
    return {
        "TITLE": "h1",
        "SUBTITLE": "h2",
        "HEADING_1": "h1",
        "HEADING_2": "h2",
        "HEADING_3": "h3",
        "HEADING_4": "h4",
        "HEADING_5": "h5",
        "HEADING_6": "h6",
    }.get(named, "p")


def render_paragraph(
    para: dict,
    inline_objects: dict,
    lists: dict,
    counters: dict,  # {listId: {level: n}}
    list_state: dict,  # {"last": (listId, level) or None}
    named_styles: dict,
) -> str:
    ps = para.get("paragraphStyle", {}) or {}
    named = ps.get("namedStyleType", "NORMAL_TEXT")
    ns_defaults = named_styles.get(named, {})

    # Inline glyph prefix for bullets.
    prefix_html = ""
    bullet = para.get("bullet")
    if bullet:
        list_id = bullet.get("listId")
        level = bullet.get("nestingLevel", 0) or 0
        list_def = lists.get(list_id, {})
        # reset deeper counters when we come up; reset everything when listId changes
        last = list_state.get("last")
        if last and last[0] != list_id:
            # entering a new list — leave other lists' counters alone (they persist
            # globally per Docs semantics) but we always reset levels deeper than
            # this one within the new list.
            for k in list(counters.get(list_id, {}).keys()):
                if k > level:
                    counters[list_id][k] = 0
        elif last and last[0] == list_id and last[1] > level:
            # returning to a shallower level — reset any deeper counters
            for k in list(counters[list_id].keys()):
                if k > level:
                    counters[list_id][k] = 0
        # increment counter at this level
        counters.setdefault(list_id, {})
        nls = (list_def.get("listProperties") or {}).get("nestingLevels") or []
        start = nls[level].get("startNumber", 1) if level < len(nls) else 1
        if level not in counters[list_id] or counters[list_id][level] == 0:
            counters[list_id][level] = start
        else:
            counters[list_id][level] += 1
        list_state["last"] = (list_id, level)
        glyph, nl_def = resolve_glyph(list_def, level, counters[list_id])
        bullet_ts = dict((nl_def.get("textStyle") or {}))
        # merge in paragraph-level bullet textStyle
        bullet_ts.update(bullet.get("textStyle") or {})
        # bullet numbers inherit paragraph's first-run style color / weight
        b_css = text_style_css(bullet_ts)
        prefix_html = (
            f'<span class="gd-bullet" style="{b_css}">{html.escape(glyph)}</span>'
            f'<span class="gd-bullet-sp">  </span>'
        )
    else:
        list_state["last"] = None

    # render elements
    body_parts: list[str] = []
    for el in para.get("elements", []) or []:
        if "textRun" in el:
            body_parts.append(render_text_run(el["textRun"]))
        elif "inlineObjectElement" in el:
            body_parts.append(render_inline_object(el, inline_objects))
        elif "richLink" in el:
            body_parts.append(render_rich_link(el))
        # other element types (footnote refs, etc.) aren't present in the target doc

    inner = "".join(body_parts)
    # an empty paragraph still needs a visible line break to preserve spacing
    if inner.strip() == "":
        inner = "<br>"

    style = join_style(
        [
            paragraph_style_css(ps, ns_defaults),
            indent_css(ps, is_bullet=bullet is not None),
        ]
    )

    tag = "p" if bullet else heading_tag(named)
    hid = ps.get("headingId")
    id_attr = f' id="{html.escape(hid)}"' if hid else ""
    cls_parts = ["gd-p"]
    if bullet:
        cls_parts.append("gd-bullet-line")
    cls_parts.append(f"gd-style-{named.lower()}")
    cls = " ".join(cls_parts)
    return (
        f'<{tag} class="{cls}"{id_attr} style="{style}">'
        f"{prefix_html}{inner}</{tag}>"
    )


# --- document-level rendering --------------------------------------------


BASE_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: #e8eaed;
  font-family: 'Arial', 'Helvetica', sans-serif;
  color: #202124;
  padding: 24px 12px 60px;
  min-height: 100vh;
}
.gd-page {
  background: #ffffff;
  margin: 0 auto;
  box-shadow: 0 2px 8px rgba(0,0,0,.15);
  border-radius: 2px;
  overflow-wrap: break-word;
}
.gd-p {
  margin: 0;
  padding: 0;
  font-size: 11pt;
  line-height: 1.15;
  white-space: pre-wrap;
}
h1.gd-p, h2.gd-p, h3.gd-p, h4.gd-p, h5.gd-p, h6.gd-p {
  font-weight: 400;
}
.gd-style-heading_1 { font-size: 20pt; color: #000; }
.gd-style-heading_2 { font-size: 16pt; color: #000; }
.gd-style-heading_3 { font-size: 14pt; color: #434343; }
.gd-style-heading_4 { font-size: 12pt; color: #666666; }
.gd-style-heading_5 { font-size: 11pt; color: #666666; }
.gd-style-heading_6 { font-size: 11pt; color: #666666; font-style: italic; }
.gd-style-title    { font-size: 26pt; color: #000; }
.gd-style-subtitle { font-size: 15pt; color: #666666; }
.gd-bullet { display: inline-block; }
.gd-bullet-sp { display: inline-block; }
a { color: #1155cc; text-decoration: underline; }
a.rich-link {
  border: 1px solid #dadce0;
  border-radius: 4px;
  padding: 1px 6px;
  text-decoration: none;
  background: #f8f9fa;
  color: #1a73e8;
  margin: 0 1px;
  font-size: .95em;
}
.gd-meta {
  max-width: 780pt;
  margin: 0 auto 16px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #5f6368;
  font-size: 12px;
  padding: 8px 12px;
  background: #fff;
  border: 1px solid #dadce0;
  border-radius: 4px;
}
.gd-meta b { color: #202124; }
.view-nav {
  max-width: 780pt;
  margin: 0 auto 12px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 13px;
}
.view-nav a {
  color: #555; text-decoration: none;
  padding: 4px 10px; border-radius: 3px;
  background: #fff; border: 1px solid #dadce0;
  margin-right: 6px; display: inline-block;
}
.view-nav a.current { background: #202124; color: #fff; border-color: #202124; }
.view-nav .exp { color: #7a5af5; font-weight: 600; font-size: 11px; margin-left: 4px; }
@media (max-width: 900px) {
  body { padding: 12px 0 40px; }
  .gd-page { border-radius: 0; }
}
@media print {
  body { background: #fff; padding: 0; }
  .gd-page { box-shadow: none; border-radius: 0; }
  .gd-meta, .view-nav { display: none; }
}
"""


VIEW_NAV_HTML = (
    '<div class="view-nav">'
    '<a href="index.html" class="current">Google Doc 原文</a>'
    '<a href="ledger.html">编辑历史</a>'
    '<a href="deduped.html">去重视图<span class="exp">实验</span></a>'
    '</div>'
)


def normalize_named_styles(named_styles_root: dict) -> dict:
    """Convert the {styles:[{namedStyleType,textStyle,paragraphStyle}, ...]}
    shape into a flat {namedStyleType: {textStyle, paragraphStyle}} dict."""
    out = {}
    for s in (named_styles_root or {}).get("styles", []) or []:
        t = s.get("namedStyleType")
        if t:
            out[t] = {
                "textStyle": s.get("textStyle", {}),
                "paragraphStyle": s.get("paragraphStyle", {}),
            }
    return out


def page_css(doc_style: dict) -> str:
    page = doc_style.get("pageSize") or {}
    page_w = pt_of(page.get("width")) or 612  # letter
    page_h_min = pt_of(page.get("height")) or 0
    m_t = pt_of(doc_style.get("marginTop")) or 72
    m_b = pt_of(doc_style.get("marginBottom")) or 72
    m_l = pt_of(doc_style.get("marginLeft")) or 72
    m_r = pt_of(doc_style.get("marginRight")) or 72
    bg = rgb_css(doc_style.get("background")) or "#ffffff"
    parts = [
        f"width:{page_w}pt",
        f"padding:{m_t}pt {m_r}pt {m_b}pt {m_l}pt",
        f"background-color:{bg}",
    ]
    if page_h_min:
        parts.append(f"min-height:{page_h_min}pt")
    return join_style(parts)


def render_html(doc: dict, meta_banner: str | None = None) -> str:
    named_styles = normalize_named_styles(doc.get("namedStyles") or {})
    inline_objects = doc.get("inlineObjects") or {}
    lists = doc.get("lists") or {}
    counters: dict[str, dict[int, int]] = defaultdict(dict)
    list_state: dict = {"last": None}

    body_parts: list[str] = []
    for element in (doc.get("body") or {}).get("content", []) or []:
        if "paragraph" in element:
            body_parts.append(
                render_paragraph(
                    element["paragraph"],
                    inline_objects,
                    lists,
                    counters,
                    list_state,
                    named_styles,
                )
            )
        elif "sectionBreak" in element:
            list_state["last"] = None  # section break ends bullet continuity
            # we don't render page-break section breaks; they're implicit
        elif "table" in element:
            body_parts.append(render_table(element["table"], inline_objects, lists,
                                            counters, list_state, named_styles))

    title = doc.get("title") or "Document"
    page_style = page_css(doc.get("documentStyle") or {})
    meta_html = f'<div class="gd-meta">{meta_banner}</div>' if meta_banner else ""
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="zh"><head>'
        f'<meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        f"<style>{BASE_CSS}</style>"
        f"</head><body>"
        f"{VIEW_NAV_HTML}"
        f"{meta_html}"
        f'<article class="gd-page" style="{page_style}">'
        f"{''.join(body_parts)}"
        f"</article></body></html>"
    )


def render_table(table, inline_objects, lists, counters, list_state, named_styles):
    """Minimal table renderer (the target doc has none, but keep for generality)."""
    rows_html = []
    for row in table.get("tableRows", []) or []:
        cells_html = []
        for cell in row.get("tableCells", []) or []:
            cell_parts = []
            for el in cell.get("content", []) or []:
                if "paragraph" in el:
                    cell_parts.append(render_paragraph(
                        el["paragraph"], inline_objects, lists, counters,
                        list_state, named_styles,
                    ))
            cells_html.append(
                f'<td style="border:1px solid #ccc;padding:4px;vertical-align:top">'
                f"{''.join(cell_parts)}</td>"
            )
        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")
    return f'<table style="border-collapse:collapse;width:100%;margin:8px 0">{"".join(rows_html)}</table>'


# --- cli ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Path to documents.get JSON")
    ap.add_argument("output", type=Path, help="Path to write HTML")
    ap.add_argument(
        "--meta",
        default=None,
        help="Optional HTML fragment shown above the page (e.g. snapshot info)",
    )
    args = ap.parse_args()

    doc = json.loads(args.input.read_text(encoding="utf-8"))
    html_out = render_html(doc, meta_banner=args.meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_out, encoding="utf-8")
    print(f"[ok] wrote {args.output} ({len(html_out):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
