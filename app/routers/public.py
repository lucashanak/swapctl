"""Public, read-only view of measured model results.

Exists so the results can be embedded somewhere without a login — a
presentation, a wiki page. It renders the numbers the /testing dashboard
already shows and nothing else: scores, speeds, sizes. No controls, no
configuration, no filenames, no repo ids, no sampling parameters, no keys.

Whitelisted in ``main._PUBLIC_EXACT``; everything else stays behind the
session gate.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["public"])

# Pořadí a zkratky kategorií inteligenčního testu. Cokoli, co v datech přijde
# navíc, se přidá na konec — tabulka se tím nerozbije.
_CATS = [
    ("knowledge", "znal"), ("reasoning", "úvah"), ("math", "mat"),
    ("code", "kód"), ("coding", "prog"), ("czech", "češ"),
    ("instruction", "instr"), ("structured", "struk"),
    ("summarization", "souhrn"), ("sysadmin", "sys"),
    ("longctx", "dlouhý"), ("longctx_agg", "dl.agg"), ("exec", "exec"),
]

_STYLE = """
  :root {
    --bg: #f6f8f9; --ink: #10161b; --ink2: #3d4a55; --ink3: #6b7883;
    --line: #afb9c3; --line-soft: #c6cfd7; --panel: #edf1f3;
    --accent: #0c6f79; --warn: #a8481a; --ok: #2f6b45;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #141b21; --ink: #dce4ea; --ink2: #a3b0ba; --ink3: #74838e;
      --line: #3a4854; --line-soft: #2b3740; --panel: #1a2329;
      --accent: #35b3b0; --warn: #db9058; --ok: #6fb88a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0.45rem;
    background: var(--bg); color: var(--ink);
    font: 13px/1.35 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  }
  .head {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 1rem; padding: 0 0.2rem 0.4rem; flex-wrap: wrap;
    font-family: ui-monospace, "JetBrains Mono", monospace;
    font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--ink3);
  }
  .head b { color: var(--accent); font-weight: 700; }
  h2 {
    margin: 0.85rem 0.2rem 0.35rem;
    font-family: ui-monospace, monospace; font-size: 0.7rem;
    letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--ink3); font-weight: 500;
  }
  .wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 2px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 0.3rem 0.6rem; border-bottom: 1px solid var(--line-soft); white-space: nowrap; }
  thead th {
    position: sticky; top: 0; z-index: 1;
    background: var(--panel); border-bottom: 1px solid var(--line);
    font-family: ui-monospace, monospace; font-size: 0.65rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink3);
    font-weight: 500; text-align: right;
  }
  thead th.l { text-align: left; }
  tbody tr:last-child td { border-bottom: none; }
  td {
    font-family: ui-monospace, "JetBrains Mono", monospace;
    font-size: 0.78rem; font-variant-numeric: tabular-nums;
    text-align: right; color: var(--ink2);
  }
  td.name { text-align: left; color: var(--ink); }
  td.dim { color: var(--ink3); }
  td.hi { color: var(--accent); font-weight: 700; }
  td.lo { color: var(--warn); }
  .note {
    margin: 0.5rem 0.2rem 0; font-size: 0.72rem; line-height: 1.45; color: var(--ink3);
  }
"""


def _rows() -> tuple[list[dict], int, list[tuple[str, str]]]:
    from .. import registry

    records = registry.list_all()
    rows: list[dict] = []
    seen_cats: set[str] = set()

    for r in records:
        tr = r.test_results or {}
        iq = tr.get("intelligence") or {}
        ag = tr.get("agentic") or {}
        if not iq.get("score") and not ag.get("score"):
            continue

        gen = pp = 0.0
        vram = accept = None
        for e in (r.speed_history or []):
            if e.get("test") == "gen":
                if (e.get("gen_tps") or 0) > gen:
                    gen = e["gen_tps"]
                    vram = e.get("vram_peak_mb") or vram
                    accept = e.get("accept_pct") if e.get("accept_pct") is not None else accept
            elif e.get("test") == "pp":
                if (e.get("pp_tps") or 0) > pp:
                    pp = e["pp_tps"]
                    vram = e.get("vram_peak_mb") or vram

        cats = {}
        for name, block in (iq.get("categories") or {}).items():
            if isinstance(block, dict) and block.get("score") is not None:
                cats[name] = block["score"]
                seen_cats.add(name)

        rows.append({
            "id": r.model_id or "?",
            "quant": r.quant or "",
            "gb": (r.size_bytes or 0) / 1073741824,
            "ctx": r.ctx_size or None,
            "iq": iq.get("score"),
            "iqw": iq.get("weighted_score"),
            "passed": iq.get("passed"),
            "total": iq.get("total"),
            "iq_dur": iq.get("duration"),
            "degraded": bool(iq.get("degraded")),
            "ag": ag.get("score"),
            "solve": ag.get("solve_rate"),
            "turns": ag.get("avg_turns"),
            "tools": ag.get("avg_tool_calls"),
            "gen": gen or None,
            "pp": pp or None,
            "vram": vram,
            "accept": accept,
            "backend": tr.get("best_backend") or "",
            "cats": cats,
        })

    rows.sort(key=lambda x: -(x["iq"] or 0))
    ordered = [(k, lbl) for k, lbl in _CATS if k in seen_cats]
    ordered += [(k, k[:6]) for k in sorted(seen_cats) if k not in dict(_CATS)]
    return rows, len(records), ordered


def _n(v, fmt="{:.0f}", dash="—"):
    return dash if v is None else fmt.format(v)


def _main_table(rows: list[dict]) -> str:
    best_iq = max((r["iq"] or 0) for r in rows) if rows else 0
    best_ag = max((r["ag"] or 0) for r in rows) if rows else 0
    best_gen = max((r["gen"] or 0) for r in rows) if rows else 0
    head = ("<tr><th class='l'>Model</th><th class='l'>kvant</th><th>velikost</th><th>ctx</th>"
            "<th>IQ</th><th>váž.</th><th>prošlo</th><th>čas</th>"
            "<th>agentic</th><th>solve</th><th>kroků</th><th>nástrojů</th>"
            "<th>t/s</th><th>pp t/s</th><th>VRAM</th><th class='l'>backend</th></tr>")
    body = []
    for r in rows:
        iq_cls = "hi" if r["iq"] and r["iq"] >= best_iq else ""
        ag_cls = "hi" if r["ag"] and r["ag"] >= best_ag else ("lo" if r["ag"] == 0 else "")
        gen_cls = "hi" if r["gen"] and r["gen"] >= best_gen else ""
        body.append(
            f"<tr><td class='name'>{r['id']}</td>"
            f"<td class='name dim'>{r['quant']}</td>"
            f"<td>{_n(r['gb'], '{:.0f} GB')}</td>"
            f"<td class='dim'>{_n(r['ctx'] / 1024 if r['ctx'] else None, '{:.0f}k')}</td>"
            f"<td class='{iq_cls}'>{_n(r['iq'], '{:.1f}')}</td>"
            f"<td class='dim'>{_n(r['iqw'], '{:.1f}')}</td>"
            f"<td class='dim'>{r['passed']}/{r['total']}</td>"
            f"<td class='dim'>{_n(r['iq_dur'] / 60 if r['iq_dur'] else None, '{:.0f} min')}</td>"
            f"<td class='{ag_cls}'>{_n(r['ag'], '{:.1f}')}</td>"
            f"<td class='dim'>{_n(r['solve'], '{:.0f} %')}</td>"
            f"<td class='dim'>{_n(r['turns'], '{:.1f}')}</td>"
            f"<td class='dim'>{_n(r['tools'], '{:.1f}')}</td>"
            f"<td class='{gen_cls}'>{_n(r['gen'], '{:.0f}')}</td>"
            f"<td class='dim'>{_n(r['pp'], '{:.0f}')}</td>"
            f"<td class='dim'>{_n(r['vram'] / 1024 if r['vram'] else None, '{:.1f} GB')}</td>"
            f"<td class='name dim'>{r['backend']}</td></tr>"
        )
    return f"<div class='wrap'><table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"


def _cat_table(rows: list[dict], cats: list[tuple[str, str]]) -> str:
    have = [r for r in rows if r["cats"]]
    if not have or not cats:
        return ""
    head = "<tr><th class='l'>Model</th>" + "".join(f"<th>{lbl}</th>" for _, lbl in cats) + "</tr>"
    body = []
    for r in have:
        cells = []
        for key, _ in cats:
            v = r["cats"].get(key)
            cls = "dim" if v is None else ("hi" if v >= 90 else ("lo" if v < 40 else ""))
            cells.append(f"<td class='{cls}'>{_n(v, '{:.0f}')}</td>")
        body.append(f"<tr><td class='name'>{r['id']}</td>{''.join(cells)}</tr>")
    return f"<div class='wrap'><table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"


@router.get("/public/models", response_class=HTMLResponse)
def public_models() -> HTMLResponse:
    from datetime import datetime, timezone

    rows, total, cats = _rows()
    ts = datetime.now(timezone.utc).strftime("%d. %m. %Y %H:%M UTC")

    page = (
        "<!doctype html><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Výsledky testů modelů</title>"
        f"<style>{_STYLE}</style>"
        f"<div class='head'><span>Výsledky testů modelů · <b>{len(rows)}</b> změřených z {total}</span>"
        f"<span>swapctl · {ts}</span></div>"
        + _main_table(rows)
        + ("<h2>Inteligenční test po kategoriích</h2>" + _cat_table(rows, cats) if cats else "")
        + "<p class='note'>Skóre je poslední výsledek na model, ne průměr přes historii. "
          "<b>váž.</b> je skóre po váhách kategorií, <b>solve</b> podíl vyřešených epizod agentického testu, "
          "<b>t/s</b> generování a <b>pp t/s</b> zpracování promptu, obojí nejlepší naměřené.</p>"
    )
    return HTMLResponse(page, headers={"Cache-Control": "public, max-age=60"})
