"""Build the results dashboard: one self-contained HTML file.

Reads the metrics.json each training run wrote, draws the figures with
matplotlib, and renders a page that opens offline with no server. Every
number on the page comes from a metrics file; nothing is typed in here, so
the dashboard cannot drift from what the code computed.

    python -m dashboard.build --out site/index.html
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
from pathlib import Path

from . import figures
from .theme import DARK, LIGHT

PROJECTS = [
    {
        "key": "fraud-detection",
        "title": "Fraud detection",
        "problem": "Card fraud at a 0.17% base rate.",
        "point": "The threshold is chosen by pricing the two error types "
                 "against each other, not by maximising a score. Adding "
                 "<code>scale_pos_weight</code> to handle the imbalance made "
                 "the ranking worse.",
        "figure": figures.fraud,
        "caption": "Ranking quality and money kept, as two panels: they are "
                   "different measures on different scales, so they do not "
                   "share an axis.",
    },
    {
        "key": "demand-forecasting",
        "title": "Demand forecasting",
        "problem": "28-day demand for 60 SKUs.",
        "point": "Every lag derives from the forecast horizon, so "
                 "<code>lag_1</code> cannot be built at a 28-day horizon. "
                 "Getting that wrong leaks the future into the features.",
        "figure": figures.demand,
        "caption": "Error by model across six rolling origins, and how error "
                   "grows the further ahead you forecast.",
    },
    {
        "key": "semantic-similarity",
        "title": "Semantic similarity",
        "problem": "Duplicate question detection.",
        "point": "The split protocol moves the score more than the model "
                 "does — and changes which model you would pick. A random "
                 "pair split leaks questions between train and test.",
        "figure": figures.semantic,
        "caption": "The same four models under both split protocols. The gap "
                   "between the two bars of a pair is the protocol effect.",
    },
    {
        "key": "recommender-system",
        "title": "Recommender system",
        "problem": "Implicit-feedback recommendations.",
        "point": "Implicit ALS from scratch. Ranking models on accuracy "
                 "alone picks one that recommends the same popular items to "
                 "everyone.",
        "figure": figures.recommender,
        "caption": "Recall against catalogue coverage. A model can win on "
                   "recall while showing a fraction of the catalogue.",
    },
]

CSS = """
:root {
  --surface:#ffffff; --sunken:#f7f8fa; --rule:#e4e6ea;
  --ink:#16181d; --ink-2:#454951; --ink-3:#6b7078;
  --accent:#2a78d6; --accent-ink:#1d5fae; --accent-bg:#edf3fc;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface:#1a1a19; --sunken:#232322; --rule:#3a3a37;
    --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#95948b;
    --accent:#3987e5; --accent-ink:#7fb2f0; --accent-bg:#1f2a38;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --surface:#1a1a19; --sunken:#232322; --rule:#3a3a37;
  --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#95948b;
  --accent:#3987e5; --accent-ink:#7fb2f0; --accent-bg:#1f2a38;
  color-scheme: dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink-2);
  font:15px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:36px 16px 80px}
header{border-bottom:1px solid var(--rule);padding-bottom:22px}
h1{font-size:1.75rem;margin:0 0 8px;color:var(--ink);letter-spacing:-0.02em}
.lede{margin:0;max-width:64ch}
.note{background:var(--sunken);border:1px solid var(--rule);
  border-left:3px solid var(--accent);border-radius:8px;
  padding:12px 16px;margin:20px 0;font-size:.92rem}
.note strong{color:var(--ink)}
.controls{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:8px;
  align-items:center;background:var(--surface);border-bottom:1px solid var(--rule);
  padding:12px 0;margin-bottom:6px}
.tab{font:inherit;font-size:.9rem;cursor:pointer;color:var(--ink-2);
  background:var(--sunken);border:1px solid var(--rule);border-radius:7px;
  padding:7px 14px}
.tab[aria-pressed="true"]{background:var(--accent-bg);color:var(--accent-ink);
  border-color:var(--accent);font-weight:600}
.tab:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.spacer{flex:1 1 auto}
section.project{padding-top:34px}
section.project[hidden]{display:none}
h2{font-size:1.2rem;margin:0 0 4px;color:var(--ink);letter-spacing:-0.01em}
.problem{color:var(--ink-3);font-size:.92rem;margin:0 0 14px}
.point{background:var(--sunken);border-radius:8px;padding:14px 16px;margin:0 0 18px}
figure{margin:18px 0 6px}
figure img{width:100%;height:auto;display:block;border-radius:8px}
figcaption{font-size:.86rem;color:var(--ink-3);margin-top:8px}
img.dark-only{display:none}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]) img.light-only{display:none}
  :root:not([data-theme="light"]) img.dark-only{display:block}
}
:root[data-theme="dark"] img.light-only{display:none}
:root[data-theme="dark"] img.dark-only{display:block}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:14px 0}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink-3);font-size:.74rem;text-transform:uppercase;
  letter-spacing:.04em;font-weight:600}
td{color:var(--ink-2);font-variant-numeric:tabular-nums}
td.best{color:var(--ink);font-weight:650}
code{background:var(--sunken);padding:1px 5px;border-radius:4px;font-size:.88em}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--rule);
  font-size:.85rem;color:var(--ink-3)}
@media (max-width:640px){.wrap{padding:20px 16px 60px}h1{font-size:1.4rem}}
@media print{.controls{display:none}section.project[hidden]{display:revert !important}}
"""

SCRIPT = """
(function(){
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tab[data-target]'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('section.project'));
  function show(key){
    sections.forEach(function(s){ s.hidden = (key !== 'all' && s.dataset.key !== key); });
    tabs.forEach(function(t){ t.setAttribute('aria-pressed', String(t.dataset.target === key)); });
  }
  tabs.forEach(function(t){ t.addEventListener('click', function(){ show(t.dataset.target); }); });

  var theme = document.getElementById('theme');
  function current(){
    var set = document.documentElement.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function paint(){
    var now = current();
    theme.textContent = now === 'dark' ? 'Light mode' : 'Dark mode';
    theme.setAttribute('aria-pressed', now === 'dark' ? 'true' : 'false');
  }
  theme.addEventListener('click', function(){
    document.documentElement.setAttribute('data-theme', current() === 'dark' ? 'light' : 'dark');
    paint();
  });
  paint();
  show('all');
})();
"""


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def fig_html(light: bytes, dark: bytes, alt: str, caption: str) -> str:
    safe = html.escape(alt)
    return "\n".join([
        "<figure>",
        f'  <img class="light-only" alt="{safe}" src="data:image/png;base64,{b64(light)}">',
        f'  <img class="dark-only" alt="{safe}" src="data:image/png;base64,{b64(dark)}">',
        f"  <figcaption>{caption}</figcaption>",
        "</figure>",
    ])


def table(headers: list[str], rows: list[list[str]], best_row: int | None = None) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for i, r in enumerate(rows):
        cls = ' class="best"' if best_row is not None and i == best_row else ""
        cells = "".join(f"<td{cls if j == 0 else ''}>{c}</td>" for j, c in enumerate(r))
        body.append(f"<tr>{cells}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead><tbody>"
            + "".join(body) + "</tbody></table>")


def fraud_table(m: dict) -> str:
    rows, best, best_v = [], None, -1.0
    for i, (name, d) in enumerate(m["models"].items()):
        pr = d["ranking"]["pr_auc"]
        if pr > best_v:
            best, best_v = i, pr
        rows.append([
            figures.label(name), f"{pr:.3f}",
            f"{d['operating_point']['precision'] * 100:.1f}%",
            f"{d['operating_point']['recall'] * 100:.1f}%",
            f"{d['operating_point']['savings_rate'] * 100:.0f}%",
        ])
    return table(["Model", "PR-AUC", "Precision", "Recall", "Net savings"], rows, best)


def demand_table(m: dict) -> str:
    rows = sorted(m["models"], key=lambda r: r["wape"])
    out = [[figures.label(r["model"]), f"{r['wape']:.3f}",
            f"{r['wape_worst_fold']:.3f}",
            f"{r['fva_vs_seasonal_naive'] * 100:+.0f}%"] for r in rows]
    return table(["Model", "WAPE", "Worst fold", "Value add vs naive"], out, 0)


def semantic_table(m: dict) -> str:
    protos = list(m["protocols"])
    models = list(m["protocols"][protos[0]]["models"])
    key = next(k for k in ("pr_auc", "average_precision", "roc_auc", "f1")
               if k in m["protocols"][protos[0]]["models"][models[0]]["overall"])
    rows = []
    for mod in models:
        vals = [m["protocols"][p]["models"][mod]["overall"][key] for p in protos]
        rows.append([figures.label(mod)] + [f"{v:.3f}" for v in vals]
                    + [f"{vals[0] - vals[1]:+.3f}"])
    return table(["Model"] + [p.replace("_", " ") for p in protos] + ["Difference"], rows)


def recommender_table(m: dict) -> str:
    rows, best, best_v = [], None, -1.0
    for i, (name, d) in enumerate(m["models"].items()):
        if d["recall"] > best_v:
            best, best_v = i, d["recall"]
        rows.append([figures.label(name), f"{d['recall']:.3f}", f"{d['ndcg']:.3f}",
                     f"{d['catalogue_coverage'] * 100:.1f}%", f"{d['novelty']:.2f}"])
    return table([f"Model", f"Recall@{m.get('k', 10)}", "nDCG",
                  "Catalogue coverage", "Novelty"], rows, best)


TABLES = {
    "fraud-detection": fraud_table,
    "demand-forecasting": demand_table,
    "semantic-similarity": semantic_table,
    "recommender-system": recommender_table,
}


def build(root: Path, out_path: Path) -> Path:
    parts: list[str] = []
    generated = dt.datetime.now().strftime("%d %B %Y")

    parts.append(f"""<header>
  <h1>ML portfolio — results</h1>
  <p class="lede">Four self-contained machine learning projects. Each runs end
  to end on a laptop CPU in under three minutes, has a test suite, and is built
  around a decision someone would have to make rather than a leaderboard
  score.</p>
</header>""")

    tabs = ['<button class="tab" data-target="all" aria-pressed="true">All</button>']
    for p in PROJECTS:
        tabs.append(f'<button class="tab" data-target="{p["key"]}" '
                    f'aria-pressed="false">{html.escape(p["title"])}</button>')
    parts.append('<div class="controls">' + "".join(tabs)
                 + '<div class="spacer"></div>'
                 + '<button class="tab" id="theme" aria-pressed="false">Dark mode</button>'
                 + "</div>")

    parts.append("""<div class="note">
  <strong>Synthetic data throughout.</strong> Every dataset is generated by code
  in this repository. That keeps the feature engineering, the leakage tests and
  the evaluation protocols inspectable — and it means none of these numbers are
  evidence about a real business.
</div>""")

    for p in PROJECTS:
        metrics_path = root / p["key"] / "artifacts" / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"missing {metrics_path}; run the training first")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        body = [
            f'<h2>{html.escape(p["title"])}</h2>',
            f'<p class="problem">{html.escape(p["problem"])}</p>',
            f'<div class="point">{p["point"]}</div>',
            fig_html(p["figure"](metrics, LIGHT), p["figure"](metrics, DARK),
                     f'{p["title"]} results', p["caption"]),
            TABLES[p["key"]](metrics),
        ]
        parts.append(f'<section class="project" data-key="{p["key"]}">'
                     + "".join(body) + "</section>")

    parts.append(f"""<footer>
  Generated {generated} from the metrics each training run wrote, by
  <code>python -m dashboard.build</code>. Figures are drawn with matplotlib
  from those files, so a figure cannot disagree with the table beside it.
  Source: <a href="https://github.com/Wrlog/ml-portfolio">Wrlog/ml-portfolio</a>.
</footer>""")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML portfolio — results</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{"".join(parts)}
</div>
<script>{SCRIPT}</script>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("site/index.html"))
    args = ap.parse_args()
    path = build(args.root, args.out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
