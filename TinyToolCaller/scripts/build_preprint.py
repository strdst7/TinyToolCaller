#!/usr/bin/env python3
"""Build the TinyToolCaller publication as a print-ready PDF (A4).

Converts README.md to HTML (Python-Markdown: tables, fenced_code, sane_lists)
and renders it through WeasyPrint with a publication stylesheet: a cover page,
running headers/footers with page numbers, styled tables, callouts, and code.

Usage:
    pip install markdown weasyprint
    python scripts/build_preprint.py                     # -> preprint/TinyToolCaller_Publication.pdf
    python scripts/build_preprint.py --input README.md --output preprint/out.pdf
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import markdown
from weasyprint import HTML

COVER = """\
<div class="cover">
  <div class="cover-meta">APPLIED LLM ENGINEERING STUDY &middot; EVIDENCE-LED PREPRINT &middot; {date}</div>
  <h1 class="cover-title">TinyToolCaller</h1>
  <div class="cover-subtitle">A QLoRA Study of a 1.5B Model for Structured Function Calling</div>

  <div class="cover-abstract">
    <p>TinyToolCaller studies parameter-efficient specialization of
    <code>Qwen2.5-1.5B-Instruct</code> for one narrow contract: map a request and candidate tool
    schemas to a structured call. It applies QLoRA to 5,000 supervised examples from a
    deterministic xLAM subset. The recorded numbers below are a development-set snapshot, not an
    external benchmark or a claim of production reliability. The engineering contribution is an
    inspectable pipeline and a deployment boundary in which deterministic software validates,
    authorizes, and executes every proposed call.</p>
  </div>

  <table class="cover-results">
    <tr><th>Recorded development metric (n=200)</th><th>Base model</th><th>TinyToolCaller</th><th>Change</th></tr>
    <tr><td>Extractable JSON object</td><td>78.5%</td><td><strong>98.0%</strong></td><td>+19.5 pp</td></tr>
    <tr><td>Tool-name accuracy</td><td>65.0%</td><td><strong>92.5%</strong></td><td>+27.5 pp</td></tr>
    <tr><td>Argument exact match</td><td>42.0%</td><td><strong>84.0%</strong></td><td>+42.0 pp</td></tr>
    <tr><td>Recorded GSM8K probe (n=50)</td><td>52.0%</td><td><strong>50.0%</strong></td><td>&minus;2.0 pp</td></tr>
  </table>

  <div class="cover-caveat"><strong>Read before quoting:</strong> the 200-row split was used during
  development, JSON is scored after extraction rather than as raw-output purity — see the evidence
  notice and limitations. Tool distribution is now profiled (§8: 1,774 unique tools, top 1.62%,
  train/val χ² p=0.114).</div>

  <div class="cover-links">
    <div><span>Code</span> github.com/strdst7/TinyToolCaller</div>
    <div><span>Dataset</span> huggingface.co/datasets/strdst77/TinyToolCaller</div>
    <div><span>Source data</span> huggingface.co/datasets/Salesforce/xlam-function-calling-60k</div>
    <div><span>Base model</span> huggingface.co/Qwen/Qwen2.5-1.5B-Instruct</div>
  </div>
</div>
"""

CSS = """
@page {
  size: A4;
  margin: 2.1cm 1.9cm 2.3cm 1.9cm;
  @top-center { content: "TinyToolCaller — Publication"; font-size: 7.5pt; color: #999; }
  @bottom-left { content: "Nur Amirah Mohd Kamil | 2026"; width: 5cm; font-size: 7pt; color: #555; }
  @bottom-center { content: "Ready Tensor for LLM Fine-Tuning Specialist — Fine-tune and optimize an LLM using PEFT techniques"; font-size: 6.5pt; color: #555; }
  @bottom-right { content: counter(page) " / " counter(pages); font-size: 7.5pt; color: #666; }
}
@page :first {
  @top-center { content: none; }
}

html { font-size: 10.5pt; }
body {
  font-family: Georgia, "DejaVu Serif", serif;
  font-size: 10.5pt; line-height: 1.52; color: #1b1b1b;
  hyphens: auto;
}
h1, h2, h3, h4, h5, h6 {
  font-family: "DejaVu Sans", "Helvetica Neue", Arial, sans-serif;
  color: #10243e; line-height: 1.25; break-after: avoid;
}
h1 { font-size: 20pt; margin: 0 0 0.35em 0; border-bottom: 2px solid #10243e; padding-bottom: 0.18em; }
h2 { font-size: 14pt; margin: 1.4em 0 0.5em 0; }
h3 { font-size: 11.5pt; margin: 1.2em 0 0.4em 0; }
h4 { font-size: 10.5pt; margin: 1em 0 0.3em 0; }
p { margin: 0.5em 0; }
ul, ol { margin: 0.5em 0 0.7em 0; padding-left: 1.4em; }
li { margin: 0.18em 0; }
a { color: #0b4d8c; text-decoration: none; }
strong { color: #10243e; }
hr { border: 0; border-top: 1px solid #d8d8d8; margin: 1.6em 0; }

code {
  font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace;
  font-size: 8.6pt; background: #f2f4f7; padding: 0.5px 3px; border-radius: 2px;
}
pre {
  font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace;
  font-size: 8.3pt; line-height: 1.4; background: #f6f8fa;
  border: 1px solid #e2e6ea; border-radius: 4px; padding: 8px 10px;
  white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;
  break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.3pt; }

table {
  border-collapse: collapse; width: 100%; margin: 0.7em 0 1em 0;
  font-family: "DejaVu Sans", Arial, sans-serif; font-size: 8.6pt;
  break-inside: avoid;
}
th, td { border: 1px solid #cdd3da; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #eef1f5; color: #10243e; font-weight: bold; }
tr:nth-child(even) td { background: #f8fafc; }
.evidence-map { break-inside: auto; }
.evidence-map tr { break-inside: avoid; }

blockquote {
  margin: 0.8em 0; padding: 6px 12px; border-left: 4px solid #9db4cc;
  background: #f4f7fb; color: #2a2f36; break-inside: avoid;
}
blockquote p { margin: 0.3em 0; }

/* ---- cover page ---- */
.cover { page-break-after: always; padding-top: 4.5cm; }
.cover-meta { font-family: "DejaVu Sans", sans-serif; font-size: 9pt; letter-spacing: 2.5px; color: #8a8a8a; margin-bottom: 1.6em; }
.cover-title { font-size: 42pt; border-bottom: none; margin: 0; }
.cover-subtitle { font-family: "DejaVu Sans", sans-serif; font-size: 15pt; color: #4a5a6d; margin: 0.4em 0 2em 0; }
.cover-abstract { border-top: 2px solid #10243e; border-bottom: 1px solid #cdd3da; padding: 1em 0; margin-bottom: 1.6em; }
.cover-abstract p { font-size: 10.8pt; line-height: 1.55; }
.cover-caveat { font-family: "DejaVu Sans", sans-serif; font-size: 8.6pt; background: #fbf3e6; border-left: 4px solid #e0a83c; padding: 8px 12px; margin: 1.2em 0; }
.cover-links { font-family: "DejaVu Sans", sans-serif; font-size: 9pt; margin-top: 2.2em; color: #333; }
.cover-links div { margin: 0.28em 0; }
.cover-links span { display: inline-block; width: 7.5em; color: #8a8a8a; }
"""


def build_html(md_text: str, date: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    showcase_start = "<h2>Showcase Evidence Map</h2>"
    showcase_end = "<h2>1."
    if showcase_start in body:
        idx_start = body.index(showcase_start)
        idx_end = body.index(showcase_end, idx_start + len(showcase_start))
        before = body[:idx_start]
        section = body[idx_start:idx_end]
        after = body[idx_end:]
        section = section.replace('<table>', '<table class="evidence-map">', 1)
        body = before + section + after
    body = body.replace("\uFE0F", "")          # strip emoji variation selector
    body = body.replace("⚠️", "⚠")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>"
        f"{COVER.format(date=date)}"
        f"{body}</body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="README.md")
    parser.add_argument("--output", default="preprint/TinyToolCaller_Publication.pdf")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = (root / args.input).read_text(encoding="utf-8")
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    date = datetime.date.today().strftime("%B %Y")
    html = build_html(src, date)
    HTML(string=html, base_url=str(root)).write_pdf(str(out))
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
