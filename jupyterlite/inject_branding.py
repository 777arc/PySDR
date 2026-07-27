#!/usr/bin/env python3
"""Inject PySDR branding into a built JupyterLite site.

Run AFTER `jupyter lite build`. JupyterLab 4 / Notebook 7 render the top-left
logo from JavaScript and offer no user-CSS hook, so we inline a small stylesheet
(pysdr-brand.css) into every generated app page. It hides the built-in Jupyter
mark (#jp-NotebookLogo for the Notebook apps, #jp-MainLogo for Lab) and paints
the PySDR logo from /_static/logo.svg instead.

Usage: python jupyterlite/inject_branding.py [output_dir]
       (output_dir defaults to _build/jupyterlite)

Exits non-zero if it patches nothing, so CI fails loudly if a JupyterLite
upgrade changes the build layout out from under us.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "pysdr-branding"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_build/jupyterlite")
    if not out.is_dir():
        print(f"[inject_branding] output dir not found: {out}", file=sys.stderr)
        return 1

    css = (HERE / "pysdr-brand.css").read_text(encoding="utf-8")
    block = f'<style data-{MARKER}="1">\n{css}\n</style>'

    patched = 0
    for html in out.rglob("*.html"):
        text = html.read_text(encoding="utf-8")
        if MARKER in text or "</head>" not in text:
            continue
        html.write_text(text.replace("</head>", block + "\n</head>", 1), encoding="utf-8")
        patched += 1

    print(f"[inject_branding] patched {patched} page(s) under {out}")
    return 0 if patched else 1


if __name__ == "__main__":
    raise SystemExit(main())
