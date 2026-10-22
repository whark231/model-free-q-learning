"""Inline data.json into template.html to produce report.html (one self-contained page).

Run from the repo root, after build_data.py:
    python reports/full_run_2026-10-22/build_report.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "/*__DATA__*/null"

template = (HERE / "template.html").read_text(encoding="utf-8")
assert template.count(PLACEHOLDER) == 1, "template.html must contain the data placeholder exactly once"
data = (HERE / "data.json").read_text(encoding="utf-8")
(HERE / "report.html").write_text(template.replace(PLACEHOLDER, data), encoding="utf-8")
print(f"wrote {HERE / 'report.html'}")
