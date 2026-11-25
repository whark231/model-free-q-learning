#!/bin/sh
# Rebuild the figures, the LaTeX paper (latex/REPORT.tex -> REPORT.pdf), the slide deck and its PDF.
# Source of the paper text is REPORT.md; pandoc turns it into LaTeX and XeLaTeX typesets it.
# Needs TeX Live (xelatex, latexmk, libertinus-otf). Run from anywhere.
set -e
cd "$(dirname "$0")"
PY=${PY:-/opt/anaconda3/bin/python}
PANDOC=${PANDOC:-/opt/anaconda3/bin/pandoc}
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
"$PY" figures/make_figures.py
"$PANDOC" REPORT.md -o latex/REPORT.tex --standalone -f markdown+implicit_figures+link_attributes \
  -V documentclass=article -V classoption=11pt -V geometry:margin=1in \
  -H latex/preamble.tex --lua-filter latex/table_captions.lua
"$PY" latex/fix_tables.py latex/REPORT.tex
(cd latex && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build REPORT.tex >/dev/null)
cp latex/build/REPORT.pdf REPORT.pdf
# The deck embeds the PNGs written above, so rebuild it before printing.
"$PY" slides/build_slides.py
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=slides/slides.pdf "file://$PWD/slides/slides.html"
