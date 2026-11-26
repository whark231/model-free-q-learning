# Report final draft: summary of changes

**TL;DR / In plain terms:** `REPORT_final.md` / `.pdf` start from Albert's v2 (`REPORT_v2_AA.md`), apply the rest of William's markup of the v1 PDF (`REPORT_AA_annotated_v2.pdf`), and shorten the main text from about 15 pages (v2) to 13 including references. No result changed. v1 and v2 are untouched. An independent review checked every new or reworded number against the committed result files.

**Key concepts to understand**
- **Main text.** Introduction to Conclusion plus References: pages 1 to 13. Appendices A to C follow on pages 14 to 23 and are outside the page count.
- **Markup colors.** Yellow means cut, green means verify, and blue means reword or shorten. Blue on a table header means fix the column formatting.

**Key takeaways**
- Five places deliberately keep a short version of something marked yellow, because the assignment requires it or because a claim would otherwise be left without its caveat (listed below). Delete them if you disagree.
- Two items need a team decision (end of file).

## Shortening

Prose in the main text fell from about 8,000 words (v2) to about 6,000. Repetition between the abstract, introduction and conclusion is cut. Each section is tightened, Sections 6.3 to 6.5 most of all. The correlation table (v2 Table 2) is dropped from the main text, because Figure A1 shows the same matrix with values. Tables 3 to 12 are therefore renumbered 2 to 11, with every cross-reference updated. A `\clearpage` before Appendices A and B keeps the appendix figures from floating past Appendix B, a layout problem v2 also had.

## William's markup, applied on top of v2

- **Yellow (cut):**
  - the "investor who knows μ and Σ" sentence;
  - the settings-commit dates in Section 1;
  - the download date in Section 2 (kept in Appendix B);
  - the master seed in Section 2 and the TC tuning seed in 6.4 (both kept in Appendix B and Table 3);
  - "on 16 October 2026, before any run" (γ choice);
  - the mean-field and spline-bias paragraph in 6.3;
  - the "no analysis / no explanation" sentences in 6.4 (see deviation d).
- **Blue (reworded or shortened):**
  - the introduction's opening;
  - the 6.3 paragraphs;
  - the first TC design paragraph;
  - the 6.4 lead sentence;
  - the Appendix C pointer;
  - the four 6.5 paragraphs;
  - the conclusion's TC sentence;
  - the headers of Tables 1 and 2, now on one line ("SE (%)", "Vol. (%)", "Kurtosis", "U(w*)", "Gap", with units moved to the captions). v2 had already reformatted the other marked tables.
- **Green (verified):** "One hypothesis, which we have not tested" (γ = 8 FA). Correct. The explanation rests on an eigenvalue calculation recorded in `AUDIT_2026-11-06.md` §2.3 and `EXPERIMENTS.md` §8.7. No committed script produces it and no counterfactual run tested it.

## Deliberate deviations from the markup

| | Marked yellow | Kept as | Why |
|---|---|---|---|
| a | Paragraph after the budget-only formula | One sentence: agrees with an independent numerical solve within 1.5e-6 (`results/benchmarks/closed_form_budget_only_checks.json`: 1.47e-6), and is not long-only | Part II: "Derive or verify the closed-form solution" |
| b | Section 4 "Disclosures" text | Two sentences: the budget was sized with a pilot scored against the benchmark, and a 2M-episode demonstration ran before the settings were fixed | It is the disclosure behind the pre-registration claim. Section 6.5 cites it. The pre-registration wording now says "before any full-scale run" so the two agree |
| c | 6.5 "Discretization error and benchmark leakage" | Two sentences | Part VI lists "Discretization error and possible benchmark leakage" as a required discussion topic |
| d | 6.4 "why its target loses so much less at $10B ... not investigated" | Moved to the abstract and conclusion as one clause | Without it, the narrowing gap reads as learned trade timing. Most of the narrowing comes from the target choice |
| e | Download date in Section 2 | Appendix B: "downloaded on 24 September 2026" | Part I: "Record the download date" |

## Fixes from the independent review

- The conclusion had said the 25% grid has the "smallest" median total gap, which is false: `all-three` and the 10% grid are lower under longer settings. It now compares the 25% and 20% grids at the baseline budget (14.3 against 19.8).
- Qualifiers restored:
  - the abstract's step-size confound;
  - "for a long time" (seeds switching portfolios);
  - "the medians move both ways" (`n26m`);
  - the TC reward's expectation "is not exactly U of the starting weights";
  - "with a constant step" (10% grid splines);
  - "at γ = 5" on the top-ten claim.
- "Level with the myopic policy" is now "statistically indistinguishable from".
- The general Q-learning update is now a display equation; inline, it ran into the margin.
- The 6.1 sentence now names the expectation that failed at γ = 3.
- "DBC began trading the day before" is now "the day after DBC's first price in the data" (`CLEANING_LOG.json`: 2006-02-06).
- Figure 1 is now cited in Section 5.3.

## Slides and speaker notes

`slides/build_slides.py` and `slides/SPEAKER_NOTES.md` now match this draft:
- Slide 4 drops the settings-commit dates.
- Slide 10 says the ranking statistics are at γ = 5.
- Slide 12 is reframed as the gap narrowing as costs rise, with the unexplained-target caveat and the myopic comparison.
- Note 10 drops the untested "biased at the peak" explanation, which is cut from the report.
- "Favourite" is now "favorite".

The notes now cite `REPORT_final.md`. `slides.html` and `slides.pdf` are rebuilt (printed with headless Chrome on Windows).

Two slides were added afterwards, taking the deck to 15 slides plus 3 backup:
- Slide 1 is a landing slide with the title, the question, the nine ETFs by asset class, the authors and the course. The old opening slide is now slide 2, with its kicker changed to "The question".
- Slide 3, "From raw prices to a trained agent", is a diagram of the Part I pipeline and the training loop. It covers data sources, cleaning, one-time estimation of μ and Σ, the frozen simulator, the four steps of an episode, 13M episodes and 10 seeds, evaluation and the benchmarks. It is drawn in HTML and CSS inside `build_slides.py`, so it has no image file. Every number on it is from Sections 2 to 4.

The speaker notes are renumbered to match. At the times given the talk now runs about 11 minutes.

## For the team to decide

1. **E6 verdict (Table A3).** E6 registered "Agent loses to myopic at every level" and is marked Met. At $10B the shortfall (−0.5, SE 0.54, ahead on 5 of 10 seeds) is not distinguishable from zero. "Partly met" may fit better, but changing it changes the "eight met, four partly met" count in Section 6.4, the Table A3 caption, `results/tc-drift/runs/README.md` and `reports/tc-drift/`. Not changed.
2. **Promote to `REPORT.md`?** `build.sh` builds only `REPORT.md`, and Appendix B's build command describes that file. Once this draft is accepted, copy it over `REPORT.md` (v1 stays in git history) so that `sh docs/report/build.sh` reproduces the submitted PDF.

## Build

The PDF was built on Windows with pandoc 3.9 (pip `pypandoc_binary`) and MiKTeX XeLaTeX, using the same `preamble.tex`, `table_captions.lua` and `fix_tables.py` steps as `build.sh`. Figures were not regenerated: the committed PNGs were used.
