# Project documents

**TL;DR / In plain terms:** the project's write-ups live here; the root
`README.md`, `CLAUDE.md` and `reports/README.md` stay where they are. Start
with the prospectus for the plan, `methods/` for how each part works,
`reviews/` for what was checked and fixed, and `notes/` for each person's open
items.

The documents refer to each other by file name (for example "`EXPERIMENTS.md`
§8.7"). Every document name in this folder is unique, so the table below says
where each one is. A bare `README.md` in these documents means the root README.
Committed results and reports also cite documents by file name; those citations
predate this folder (moved 2026-11-09) and are left unchanged.

| File | Covers | From |
|---|---|---|
| `PROJECT_PROSPECTUS.md` | Plan, task ownership, timeline, rubric coverage | William |
| `methods/DATA_CLEANING.md` | Part I data pipeline, decisions, assumptions | Albert |
| `methods/ENVIRONMENT.md` | Part I simulator: interface, seeds, leakage guard, verification | Albert |
| `methods/EVALUATION.md` | Parts IV and V evaluation: gap and convergence definitions, outputs | Albert |
| `methods/EXPERIMENTS.md` | Fixed experiment settings, every registered run, how to run them, outcomes | Albert |
| `methods/FA_INTEGRATION_NOTES.md` | Function-approximation extension: design, edits to Albert's files, tuning | William |
| `methods/TRANSACTION_COSTS.md` | Transaction-cost extension: model, benchmarks, run configurations | Albert |
| `methods/TC_DRIFT_DESIGN.md` | Transaction costs with drift: the redesign, oracle, agent, interfaces, sweep and risks (Phase 1) | William and Albert |
| `reviews/AUDIT_2026-11-06.md` | Whole-project audit: open issues, verified checks, candidate final runs | William (with Claude), for both |
| `reviews/RUN_ISSUES_2026-10-22.md` | Issues found in the first full-scale run and their fix plan | Claude, for Albert |
| `reviews/REVIEW_HISTORY.md` | Closed review issues, settled decisions, rejected ideas | Both |
| `notes/REVIEW_NOTES_FOR_WILLIAM.md` | William's open items and next steps | Albert, kept up by William |
| `notes/NEXT_STEPS_FOR_ALBERT.md` | Albert's open items and next steps | William |

`reports/README.md` (adding a run and building its report) stays beside the
reports it describes.
