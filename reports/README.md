# Run Reports: How to Add a Run and Compare It

**Owner:** Albert · **Code:** `src/reporting/` · **Tests:** `src/tests/test_reporting.py`

## TL;DR / In plain terms

Every experiment run gets its own report, and each report automatically lines
itself up against every earlier run. A run is turned into a small data file
once, on the machine that ran it, and committed. Reports are then built only
from those committed files, so anyone can rebuild any report, and the
comparison grows as runs are added. The original hand-written report for the
first run stays untouched at `reports/full_run_2026-10-22/`.

**Key concepts to understand**

- **Run.** One `--results-dir` folder with one set of settings, holding one or
  more experiments. The first run is `baseline-v1` (folder `results/`).
- **Family.** Experiments with the same γ and grid, for example "γ = 5, 20%
  grid". The comparison section lines up a family across runs and lists every
  setting that differs (B, episodes, ε schedule, seeds, code, calibration).
- **Reference run.** `baseline-v1`, the pre-registered run. Every comparison
  shows changes against it. Later runs are chosen after seeing results, so
  they are labelled post hoc and reported as sensitivity checks.

**Key takeaways**

- The training logs (`run_*.npz`) never leave the machine that ran them, so
  **extract each run right after it finishes, on that machine.** Without
  `data.json` no later report can compare against that run.
- A difference between runs can only be pinned on a parameter if that
  parameter is the only thing that changed for the family. The report warns
  when more than one setting changed between consecutive runs.

## Layout

```
reports/
  registry.json             one entry per run, oldest first (hand-written)
  README.md                 this file
  full_run_2026-10-22/      original report for baseline-v1, kept as is
  <run_id>/
    data.json               extracted once from that run's training logs (committed)
    report.html             built from registry + every data.json up to this run (committed)
src/reporting/
  extract.py                results folder -> reports/<run_id>/data.json
  build.py                  registry + data files -> reports/<run_id>/report.html
  template.html             the report page
```

## Adding a run: the checklist

Do these in order. Steps 1 and 2 happen **before** training starts.

1. **Write the plan down first.** In `EXPERIMENTS.md`, add the run: its id,
   what changes against `baseline-v1`, the expected direction (for example
   "learning gap shrinks"), the exact command and the estimated run time.
   Commit it before running, so the expected direction is on record before
   any result exists.
2. **Set the run up on its own branch.** `run_experiments` has flags only for
   the episode count, seeds, evaluation episodes, log interval and `--only`.
   Everything else is a constant in `src/config.py`: `EPISODE_LENGTH` for B,
   `EPSILON_AT_HALFWAY` for the ε schedule, and `experiments()` for the grid,
   γ values and experiment names. Edit them on the run's branch with a
   recorded reason (the docstring at the top of `src/config.py` asks for one),
   and give every experiment a name that says what differs, for example
   `baseline_g5_grid20_B60`, so names never collide across runs. Then pick a
   short run id, for example `b60-v1`, and train into its own folder:
   ```bash
   python -m src.run_experiments --results-dir results/b60-v1
   ```
3. **Tag the code** that produced the run, for example `git tag b60-v1`, on the commit you trained from. (For `baseline-v1` the tag sits on a later commit with behaviour-equivalent edits; the training commit `80a6e9c` is in each manifest and in the report header.)
4. **Register the run** in `reports/registry.json`, appending at the end:
   ```json
   {
     "id": "b60-v1",
     "title": "Longer episodes (B = 60)",
     "date": "2026-11-02",
     "results_dir": "results/b60-v1",
     "git_ref": "b60-v1",
     "scale": "full",
     "pre_registered": false,
     "what_changed": "Episode length B = 20 -> 60 for all γ on the 20% grid; everything else as baseline-v1.",
     "notes": {
       "tldr": ["Two or three plain sentences, written after reading the results."],
       "takeaways": ["What this means for the project."],
       "caveats": ["Anything a reader must know, e.g. a seed that failed and was re-run."]
     }
   }
   ```
   `pre_registered` is required: `true` only when the settings were fixed
   before any results were seen. Notes are inserted into the page as HTML, so
   treat the registry as trusted, team-written input.
   Optional fields: `"experiments": [...]` to include only some experiment
   folders, and `"compare_with": [...]` to compare against chosen runs instead
   of every earlier one. Use `"scale": "reduced"` for smoke tests; the report
   then carries a warning banner.
5. **Extract, on the machine that ran it:**
   ```bash
   python -m src.reporting.extract b60-v1
   ```
   It refuses if a seed is missing or if the calibration files differ from
   the ones the run was trained on (pass `--processed-dir` for a run that used
   a different calibration on purpose).
6. **Build the report:**
   ```bash
   python -m src.reporting.build b60-v1
   ```
   `python -m src.reporting.build --all` rebuilds every registered run, for
   example after a template change. Earlier reports stay comparable to the
   runs before them, because each report only layers in runs registered
   before it.
7. **Write the notes.** Read the report, fill in `notes.tldr`, `takeaways` and
   `caveats` in the registry, and rebuild. The numbers are generated; only the
   interpretation is written by hand.
8. **Review and commit.** Independent review (CLAUDE.md §2.6), then commit
   `results/<run_id>/` small files, `reports/<run_id>/` and the registry on a
   feature branch and open a PR to `dev`.

## What each report contains

1. **Summary.** The registry's TL;DR and takeaways, and a headline table per
   experiment.
2. **Run settings.** Every parameter of every experiment, read from the
   manifests. Cells that differ from the same family in the reference run are
   highlighted with the old value.
3. **Results by experiment.** Gaps, exact-best share, every convergence
   measure (the pre-registered rule beside the post-hoc "settled" measure and
   both near-optimal readings), regret; gap decomposition and per-seed dots.
4. **Each experiment, seed by seed.** Picker across experiments; every seed's
   greedy loss over time, learning curves, share on the final portfolio,
   excursions, and a per-seed table.
5. **Comparison with earlier runs.** The run list with what changed; a matrix
   of every family across runs; and for one family at a time: a "what changed"
   table, a metrics table with changes against the reference, overlaid loss
   and learning curves (x-axis switchable between simulated days, episodes and
   share of budget, because B and the budget change between runs), per-seed
   gaps, gap decomposition, convergence timing, and a seed-by-seed table when
   runs share a master seed.
6. **Notes, caveats and definitions.**

## Notes for the planned runs

- **Longer B.** Compare by simulated days and by episodes: a run with B = 60
  sees three times the data per episode. The report shows the settled point in
  both units. The exact reward SD falls roughly with √B, so "tries to separate
  the top two" falls roughly with B; the settings table shows it per run.
- **More episodes.** With `EPSILON_AT_HALFWAY` unchanged, β halves when the
  budget doubles, so random tries per portfolio double and K doubles. The
  settings table shows those derived values, and the comparison does not
  count them as separate changes.
- **ε schedule.** Changing `EPSILON_AT_HALFWAY` changes β, ε at the end and
  random tries per portfolio. William's surrogate study
  (`REVIEW_HISTORY.md` §7.2) expects little change in the learning gap and
  more regret; write that expectation into `EXPERIMENTS.md` first.
- **Several changes at once.** A combined run can show the joint effect but
  cannot attribute it. The report flags it; compare it with the single-change
  runs in the same family to attribute.
- **10% grid.** A new family (γ, 10% grid), so its first appearance has no
  earlier entry to compare with; the family matrix still places it beside the
  coarser grids, but the comparison section does not line it up against the
  20% grid. Run `grid10-robustness` (2026-10-29) used the default log
  interval: 2,600 checkpoints × 43,758 Q-values is 113.8M, under the 120M
  snapshot guard (`MAX_SNAPSHOT_VALUES` in `src/train.py`). Its paired
  comparison with `all-three` is in
  `results/grid10-robustness/comparison_g{3,5,8}_grid20_vs_grid10/`.
- **Fat-tailed returns (Part VI option).** The extractor's reward SD assumes
  Gaussian draws, and nothing in the manifest records the shock distribution.
  Before such a run, record the distribution in the manifest and gate
  `reward_sd` on it.
- **Size.** Each `data.json` is about 0.7 MB for four experiments; earlier runs
  are layered in without per-seed time series, so a report grows by roughly
  0.3 MB per earlier run.

## Assumptions

- Experiments within one run share one settings file (`run_experiments.py`
  enforces this), and every seed in an experiment has the same checkpoint grid.
- Families match on γ and grid step only. Two experiments in the same run with
  the same γ and grid but different B would both appear in that family.
- Gaps use the calibration each run was trained on; the extractor checks the
  hashes in every manifest.
