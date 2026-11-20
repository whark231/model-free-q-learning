# Function Approximation: Integration Notes for Albert

**From:** William (with Claude) · **Date:** 2026-11-03, updated 2026-11-06 · **Branch:** merged into `dev` (PR #23)
**Status (2026-11-06, updated 2026-11-20):**
- **Built:** agent, features, the `train.py` path and the Q-value fit metrics, all tested and independently reviewed.
- **Tuned:** the step size, on tuning seeds (§5).
- **Run:** the three constant-learning-rate runs `tabular-const`, `fa-quadratic` and `fa-spline`, 30 seeds each at commit `bc311a3`, in `dev` (PRs #24 and #25; `fa-spline` in Albert's commit `9355610`). Outcomes and the registered expectations are in `EXPERIMENTS.md` §8.7; the reports are in `reports/<run_id>/`.
- **Run on the 10% grid (2026-11-11, Albert):** `fa-spline-grid10-eps3` and `fa-spline-grid10-eps10` (`EXPERIMENTS.md` §8.8). Nothing in this file is open.

Everything here is post hoc relative to `baseline-v1`. The sections below are the design record and were written before the runs; where a result later contradicted a prediction, `EXPERIMENTS.md` §8.7 says so. Open issues from the 2026-11-06 audit are in `AUDIT_2026-11-06.md`.

---

## Albert's review fix (2026-11-04)

- **Fixed the merge blocker:** exact ties and float32 near-ties could leave training and evaluation reporting different final portfolios, then crash report extraction. Refresh alignment alone did not prevent this.
- **One final-policy convention:** FA `final_greedy`, manifest weights, evaluation and reports now use the lowest-index argmax of the final float64 Q-values. Cached training choices and random streams are preserved; the manifest explains the distinction.
- **Honest convergence reporting:** `settled_from()` returns `None` when the final policy did not settle in the checkpoints. Evaluation and reports do not claim final-policy convergence when the cached history ends on another portfolio. The report omits the missing settlement marker and has no post-settlement excursions to count.
- **Regression coverage:** a real-calibration seed reproduces the original crash end to end; controlled exact ties and float32 near-ties check the final policy and preserved history; settlement tests cover missing and empty histories.
- **Validation:** all 366 tests pass. An independent two-seed tabular comparison against `dev` produced identical saved arrays and configurations.
- **Scope:** changes are in `train.py`, `evaluate.py`, reporting extraction/template and tests. No tuning, full-scale FA runs or changes to the agent update rule are included. The integration work in §2 was still outstanding then; it was done before the runs (merged in PR #23).

---

## TL;DR / In plain terms

The function-approximation (FA) agent learns a short parameter vector θ instead of one Q-value per portfolio, so a reward from one portfolio also informs similar portfolios. To run it, I changed four of your files (listed in §1) so that tabular runs behave and record exactly as before. Before any FA run, `config.py`, `run_experiments.py` and the report extractor need small additions (§2), and α must be tuned on separate seeds (§5). A rough calculation (§4) says a constant α may be too noisy or too slow at a 13M-episode budget; the tuning sweep is designed to find out rather than assume.

**Key concepts to understand**

- **Features and θ.** Each portfolio a gets a row of features F[a], built from its weights by `src/features.py`. The agent's estimate is Q̂(a) = θ · F[a]. After each reward it nudges θ toward the features of the portfolio it tried: θ ← θ + α[(r − Q̂(a))·F[a] − l2·θ]. Every portfolio's Q̂ moves at once, which is the point of FA and the reason the runner needs a separate path.
- **The model ladder.** Tabular (existing baseline) → **splines, the main model** (additive plus pairwise in the nine weights, never told the reward is quadratic) → **quadratic, a correctly specified reference** (handed the true functional form). The quadratic is never the discovery result.
- **Constant α trades noise for speed.** A large α learns fast but keeps jittering with reward noise forever; a small α averages the noise away but learns slowly. The tabular agent's α = 1/N avoids this by averaging more as visits grow.

**Key takeaways**

- Tabular behaviour and records are unchanged: all 92 existing runner, config and evaluation tests pass, including the check against the committed `baseline-v1` manifests. The independent review also re-ran a tabular run with the old and new code (all arrays and configs identical) and confirmed that all 19 committed experiment folders still count as finished.
- FA runs use the baseline settings (ρ = 0, B = 20, same ε schedule, 13M episodes, master seed 20261016 and stream layout), so each FA seed sees exactly the returns of the matching baseline seed. If B or ε ever change for FA, the tabular baseline must be rerun at those settings too.

---

## 1. Changes already made to your files

All on `feature/function-approximation`. Please review the PR before it merges into `dev`.

| File | Change | Why |
|---|---|---|
| `src/train.py` | `TrainConfig` gains `agent` (`"tabular"` or `"linear"`), `feature_map` (`"quadratic"` or `"spline"`), `l2`, `greedy_refresh_every`. Defaults mean tabular; a tabular config with any FA setting is rejected, and a linear config needs a feature map, a constant `alpha_rule`, and `discount = q_init = 0` | Record which agent ran, refuse silently ignored settings |
| `src/train.py` | `config_record(config)`: the config as written to `manifest.json` and each `.npz`. Tabular runs omit the four FA keys | Tabular manifests stay identical to the `baseline-v1` ones, so existing folders are not seen as "different settings" |
| `src/train.py` | `build_features(config, grid)` and `FEATURE_MAPS`; `run_experiment` builds F once per experiment from the grid and passes it to `default_agent_factory(..., features=F)` | The agent gets F, never the grid, μ or Σ |
| `src/train.py` | `train_one_run` hands agents with `updates_all_actions = True` to a new `train_one_run_fa`. The tabular loop is untouched | See §1.1 |
| `src/train.py` | `CODE_FILES` plus `src/fa_agent.py`, `src/features.py` hashed for FA runs only; FA manifests also record `n_features` and FA log-schema notes; FA `.npz` files add `final_theta` | Provenance |
| `src/run_experiments.py` | `training_status` compares against `config_record(...)` instead of `asdict(...)`; unused `asdict` import removed | Same keys as old manifests |
| `src/tests/test_config.py` | The `baseline-v1` match test uses `config_record` | Same reason |
| `src/train.py` | Linear runs need `n_episodes` and `log_every` to be multiples of `greedy_refresh_every`; an end-of-run check raises if the last logged greedy is not the best final Q-value (beyond float32 rounding); `run_experiment` refuses a factory whose agent does not match `config.agent` | Found by the independent review: without them, 6 of 16 small runs ended on a stale greedy, which makes `evaluate.py`'s learned portfolio disagree with the logged one and crashes `reporting/extract.py`'s `settled_from` |
| `src/run_experiments.py` | `training_status` compares every key on either side, a missing FA key meaning its tabular default | A tabular config could otherwise match an FA folder with the same constant α |
| `src/evaluate.py` (2026-11-04) | New `q_fit()` and `TOP_K = 10`; each seed gains `q_fit_all`, `q_fit_visited` and `optimism_of_learned_bps`; the summary gains a `q_fit` block and a note; `per_seed.csv` gains nine columns at the end; `comparison.json` carries each resolution's `q_fit` block. Nothing existing changed | The §6 metrics, requested by William. Re-evaluating `b63` and `eps-decay` left every earlier number and both other CSVs identical |
| `src/tests/test_evaluate.py` (2026-11-04) | Six new tests (hand examples, scipy cross-check with ties, edge cases, an oracle agent, a shifted oracle, optimism rebuilt from logged rewards) and the new CSV columns checked in the existing CSV test | |
| `EVALUATION.md` (2026-11-04) | Q-value fit definitions in §2, two decisions in §3, two verification rows in §4 | |
| `src/config.py` (2026-11-05) | `Experiment` and `RunVariant` gain `alpha_rule`, `agent`, `feature_map`, `l2` and `greedy_refresh_every`, defaulting to the baseline's. `changes()` reads them from the new `BASELINE_RUN_SETTINGS`. `train_config()` passes them on. New constants `TUNED_ALPHA_TABULAR`, `TUNED_ALPHA_QUADRATIC`, `TUNED_ALPHA_SPLINE` and `FA_GREEDY_REFRESH`. Three new `POST_HOC_RUNS` entries | Registering the runs (§2). `baseline-v1` and the five earlier variants produce exactly the configs they did before (tested) |
| `src/run_experiments.py` (2026-11-05) | `batch_settings` records `alpha_rule` for every variant, and the four agent fields only for FA variants. The dry run and `INDEX.md` show the agent and α; `INDEX.md` points to `EXPERIMENTS.md` §7 and §8 | The earlier variants' committed `settings.json` files still match (new test); FA and tabular folders cannot mix |
| `src/reporting/extract.py`, `build.py`, `template.html` (2026-11-05) | `params` gains `agent`, `feature_map`, `l2` and `greedy_refresh_every`; tabular manifests read as tabular. `COMPARED_PARAMS` lists them. `load_run_data` gives older `data.json` files, which lack these fields, the tabular defaults, so a comparison does not report agent changes that never happened (found in review). The template shows a numeric α or l2 in scientific notation, where it showed 3.2e-4 as "0.000" | The settings table and run comparisons show what changed |
| `src/tests/test_config.py`, `src/tests/test_reporting.py` (2026-11-05) | Each new run's values checked against `results/tuning/fa_tuning.json`; earlier variants' settings still match; an FA variant runs end to end through `run_all`; the extractor records the agent | |
| `EXPERIMENTS.md` (2026-11-05) | New §8: tuning summary, shared settings, one subsection per run with its expected direction, commands and run time; status line updated | Registration before running, as in §7 |
| `src/tests/test_train.py`, `src/tests/test_config.py` | FA tests in `test_train.py` (rewards rebuilt independently; logged Q and θ consistent with each other; greedy changes only at refresh multiples; final policy matches the float64 argmax while cached tie choices are preserved; seed reproducibility; tabular record unchanged; wrong agent type and inconsistent settings rejected) and one in `test_config.py` (a tabular config never matches an FA folder) | |

### 1.1 What `train_one_run_fa` logs differently

Same episodes, same random streams, same log arrays, so `evaluate.py` reads FA runs unchanged (checked end to end on a 30,000-episode smoke run of both models). Three quantities mean something slightly different:

- **Logged greedy** is the agent's cached greedy (float32 scores, random tie-break), refreshed every `greedy_refresh_every` updates. "Greedy unchanged for K episodes" is exact to that resolution.
- **|ΔQ|** is the change in the chosen portfolio's Q̂, the same definition as tabular. Other portfolios' Q̂ also move and are not in it.
- **Full Q-values** are read only at checkpoints and at the end (exact float64). Every run and checkpoint ends on a refresh; the run raises if the cached greedy's score falls below the maximum beyond float32 tolerance. Exact ties and near-ties can still produce a different cached choice. The final reported policy (`final_greedy`, manifest, evaluation and reports) always uses the lowest-index argmax of `final_q`; checkpoint and change histories retain the actual cached choices.
- **Fixed by Albert, 2026-11-04:** `settled_from()` now returns `None` if the final checkpoint does not select the reported final policy, including on older FA logs. Reports handle missing settlement without crashing or drawing a marker at zero; evaluation and reports withhold final-policy convergence when the cached history ends on another action.

Two more deliberate differences from tabular, for the report: FA uses a constant α where the baseline used α = 1/N(a) (with one-hot features and constant α both give the same Q-values, `src/tests/test_fa_agent.py`), and FA exploits a greedy up to `greedy_refresh_every − 1` updates old.

## 2. Integration for the runs (done 2026-11-05)

The list that stood here before tuning is now done, as described in §1:
- the `RunVariant` fields;
- the names `fa-quadratic` and `fa-spline` (suffixes `FAquad` and `FAspline`), plus `tabular-const` (suffix `TABconst`) for the tabular comparison;
- `batch_settings`;
- the extractor's `params`;
- `EXPERIMENTS.md` §8.

The `train.py` CLI still has no FA flags. None are needed, since the runs go through `run_experiments`.

Still open, and not blocking:
- ~~**A 10% grid FA run** (`fa-spline-grid10`). Not registered; decide after the 20% grid results.~~ Done 2026-11-11: registered and trained as `fa-spline-grid10-eps3` and `fa-spline-grid10-eps10` (α = 2.78e-5, noise-matched from `fa-spline`; `EXPERIMENTS.md` §8.8). Both have larger learning gaps than the tabular `grid10-robustness` at every γ, and ε 10% did not shrink the gap.
- ~~**The Windows print crash** in `run_experiments`.~~ Fixed 2026-11-09 (`AUDIT_2026-11-06.md` M1).

## 3. Speed and the greedy refresh interval

`python -m src.bench_fa --with-runner` on William's machine, 2026-11-03. Absolute times swing by 3× with machine load, so read the runner rows relative to tabular from the same run.

| Grid | Runner (refresh 100), s per 100k episodes | Relative to tabular |
|---|---|---|
| 20% | tabular 5.2, quadratic 12.4, spline 17.8 | 1 : 2.4 : 3.4 |
| 10% | tabular 25.2, quadratic 13.4, spline 29.0 | 1 : 0.53 : 1.15 |

The agent alone (fake rewards): refreshing every episode costs the spline model 119 min per 13M episodes on the 20% grid and about 25 hours on the 10% grid; refresh 100 cuts these to 15 and 32 minutes; refresh 1,000 saves little more. **Recommendation: `greedy_refresh_every = 100` for every FA run.** The greedy is then at most 99 updates old (under 0.001% of the budget). Scaled from `baseline-v1`'s 300 s per seed on your machine: about 12 min per seed (quadratic) and 17 min (spline) on the 20% grid, so 6 and 8.5 hours for 30 seeds.

## 4. Step size: stability bound and the noise-speed trade-off

Constant-α SGD is stable only for α < 2 / (max‖φ‖² + l2). Measured on the grids: max‖φ‖² = 1 for the quadratic and 45 for the splines, so the bounds are about 2 and 0.044.

Rough design calculation on the 20% grid, assuming uniform exploration (late training concentrates on a few portfolios, so treat this as order of magnitude). Q̂ noise RMS ≈ σ·√(α·tr E[φφᵀ]/2), with σ = 1.55e-3 the reward SD at B = 20 (an agent-observable quantity: the SD of its own rewards). Learning time of a direction with eigenvalue λ ≈ 1/(α·λ).

| Fraction of bound | Quadratic: Q̂ noise (bps/yr) | Quadratic: slowest direction (episodes) | Spline: Q̂ noise (bps/yr) | Spline: median direction (episodes) |
|---|---|---|---|---|
| 0.5 | 943 | 1.8e3 | 2,192 | 1.2e5 |
| 0.1 | 422 | 8.8e3 | 980 | 6.1e5 |
| 0.02 | 189 | 4.4e4 | 438 | 3.1e6 |
| 0.004 | 84 | 2.2e5 | 196 | 1.5e7 |
| 8e-4 | 38 | 1.1e6 | 88 | 7.7e7 |
| 1.6e-4 | 17 | 5.5e6 | 39 | 3.8e8 |
| 3.2e-5 | 7.5 | 2.8e7 | 18 | 1.9e9 |
| 6.4e-6 | 3.4 | 1.4e8 | 7.8 | 9.6e9 |

The gaps that separate the tabular agent's candidates are about 5 to 18 bps/yr (`REVIEW_HISTORY.md` §7.1, §8; known only because the baseline was evaluated, so used here only to set a wide range, never to pick α). For the quadratic, noise falls to that level only where learning takes 5M to 28M episodes, about the budget. For the splines, most directions would take 10⁸ episodes or more at that noise level. The spline E[φφᵀ] has nonzero eigenvalues from 13.2 down to 3.4e-6 (ratio 3.9e6) on the 20% grid; on the 10% grid the smallest nonzero one is 3.9e-11.

**Proposed sweep.** α as fractions of the bound: 0.5, 0.1, 0.02, 0.004 (the original proposal), extended to 8e-4, 1.6e-4, 3.2e-5 and 6.4e-6, because the calculation says the useful range may lie below 0.004. Spline l2: 0, 1e-5, 1e-4, 1e-3, 1e-2 (directions with eigenvalue below l2 are shrunk toward 0 and learn at rate α·l2 instead of α·λ). Equal fractions are not equal effective steps across models (α·E‖φ‖² differs by about 5×), so compare each model at its own best α, not at matched fractions.

If no α in the sweep beats the noise within the budget, that is a finding to report, and the standard remedy is a decaying step (α_t = α₀/(1 + t/t₀)), which would be a change from the constant α the course suggested. Decide that with the professor rather than in tuning.

## 5. Tuning script (`src/tune_fa.py`, run 2026-11-04 to 2026-11-05)

**Outcome:**
- **Chosen values:**
  - tabular α = 3.2e-4;
  - quadratic α = 3.2e-4;
  - spline α = 7.11e-6 with l2 = 0.

  Each is fraction 1.6e-4 of its model's bound, and in each model it beats both neighbouring fractions.
- **Spline close call.** The spline choice was a close call. l2 = 1e-4 led on seeds 0 to 2, and l2 = 0 won over seeds 0 to 5. A rounding bug first hid the close call; it was fixed in `945a55e` before the extra seeds ran.
- **Where it is recorded:** `results/tuning/fa_tuning.json`, `run.log` and `extend.log`, and `EXPERIMENTS.md` §8.1. These are tuning-seed scores, not results.

**In plain terms:** each model's constant α is chosen on three separate tuning seeds by training in full and keeping the α whose final portfolio earns the most realized reward on fresh draws. The tabular agent is tuned the same way, for a post-hoc constant-α tabular run (`tabular-const`) that differs from the FA runs only in its features.

- **Seeds.** Tuning master seed **20261103**, 3 seeds, never 20261016 (the 10 reported seeds). Each tuning seed is trained exactly as `run_experiment` would train run *i* of the same config (tested), so tuning uses the real training path.
- **Settings.** γ = 5, 20% grid, B = 20, 13M episodes, baseline ε schedule, `greedy_refresh_every = 100` for FA. A shorter budget would bias tuning toward large α, since learning time is the binding side of the trade-off. The chosen values are then used unchanged at γ = 3 and 8 and stated as tuned at γ = 5 only.
- **Candidates.** α as fractions 0.5, 0.1, 0.02, 0.004, 8e-4, 1.6e-4, 3.2e-5, 6.4e-6 of each model's stability bound 2 / max‖φ‖², computed from the grid's features: 2 for tabular and quadratic, 2/45 for splines.
  - **Pass 1:** all three models at l2 = 0, plus the tabular α = 1/N(a) agent as a reference row that is never chosen. That makes 25 candidates × 3 seeds.
  - **Pass 2:** splines only, l2 ∈ {1e-5, 1e-4, 1e-3, 1e-2} at the two best pass-1 fractions. That makes 8 candidates × 3 seeds; l2 = 0 is reused from pass 1.
- **Criterion: held-out validation by simulation.** For each candidate, the realized reward of each seed's final greedy portfolio, averaged over the seeds. It is measured on 200,000 evaluation episodes from the tuning plan's evaluation stream, the same draws for every candidate, so differences are paired.
  - Returns are simulated from μ and Σ, as in training. No closed-form objective, benchmark or optimal weight is computed, and the reported seeds and their evaluation draws are never touched.
  - The criterion is an unbiased estimate of each final portfolio's objective, which is what a validation score should be. The agent could not compute it itself: it scores every portfolio on the same full return draws, and the agent sees only one reward per episode.
  - A test checks that the script's own imports exclude the benchmark and evaluation modules. Those modules are still loaded, because `src/config.py` imports `evaluate.py` for `EvalConfig`, but nothing from them is called.
- **Choosing (rule fixed by William on 2026-11-04, before any full-scale result).** Each model takes its highest-criterion candidate over seeds 0–2, unless that is a close call: the runner-up's shortfall is at most one seed-paired standard error.
  - In a close call, the model's top 3 are trained on 3 more seeds (3–5), and the winner is the best of those 3 over all 6 seeds. `SeedPlan` streams do not depend on the number of runs, so seeds 0–2 are the same either way (tested).
  - An exact tie (the same set of final portfolios across the seeds) counts as a close call. If it is still a tie after 6 seeds, the smaller l2 wins, then the larger α, and the ties are recorded. Pass 2 breaks ties the same way.
- **Judging close calls.** All candidates share the same training streams for a given seed index, so the table pairs every candidate with its model's best seed by seed. It shows:
  - each seed's reward;
  - the shortfall's standard error over the 3 training seeds ("seed SE");
  - on how many seeds the candidate did at least as well ("seeds >=").

  The evaluation-episode SE ("eval SE") is much smaller. It shows only that the final portfolios were scored precisely, not that the choice is right. The independent review estimated that one seed landing on the second-best grid portfolio moves the criterion by about 3.5 bps/yr, against an eval SE near 0.25.
- **Resume and safety.**
  - One JSON record per (candidate, seed) goes to `results/tuning/records/` the moment it finishes, so a rerun skips finished work.
  - A record written with other settings (for example a smoke test left in the same folder) is refused rather than reused, and the summary refuses records from mixed budgets.
  - A failed run is reported while the others continue, and it has no record, so a rerun retries it. Ctrl+C cancels the queued runs.
  - Commit the code before the full run, so each record's `git_commit` contains the code that wrote it.
- **Output.** `results/tuning/fa_tuning.json` holds the scored table, the budget, the git commits, the diverged and incomplete candidates, and the choice. The chosen values go into `config.py` with a pointer to it (§2).
- **Commands.**
  ```
  python -m src.tune_fa --all --jobs 4      # pass 1, pass 2, extra seeds, summary
  python -m src.tune_fa --summarize         # reprint the table from the records
  ```
  The steps can also be run one at a time with `--pass 1`, `--pass 2` and `--extend`.
- **Cost estimate (not measured at full scale).** These are the §3 per-run times on William's machine: tabular about 5 min, quadratic 12 and splines 17 per 13M episodes. Pass 1 is 3 × (9 × 5 + 8 × 12 + 8 × 17) ≈ 830 min ≈ 14 hours in one process. Pass 2 is 24 × 17 ≈ 6.8 hours. Extra seeds, if all three models are close calls, add 3 × 3 × (5 + 12 + 17) ≈ 300 min. `--jobs 4` on an 8-thread machine should cut the wall time to roughly a quarter if the runs don't slow each other down. I have not checked that.
- **Smoke test, 2026-11-04.** Both passes and the summary ran end to end through the CLI at 20,000 episodes, with 3 and with 4 workers; pass 1 took 52 s with 3. Reusing that folder at another budget was refused. The numbers from those runs say nothing about the right α at 13M and are not reported.
- **Review, 2026-11-04.** The independent review confirmed the seeding (each tuning seed matches `run_experiment`), the criterion and standard-error math, and the absence of leakage. It found eight issues, all fixed before commit:
  - seed noise was not reported;
  - records were reused across budgets;
  - pass 2 broke ties by file order;
  - a diverged spline seed blocked pass 2;
  - the leakage wording overclaimed;
  - pool failures stalled silently;
  - temp-file names could collide;
  - evaluation chunks used too much memory.

  Six of its planted bugs had slipped past the tests; tests now catch them. The smoke test then showed the budget guard could be bypassed through the run queue, which is also fixed and tested.

## 6. Suggested new metrics (evaluation only)

Computed after training from μ and Σ, like the near-optimal measure, and never fed back:

- **Fitted-vs-true Q correlation**: Pearson and Spearman correlation across all portfolios between the final Q̂ and the true objective w'μ − (γ/2)w'Σw. Says whether the model learned the landscape or only got lucky on the top pick.
- **Top-k recall**: how many of the true top 10 portfolios are in the learned top 10.
- **Calibration of the winner**: Q̂ of the learned portfolio minus its true objective (optimism from picking the maximum of noisy estimates).
- For the tabular baseline the same metrics can be computed from `final_q`, which gives a like-for-like comparison.

Implemented in `src/evaluate.py` on 2026-11-04 (see §1 and `EVALUATION.md` §2). Top-k uses k = 10, and both "all portfolios" and "visited portfolios" are reported, because a tabular agent's untried portfolios keep Q̂ = 0.
