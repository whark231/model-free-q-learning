# Function Approximation: Integration Notes for Albert

**From:** William (with Claude) · **Date:** 2026-11-03 · **Branch:** `feature/function-approximation`
**Status:** agent, features and the `train.py` path are built, tested and independently reviewed; nothing has been trained at full scale, and no FA result exists yet. Everything here is post hoc relative to `baseline-v1`.

---

## Albert's review fix (2026-11-04)

- **Fixed the merge blocker:** exact ties and float32 near-ties could leave training and evaluation reporting different final portfolios, then crash report extraction. Refresh alignment alone did not prevent this.
- **One final-policy convention:** FA `final_greedy`, manifest weights, evaluation and reports now use the lowest-index argmax of the final float64 Q-values. Cached training choices and random streams are preserved; the manifest explains the distinction.
- **Honest convergence reporting:** `settled_from()` returns `None` when the final policy did not settle in the checkpoints. Evaluation and reports do not claim final-policy convergence when the cached history ends on another portfolio. The report omits the missing settlement marker and has no post-settlement excursions to count.
- **Regression coverage:** a real-calibration seed reproduces the original crash end to end; controlled exact ties and float32 near-ties check the final policy and preserved history; settlement tests cover missing and empty histories.
- **Validation:** all 366 tests pass. An independent two-seed tabular comparison against `dev` produced identical saved arrays and configurations.
- **Scope:** changes are in `train.py`, `evaluate.py`, reporting extraction/template and tests. No tuning, full-scale FA runs or changes to the agent update rule are included. The integration work in §2 is still outstanding.

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
| `src/tests/test_train.py`, `src/tests/test_config.py` | FA tests in `test_train.py` (rewards rebuilt independently; logged Q and θ consistent with each other; greedy changes only at refresh multiples; final policy matches the float64 argmax while cached tie choices are preserved; seed reproducibility; tabular record unchanged; wrong agent type and inconsistent settings rejected) and one in `test_config.py` (a tabular config never matches an FA folder) | |

### 1.1 What `train_one_run_fa` logs differently

Same episodes, same random streams, same log arrays, so `evaluate.py` reads FA runs unchanged (checked end to end on a 30,000-episode smoke run of both models). Three quantities mean something slightly different:

- **Logged greedy** is the agent's cached greedy (float32 scores, random tie-break), refreshed every `greedy_refresh_every` updates. "Greedy unchanged for K episodes" is exact to that resolution.
- **|ΔQ|** is the change in the chosen portfolio's Q̂, the same definition as tabular. Other portfolios' Q̂ also move and are not in it.
- **Full Q-values** are read only at checkpoints and at the end (exact float64). Every run and checkpoint ends on a refresh; the run raises if the cached greedy's score falls below the maximum beyond float32 tolerance. Exact ties and near-ties can still produce a different cached choice. The final reported policy (`final_greedy`, manifest, evaluation and reports) always uses the lowest-index argmax of `final_q`; checkpoint and change histories retain the actual cached choices.
- **Fixed by Albert, 2026-11-04:** `settled_from()` now returns `None` if the final checkpoint does not select the reported final policy, including on older FA logs. Reports handle missing settlement without crashing or drawing a marker at zero; evaluation and reports withhold final-policy convergence when the cached history ends on another action.

Two more deliberate differences from tabular, for the report: FA uses a constant α where the baseline used α = 1/N(a) (with one-hot features and constant α both give the same Q-values, `src/tests/test_fa_agent.py`), and FA exploits a greedy up to `greedy_refresh_every − 1` updates old.

## 2. Changes still needed before an FA run

Not made yet, because they depend on α and l2, which come from tuning (§5).

1. **`src/config.py`.** Add to `RunVariant`: `agent="tabular"`, `feature_map=None`, `alpha=None`, `l2=0.0`, `greedy_refresh_every=1`; include them in `changes()`; pass them through `experiments()` and `train_config()` (`alpha_rule=run.alpha` for linear runs). Register the FA runs in `POST_HOC_RUNS` with their expected direction before running, as for the other post-hoc runs.
2. **Names.** Run ids `fa-spline` and `fa-quadratic` (20% grid), later `fa-spline-grid10`. Experiment suffixes `FAspline` and `FAquad`, for example `baseline_g5_grid20_FAspline`, folders `results/<run_id>/`.
3. **`src/run_experiments.py`.** `batch_settings` should add `agent`, `feature_map`, `alpha_rule`, `l2` and `greedy_refresh_every` for FA runs, so a results folder cannot mix agents. Also the Windows `γ` print crash is still open (`NEXT_STEPS_FOR_ALBERT.md`).
4. **`src/reporting/extract.py`.** Add the agent fields to the settings table (`params`). `random_tries_per_portfolio` stays valid; `alpha_rule` will be a number for FA runs.
5. **`src/train.py` CLI** (`main`) has no FA flags. Not needed if FA runs go through `run_experiments`.
6. **`EXPERIMENTS.md`.** A §8 for the FA runs: settings, expected direction, command, run time.

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

## 5. Tuning script design (not written, not run)

`src/tune_fa.py`, run once per model before the FA runs are registered.

- **Seeds.** Tuning master seed **20261103**, 3 seeds, never 20261016 (the 10 reported seeds). `SeedPlan(20261103, 3)` gives separate returns, agent and evaluation streams, so no reported draw is used for tuning.
- **Settings.** γ = 5, 20% grid, B = 20, 13M episodes, baseline ε schedule, `greedy_refresh_every = 100`. A shorter budget would bias tuning toward large α, since learning time is the binding side of the trade-off. The chosen α and l2 are then used unchanged at γ = 3 and 8 and stated as tuned at γ = 5 only.
- **Grid.** Pass 1: the 8 α values at l2 = 0, both models. Pass 2, splines only: the 5 l2 values at the best two α from pass 1.
- **Criterion, agent-observable.** For each candidate and seed, the mean realized reward of the final greedy portfolio over 200,000 fresh evaluation episodes from the tuning plan's evaluation stream, with the same draws for every candidate (common random numbers: differences between candidates are paired, which cuts their noise by orders of magnitude for close portfolios). Pick the α (and l2) with the highest mean over the 3 seeds. No μ, Σ or benchmark enters the choice.
- **Output.** `results/tuning/fa_tuning.json` with every candidate, seed, criterion value and the code commit; the chosen values copied into `config.py` with a comment pointing to it.
- **Cost estimate.** Pass 1: 8 × 3 × (12 + 17) min ≈ 11.6 hours; pass 2: 10 × 3 × 17 min ≈ 8.5 hours, on your machine's scale. Runs are independent, so two or three processes in parallel shorten this.

## 6. Suggested new metrics (evaluation only)

Computed after training from μ and Σ, like the near-optimal measure, and never fed back:

- **Fitted-vs-true Q correlation**: Pearson and Spearman correlation across all portfolios between the final Q̂ and the true objective w'μ − (γ/2)w'Σw. Says whether the model learned the landscape or only got lucky on the top pick.
- **Top-k recall**: how many of the true top 10 portfolios are in the learned top 10.
- **Calibration of the winner**: Q̂ of the learned portfolio minus its true objective (optimism from picking the maximum of noisy estimates).
- For the tabular baseline the same metrics can be computed from `final_q`, which gives a like-for-like comparison.

These belong in `src/evaluate.py` (your file); not implemented yet.
