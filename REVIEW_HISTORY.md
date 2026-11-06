# Review History

**Covers:** every closed review item and settled decision between Albert and
William, 2026-09-29 to 2026-10-28 (§8 was added 2026-10-28). Later work is
recorded elsewhere: `grid10-robustness` in `EXPERIMENTS.md` §7.7, the FA
extension and its tuning in `FA_INTEGRATION_NOTES.md` and `EXPERIMENTS.md` §8,
the transaction-cost extension in `TRANSACTION_COSTS.md`, and the 2026-11-06
audit in `AUDIT_2026-11-06.md`. Moved here from
`REVIEW_NOTES_FOR_WILLIAM.md` on 2026-10-26 so that file can list only open
work. Section numbers §1 to §6 are unchanged from the original, because code
comments in `src/config.py` and `src/tests/test_agent.py` cite them.

---

## TL;DR / In plain terms

This is the paper trail. It records what was found in William's files and how
each issue was fixed (§1), how the Q-learning core was split between us (§2),
how γ, B and the convergence rule were decided and why (§3, §6), and the
follow-up ideas we tested after the full run and rejected (§7). Nothing in
here is waiting on anyone, except the one item marked as reopened (§5.3, the
tabular `update()` return value), which is tracked in
`REVIEW_NOTES_FOR_WILLIAM.md`.

**Key concepts to understand**

- **Closed vs. flagged.** An item is closed here only once the code changed
  and a test or check confirmed it (CLAUDE.md §2.3). Each row says where.
- **Pre-registration.** γ, B, K and the other settings in §6 were fixed on
  2026-10-16, before any full-scale result existed. Ideas that came from
  looking at results (§7) are labelled post-hoc.

**Key takeaways**

- Read this file to answer "why did we do it this way" in the report.
- For "what is left to do", read `REVIEW_NOTES_FOR_WILLIAM.md` and
  `NEXT_STEPS_FOR_ALBERT.md` instead.

---

## 1. Review issues: all closed

| # | Issue | Fixed in |
|---|---|---|
| 1 | SLSQP stopped early on daily moments | `51b3eb7` |
| 2 | `best_discrete_weights` hardcoded the grid | `51b3eb7` |
| 3 | Float drift in grid weights | `02f6a49` |
| 4 | `n = mu.shape` tuple | `51b3eb7` |
| 5 | `.vscode/settings.json` Windows path | `51b3eb7` |
| A | Reward NaN for a one-day batch | `0e2201d` |
| B | `best_discrete_weights` returned a view | `0e2201d` |
| C | `nit <= 1` guard replaced with a relative KKT check | `0e2201d` |
| D | `run_experiments.py` wrote `INDEX.md` with no encoding; crashed on Windows (`UnicodeEncodeError` on γ under cp1252, never surfaced on Mac). Fixed on all six read/write calls in that file, not just the one that failed | William, 2026-10-19 |
| E | `.gitignore` rule `results/*/*.npz` only matched logs one folder deep, so a run with `--results-dir results/smoke` left its 8 `.npz` logs and 58 reduced-settings outputs trackable; `results/settings.json` was also ignored despite the comment saying settings get committed. Fixed by Albert in your file, see §6.5. **Reviewed by William 2026-10-26: correct, kept.** The original rule had only been tested against the default `results/` layout, not the `--results-dir results/smoke` layout `EXPERIMENTS.md` itself recommends | Albert, 2026-10-20 |

Re-checked against real calibration (γ 0.25–30), 200 random 9-asset problems,
and deliberately bad solves. Nothing fires at realistic scales; everything
genuinely wrong gets caught. Full suite at the time: 149 passed.

---

## 2. The tabular Q-learning core: split by file, done

We split instead of building side by side: you own `src/agent.py` (the part
that picks portfolios and updates Q-values), I own `src/train.py` (the part
that runs episodes, manages seeds, saves results). Both are merged into `dev`.

The interface that made this work, exactly as agreed:

```python
agent = TabularQAgent(n_actions, *, alpha_rule, epsilon0, beta, q_init, discount, rng)
a = agent.select_action()          # epsilon-greedy, eps_t = epsilon0 * exp(-beta * t)
agent.update(a, reward)            # Q(a) <- Q(a) + alpha * [R + rho * max Q - Q(a)]
agent.greedy_action                # current best action index
agent.q_values, agent.visit_counts # read-only copies, for logging
```

`train.py` never sees returns, μ, Σ, or a benchmark, only an action index and
a reward. The leakage tests in `test_environment.py` guard `src/agent.py`.

**Rules that are still worth keeping**, now that both sides exist:
1. I touch `agent.py` only through its interface; you touch `train.py` only
   through its outputs (the `.npz` logs and `manifest.json`).
2. ~~Once both are stable, run an integration check: fixed seed, a few thousand
   episodes on the 25% grid, rerun to confirm identical output.~~ Satisfied by
   the 2026-10-22 review: 30,000 episodes of three seeds re-run bit-identical
   to the saved logs (`RUN_ISSUES_2026-10-22.md` §2).
3. Review swap: I review `agent.py` line by line, you review `train.py`.
   **William's half is still open**, tracked in `REVIEW_NOTES_FOR_WILLIAM.md`.

**Prospectus:** §4's ownership row now reads "Agent (`agent.py`): William;
training runner (`train.py`): Albert; each reviews the other's PR."

---

## 3. Decisions

### 3.1 Were open; decided 2026-10-16 (see §6)

These two aren't hyperparameters to tune. γ defines the research question
itself, and the assignment requires convergence to be defined before anyone
looks at how a run did, specifically so neither gets chosen because it makes
a result look good.

| Decision | Owner | Why it can't be deferred |
|---|---|---|
| γ | William | It's the investor's risk aversion, not an algorithm setting. Choosing it by seeing which value gives a flattering result is the exact researcher-degrees-of-freedom problem `DATA_CLEANING.md` §8.4 already warns about. Needs justification on outside grounds (matching a known value, or the three-asset version of this problem if it used one) before any results exist. |
| Convergence rule (the definition, not the threshold) | Both | Assignment: "define convergence before examining the final results." The shape of the rule is basically agreed already: agent-observable quantities only (greedy action unchanged for K episodes, or largest \|ΔQ\| below a tolerance), never the gap to the benchmark, since that needs μ and Σ. Needs writing down as final, not as a proposal, before any run is read as converged or not. |

### 3.2 Were parked; set as adopted defaults 2026-10-16 (see §6)

Everything below is an implementation detail with a real, explorable
tradeoff (compute time vs. statistical confidence, mostly). The code already
takes every one of these as a parameter rather than a hardcoded value, so
there's no urgency and no reason to guess now. Revisit when there's a
notebook to run the exploration in, and pick based on agent-observable
criteria (how fast/reliably Q-values settle), never by comparing to the
benchmark. That comparison is what makes a choice here a leakage risk
instead of ordinary tuning.

- Episode length B (proposed baseline: 20)
- Episode count / training budget
- Seed count (proposed: 10)
- β, the exploration-decay parameter
- Learning-rate magnitude, if not using sample averaging
- The numeric convergence threshold K (once 3.1's rule is written down)
- Which second Part VI robustness check runs alongside the grid-resolution one

### 3.3 Already settled, no longer open

- **What the agent sees**: rewards only, never a return batch. Settled, enforced by the leakage tests.
- **Learning-rate rule**: sample averaging, α = 1/N(a), is what `agent.py` and `train.py` both implement as the default.
- **Discount ρ**: defaults to 0, the bandit baseline. Still an explicit constructor argument, so a nonzero value stays a stated choice if one's ever needed.
- **Q initialization, tie-breaking**: `q_init` is a parameter; ties break uniformly at random. Built and tested (`test_agent.py`).

---

## 4. Numbers to keep in mind for the agent

- **Noise.** At γ = 5, B = 20, one episode's reward for the best menu
  portfolio has a standard deviation 7.5 times its true value. The best and
  second-best menu portfolios differ by 2%.
- **Budget.** With 1/N averaging and uniform exploration, about 13 million
  episodes (260 million simulated days) got within roughly 2% of the best
  menu portfolio on a typical seed, finding the exact best about a quarter
  of the time.
- **Lock-in risk.** In a 3-asset toy run with fast-decaying ε, the agent
  settled on the second-best portfolio because the best one got an unlucky
  early average and was rarely retried.
- **Speed**, measured with your agent filled in:

  | Grid | Actions | Runner | Bare loop | Per 13M-episode run |
  |---|---|---|---|---|
  | 20% | 1,287 | 23.6 µs/episode | 19.7 µs | ~5 minutes |
  | 10% | 43,758 | 91 µs/episode | 51.6 µs | ~20 minutes (3.3 hr for 10 seeds) |

  Most of the runner's overhead is copying the full Q-table after every
  update. See §5.3.

---

## 5. Four things found while building `train.py` against your interface

### 5.1 Learning-rate visit count: resolved, not a live risk

I'd worried `_alpha` might call `alpha_rule(0)` on the first visit, which
numpy turns into `inf` with only a warning. Checked your actual code: `_alpha`
reads `self._visits[action] + 1`, so the first call is always `alpha_rule(1)`.
Confirmed by `test_sample_average_first_update_overwrites_q_init_exactly`.
Nothing to do.

### ~~5.2 `greedy_action` and randomness: mostly already handled, two small adds left~~ Done 2026-10-26 on `feature/post-run-fixes`

Your tie-break uses a separately spawned `_tie_rng`, not the main `rng`, so
reading it for logging can't perturb explore/exploit decisions, confirmed by
`test_greedy_action_tiebreak_uses_a_separate_rng_stream`. `train.py` never
calls it anyway; it computes its own `np.argmax(q_values)` for logging.

Two cheap things still worth adding:
- a docstring note on `greedy_action` that reading it can consume a draw from `_tie_rng` (only when an actual tie exists, which is free to check for but costs nothing to document)
- a public `epsilon` property exposing `_epsilon()`, so `train.py` can stop recomputing `epsilon0 * exp(-beta * t)` independently. Harmless today since the two formulas agree, but would silently drift if the schedule ever changes shape.

**Applied 2026-10-26.** Both adds are in `src/agent.py`, and `train.py` now
logs `agent.epsilon` instead of recomputing the schedule. This touched three
of Albert's files, so he should look at it: `src/train.py` (one line plus two
docstrings), and the stand-in agents in `src/tests/test_train.py`,
`test_evaluate.py` and `test_config.py`, which each gained an `epsilon`. Any
future agent passed to `run_experiment` (the function-approximation extension
included) must now expose `epsilon` too. Three new tests: two in
`test_agent.py` (the property is the rate the next selection uses; reading it
does not advance t) and one in `test_train.py` (with the real agent, the
logged rate after e episodes is ε0·exp(−βe), recomputed from the formula).
Checked against the pre-change code: two seeds × 30,000 episodes at the
full-run settings give identical `.npz` arrays (13 of 13). The full run's
manifests record the old `train.py` and `agent.py` hashes, so a hash check
against current code will now differ even though the logs do not.

### 5.3 Full Q-table copy every episode: deferred 2026-10-19, reopened 2026-10-26

Reopened because the 10% grid is now the Part VI action-space check. Open item,
tracked in `REVIEW_NOTES_FOR_WILLIAM.md`. Original note:

`train.py` calls `agent.q_values` once per episode just to read one scalar
(`new_q[action]`). Fine at the 20% grid; it's the dominant cost at the 10%
grid (table above). Proposed fix, still not applied: have `update()` return
the new `Q(action)` value, and `train.py` keeps its own local mirror it
updates at one index instead of re-copying the whole table every time.

Decided to defer rather than fix: none of the four planned experiments use
the 10% grid (`src/config.py` only has 20% and 25%), so this costs nothing
today. Fixing it would also need both files to change together (my `update()`
return value plus Albert's loop using it), so a one-sided fix would buy
nothing. Revisit only if a 10% grid run gets added to the plan.

### 5.4 Log format differs from what §2 originally promised: informational

A per-episode log (action, reward, ε) would be hundreds of MB at 13M
episodes, so `train.py` saves windowed checkpoints instead (every
`log_every` episodes: mean reward, largest |ΔQ|, logged greedy action,
per-action choice counts, ε, a float32 Q snapshot) plus every change of the
logged greedy action. Enough for learning curves, exact regret, "greedy
unchanged for K episodes" at any K, and the final policy to evaluate. Not
enough for a per-episode reward plot: say so if the report needs one and
I'll add an optional full log for short runs.


---

## 6. Decisions of 2026-10-16, and what I changed

### 6.1 The decisions

| Setting | Value | Where |
|---|---|---|
| γ | {3, 5, 8}; 5 is the baseline for headline tables, 3 and 8 are the Part VI risk-aversion check | `src/config.py` |
| Episode length B | 20 trading days | `src/config.py` |
| Convergence rule | Greedy action unchanged for K = 10% of training episodes (K = 1,300,000 at 13M episodes). Primary reading: the final greedy action held for at least K episodes through the end (`greedy_stable_through_end`); "first time any action held for K" is secondary | `src/config.py` |

That settles both items in §3.1. Most of §3.2 is now set as adopted defaults
(all one-line changes in `src/config.py`):

- 10 seeds, master seed 20261016;
- 13 million episodes per seed;
- β chosen so ε = 1% halfway through training (about 7.08e-7);
- \|ΔQ\| tolerance 1e-5 per window, held for good;
- near-optimal line 25 bps/yr of certainty-equivalent loss;
- 20% grid baseline, 25% grid as the second resolution;
- the second Part VI check is the γ sweep.

### 6.2 Edits to your files, flagged here as CLAUDE.md requires

`PROJECT_PROSPECTUS.md`:

- **Row 7** (training loop + convergence criterion) is now marked done, with
  the decisions listed.
- **Rows 8 and 9** said `evaluate.py` was empty. They now say the code is done
  and waiting on the full-scale run.

Nothing else in the prospectus changed. Revert or reword freely.

### 6.3 Done, 2026-10-19

`DATA_CLEANING.md` §8.4 now has the written outside-grounds justification:
γ = 5 within the 2–7 corrected finance-literature range (Elminejad, Havranek
and Irsova, 2022), below the ~7 household-survey median (Barsky et al.,
1997), under the Mehra-Prescott (1985) cap of 10; γ = 8 chosen to probe past
the finance range while staying under the other two. Sources are listed in
that section. The market-implied SPY-only candidate (≈2.9) was considered
and not used, since it's calibrated from the same dataset the benchmarks are
built on.

### 6.4 New code and docs

- `src/config.py`: every experiment setting, plus the list of four
  experiments (baseline, 25% grid, γ = 3, γ = 8).
- `src/run_experiments.py`: one command trains, evaluates, compares the two
  grids and writes `results/INDEX.md`. It resumes after interruption and
  refuses half-finished folders.
- `src/tests/test_config.py`: 12 tests pinning the decisions and exercising
  the runner.
- `EXPERIMENTS.md`: the runbook.
- `EVALUATION.md`: §7 updated.

One more item for awareness:

- **All four experiments share the master seed.** Seed i sees the same
  simulated returns in every experiment, and every experiment uses the same
  evaluation sample. That makes the comparisons paired; it is recorded in
  `EXPERIMENTS.md`.

A full run is about 3.5 hours. Two items closed, 2026-10-19:

- **`requirements.txt` is pinned**, to the versions tested on William's
  Python 3.10 (numpy capped at 2.2.6; 2.3+ requires Python 3.11+). Both
  combinations (3.10 and Albert's 3.13) are documented in `README.md` and
  `DATA_CLEANING.md` §7.5.
- **Results-sharing is adopted.** `.gitignore` now commits `INDEX.md`,
  `index.json`, each experiment's `manifest.json`/`settings.json`, and every
  `evaluation/` folder; only the `.npz` training logs stay local. Verified
  against the documented `results/` layout before trusting it.

### 6.5 `.gitignore` fix, 2026-10-20 (edit to your file)

Albert asked for this one to be fixed directly rather than left for you, so
it is fixed in the code, not just flagged. Revert or reword freely.

Your rule `results/*/*.npz` ignores `results/baseline_g5_grid20/run_000.npz`
but not `results/smoke/baseline_g5_grid20/run_000.npz`, because `*` matches
one folder level. After the smoke run, `git status` showed 66 untracked files
under `results/smoke/`, including all 8 `.npz` logs, so a `git add -A` would
have committed training logs plus numbers from a 200,000-episode budget that
could be mistaken for results.

Changes, all in the results block:

- `results/*/*.npz` became `results/**/*.npz`, which matches at any depth.
- `results/smoke*/` is ignored entirely, since reduced-settings checks are
  not report results (`INDEX.md` there already says so).
- `results/.*.new/` is ignored: the runner builds the comparison in a staging
  folder and renames it at the end, so the folder only survives a crash.
- `!results/settings.json` added. The comment said settings get committed,
  but `results/*` ignored that file. It is what makes the runner refuse to mix
  settings in one folder, so a clone without it loses that guard.

Checked with `git check-ignore` on 15 paths: logs ignored at depth 2 and 3,
everything under `results/smoke/` and the staging folder ignored; `INDEX.md`,
`index.json`, `settings.json`, manifests, `evaluation/` files (JSON and PNG),
the comparison folder, `data/raw/` and `src/` all still trackable. `git
status` now shows nothing under `results/`.

---

## 7. Follow-up ideas after the full-scale run, 2026-10-26

### 7.1 Why exact recovery fails

Only 4 of the 1,287 portfolios on the 20% grid are within 25 bps/yr of the
best (at 0, 10.5, 12.3 and 15.9 bps), and 9 within 50. One episode's reward has
a standard deviation about 370 times the 10.5 bps gap between the top two, so
separating them by 2 standard errors takes about 1.1 million visits to each
(`reports/full_run_2026-10-22/data.json`, unpaired). The learned portfolio got
about 11.4 million visits; the other 1,286 got about 1,200 each. The agent
reliably reaches the top four and then the choice among them is largely luck.

### 7.2 Rejected: slower ε decay and a minimum-tries floor

Tested with a fast surrogate of the tabular agent: per-portfolio rewards drawn
from their exact distribution under the Gaussian simulator (sample mean normal,
sample variance scaled chi-square, independent), the same ε-greedy schedule and
1/N averaging, greedy action recomputed every 500 episodes. Analysis only: it
uses μ and Σ, which the agent never does. Validated against the real run on
the baseline (simulator 45% exactly best, median gap 10.5 bps, training regret
62 bps/yr; real 30%, 14.1 bps, about 66 bps/yr; within noise for 10 real
seeds). Baseline setting (γ = 5, 20% grid, 13M episodes), 100 seeds each; the
exactly-best share has a sampling error of about ±5 points.

| Change | Exactly best | Mean learning gap (bps/yr) | Training regret (bps/yr) |
|---|---|---|---|
| Current: ε = 1% at halfway | 44% | 9.3 | 62 |
| ε = 3% at halfway | 36% | 9.5 | 77 |
| ε = 10% at halfway | 41% | 9.0 | 110 |
| ε = 30% at halfway | 48% | 8.1 | 184 |
| ε = 1 throughout | 26% | 19.5 | 468 |
| Floor of 1,000 tries each | 37% | 11.4 | 102 |
| Floor of 2,000 tries each | 31% | 10.5 | 141 |
| Floor of 5,000 tries each | 42% | 8.7 | 262 |
| Floor of 9,000 tries each (89% of budget) | 53% | 6.2 | 423 |
| B = 63, current schedule | 44% | 7.6 | 59 |
| B = 126, current schedule | 49% | 6.2 | 57 |
| B = 252, current schedule | 68% | 3.7 | 54 |

Decision (William, 2026-10-26): drop slower decay and the floor. Neither moves
the learning gap beyond noise except at several times the regret, because
uniform extra tries are still about 100 times short of separating the top
contenders. Pure exploration is worse than the current schedule because
exploitation corrects lucky overestimates (a portfolio that looks too good
gets tried until it falls back), and uniform exploration gives that up. Longer
episodes help because they cut the noise of every reward at once. That became
the B proposal in `REVIEW_NOTES_FOR_WILLIAM.md`.

The surrogate script is not committed; it lived in William's session
scratchpad. If the report cites this table, commit the script first
(CLAUDE.md §6).


---

## 8. Post-hoc runs b63, n26m, eps-decay and all-three, 2026-10-28

All four ran at full scale (3 γ × 10 seeds each, 20% grid), were extracted
and reported under `reports/<run_id>/`, and are merged into `dev`. Each was
registered with its expected direction on 2026-10-27, before it ran
(`EXPERIMENTS.md` §7). Numbers below are from each experiment's
`evaluation/summary.json`; learning gap in bps/yr against the best 20%-grid
portfolio, 10 seeds.

| Run | What changed | Mean gap γ = 5 / 3 / 8 | Worst seed γ = 5 / 3 / 8 | Exactly best γ = 5 / 3 / 8 | Regret per episode γ = 5 (bps/yr) |
|---|---|---|---|---|---|
| `baseline-v1` | (pre-registered) | 17.6 / 30.6 / 13.0 | 47.3 / 81.1 / 53.1 | 3 / 5 / 5 | 65.5 |
| `b63` | B 20 → 63 | 5.1 / 10.0 / 9.1 | 15.9 / 20.0 / 29.5 | 6 / 5 / 5 | 51.2 |
| `n26m` | 13M → 26M episodes | 8.3 / 16.2 / 5.0 | 15.9 / 62.3 / 27.2 | 3 / 4 / 7 | 60.9 |
| `eps-decay` | ε at halfway 1% → 3% | 16.1 / 4.0 / 11.6 | 37.2 / 20.0 / 51.6 | 4 / 8 / 5 | 77.3 |
| `all-three` | all three together | 7.9 / 4.0 / 1.1 | 29.4 / 20.0 / 11.5 | 5 / 8 / 9 | 72.1 |

What closed with these runs:

- **The B proposal** (`REVIEW_NOTES_FOR_WILLIAM.md`, 2026-10-26) became run
  `b63`. B = 252 was not run. The 10% grid runs at `all-three`'s settings
  (Albert's choice, 2026-10-27, `EXPERIMENTS.md` §7.7) rather than at B = 20
  as William proposed; agreed, since it is compared with `all-three`.
- **The slower-decay idea** was dropped by William on 2026-10-26 (§7.2) and run
  anyway by Albert as `eps-decay`. The run agrees with the surrogate at γ = 5
  and 8 (gap unchanged within noise, regret up) and did better than expected
  at γ = 3.
- **Rebuilding reports** (`NEXT_STEPS_FOR_ALBERT.md`, 2026-10-26) is solved by
  `src/reporting/` (`reports/README.md`): each run is extracted once and every
  report compares with the runs registered before it.

What the runs show, for the report:

- Every variant lowered the mean learning gap and the worst seed at every γ
  (12 of 12 comparisons each). Medians and exact-best shares did not all
  improve: `n26m` at γ = 3 went from a median of 10.0 to 20.0 and from 5 to 4
  exactly best. With 10 seeds, part of the mean improvement is the baseline's
  single bad seeds (81.1 bps at γ = 3, 53.1 at γ = 8) not recurring.
- Longer episodes were the cheapest gain: `b63` cut the γ = 5 mean gap from
  17.6 to 5.1 with lower regret. More exploration (`eps-decay`, `all-three`)
  raised regret at γ = 5 and 8.
- Measured in simulated days every variant takes longer to meet the
  convergence rules, because each sees more data. Measured in episodes,
  `b63` is faster than the baseline (γ = 5 primary rule 9.25M → 6.90M
  episodes; near-optimal for good 1.29M → 0.70M), while `n26m` and
  `eps-decay` are slower (near-optimal about 2.9M episodes), since a longer
  budget or slower decay keeps ε high for longer.
- Against the registered expectations: `b63` matched (smaller gap at every
  γ). `n26m` expected a slight improvement likely within noise; the mean gap
  fell at every γ, but the median and exact-best share got worse at γ = 3.
  `eps-decay` matched at γ = 5 and 8. `all-three` expected the smallest gap of
  runs 1 to 4 at every γ; it was smallest at γ = 8, tied at γ = 3 and behind
  `b63` at γ = 5.
