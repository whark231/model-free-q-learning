# Notes for William

**From:** Albert (§7 added by William)
**Date:** 2026-10-26 (first written 2026-10-13)
**Status:** the full-scale run finished cleanly on 2026-10-22 and its results
are committed (`results/`, `RUN_ISSUES_2026-10-22.md`). Every review issue
across all rounds is closed. None of the nine run findings is in William's
files. **What is still William's is in §7**: two small `agent.py` additions,
one `.gitignore` call, reviews, the extension decision, and a proposal on
follow-up experiments. §5.3 stays deferred unless a 10% grid run is added.

---

## TL;DR / In plain terms

The split worked: I wrote `train.py`, you wrote `agent.py`, and the interface
we agreed on first meant neither of us had to guess the other's method names.
247 tests pass across the project.

Update 2026-10-16: γ, B and the convergence rule are now decided, and the
parked choices are set as adopted defaults (§6). §3 below is kept as the
record of how the decisions were framed.

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
   **William's half is still open** (§7.2, item 1).

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

### 5.3 Full Q-table copy every episode: confirmed deferred, 2026-10-19

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

## 7. After the full-scale run (added by William, 2026-10-26)

### TL;DR / In plain terms

The run worked and every number was re-derived independently
(`RUN_ISSUES_2026-10-22.md`). The agent lands near the best grid portfolio
but finds the exact best in only 30% to 50% of seeds, and on total loss
against the continuous optimum the coarser 25% grid beats the 20% grid. The
reason is arithmetic, not a bug: the few portfolios that compete for first
place each get about a thousand tries, and telling them apart needs hundreds
of thousands. Albert's fix plan covers the reporting problems and touches
only his files. My list is below.

**Key concepts to understand**

- **Contenders.** Only 4 of the 1,287 portfolios on the 20% grid are within
  25 bps/yr of the best, and 9 within 50. Learning is a contest between
  those few; the other 1,278 are clearly worse.
- **Visits needed vs. visits received.** One episode's reward has a standard
  deviation 7.5 times the best portfolio's true value. Telling apart two
  portfolios 10 bps/yr apart (2 standard errors) takes about 617,000 visits
  to each. The learned portfolio got about 11.4 million; the other 1,286
  shared about 1.6 million, roughly 1,200 each. That is about 500 times short.
- **Episode length and episode count trade off.** The visits needed scale
  with 1/B, so what a portfolio needs is about 12 million simulated days
  whatever B is. More days can come from more episodes or longer ones.

**Key takeaways**

- The binding constraint is exploration spread evenly over 1,287
  portfolios, most of them hopeless, not the total budget.
- That is the main Part V/VI finding, and it answers the assignment's "what
  prevented exact recovery" directly. It belongs in the report as a result.

### 7.1 Results with the total gap added

From `results/INDEX.md`; the total column is `RUN_ISSUES` Finding 3. All in
bps/yr of certainty-equivalent loss.

| Experiment | Discretization | Learning gap (median) | Total vs. continuous (median) | Exactly best |
|---|---|---|---|---|
| γ = 5, 20% grid (baseline) | 5.7 | 14.1 | 19.8 | 30% |
| γ = 5, 25% grid | 11.7 | 2.5 | 14.3 | 50% |
| γ = 3, 20% grid | 2.5 | 10.0 | 12.5 | 50% |
| γ = 8, 20% grid | 6.1 | 5.7 | 11.9 | 50% |

### 7.2 William's list

1. **Review swap.** Review `train.py` (outstanding since §2), and Albert's
   `evaluate.py` / `run_experiments.py` change for Findings 1 to 3 when he
   opens the PR.
2. ~~**`.gitignore`: keep the run log**~~ Done 2026-10-26:
   `!results/full_run_*.log` added. Albert still has to add the log itself,
   since it only exists on his machine.
3. ~~**`agent.py`, §5.2.**~~ Done 2026-10-26, see §5.2. Albert to review the
   `train.py` and test changes.
4. **Prospectus rows 8 to 10** still say "waiting on the full-scale run."
   Update after Albert's evaluation fix lands, so the evidence cites the
   corrected tables rather than the ones `RUN_ISSUES` says mislead.
5. **Extension scope, before Week 7 starts on 2026-10-30.** The results point
   at function approximation (already mine in prospectus §4): a Q-function
   over portfolio weights lets one portfolio's rewards inform its neighbours',
   which is exactly what the tabular agent lacks. One caution for the write-up:
   a model quadratic in the weights can represent the mean-variance objective
   exactly, so the report should say plainly that it estimates the reward
   function from rewards only, which is still model-free in the assignment's
   sense, and that it never sees μ or Σ.
6. **Report sections that are mine** (prospectus §4): the Theory/Benchmark
   derivation and the Q-learning section. §7's arithmetic is the core of the
   second.
7. **Agree the follow-up experiments in §7.3 with Albert before anything is
   run.**

### 7.3 Proposed follow-up experiments (none run, none agreed yet)

The 2026-10-16 settings stay the headline, unchanged. Anything below is an
additional Part VI check. The assignment's own list includes "different
training budgets, episode lengths, and random seeds," so these fit, but the
motivation came from seeing results, so each must be written into
`src/config.py` and `EXPERIMENTS.md` with its expected direction before it
runs, and labelled post-hoc in the report.

| Idea | What the numbers say | Recommendation |
|---|---|---|
| Minimum number of tries per action | To matter, the floor has to approach ~600,000 for the contenders, but a floor applies to all 1,287 portfolios: about 800 million episodes. An affordable floor (a few thousand) matches what ε = 1 early in training already gives. It would also be the one change in my file | No |
| Slower ε decay | Moves visits around inside the same 13M episodes. Even ε near 1 for the whole run gives about 5,000 visits per portfolio, still about 100 times short, and raises the cumulative regret Part V reports | No, as a baseline change |
| Longer budget (more episodes) | Under even exploration, resolving 10 bps needs roughly 60 times the current budget | Only through higher B, which is cheaper |
| Higher B | Measured on my machine: B = 63 costs 43.8 µs/episode vs 38.2 µs at B = 20 (+15%) for 3.15 times the simulated days, about 2.7 times cheaper per day, because the agent's per-episode overhead dominates. Reward noise falls from 7.5 to 4.2 times U*. Still about 160 times short of resolving 10 bps, so expect fewer brief switches and a smaller learning gap, not reliable exact recovery | Yes: B = 63 (a quarter, still "a short batch of returns") as the episode-length check, γ = 5, 20% grid, same seeds |
| 10% grid | 43,758 portfolios, 25 within 25 bps. About 37 exploratory visits each at the current budget. Discretization gap falls from 5.7 to about 1.3 bps; the learning gap should grow. 13M episodes fits under `train.py`'s snapshot memory guard; a longer budget would not | Yes, later, as the curse-of-dimensionality check Part VI asks us to discuss, with "learning gap gets worse" stated in advance. Un-defer §5.3 first: copying a 43,758-entry Q-table every episode is the dominant cost |

**Code cost, all in Albert's files.** `EPISODE_LENGTH`, the ε schedule and
the grid list are global in `src/config.py` and part of the results folder's
locked `batch_settings` (`run_experiments.py:39-50`), so a B = 63 or 10%
experiment cannot share `results/` with the existing four. It needs a
per-experiment B in `config.Experiment` and its own `--results-dir`.
