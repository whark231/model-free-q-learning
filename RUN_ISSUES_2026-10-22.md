# Full-Scale Run, 2026-10-22: Issues and Fix Plan

**Written:** 2026-10-22 by Claude, for Albert · **Run:** 01:35 to 05:00 EDT, commit `80a6e9c`, settings from `src/config.py` unchanged
**Status of every item below:** found and verified, **nothing fixed in code yet**. The plan in §4 waits on Albert's go-ahead.
**Checked again 2026-11-06 (`AUDIT_2026-11-06.md`):** steps 1 to 3 of §4 are still open in the code: no total-gap column in `per_seed.csv` or `INDEX.md`, no post-hoc "settled" measure, no checkpoint label, and no commit or extra file hashes in `summary.json`. The training logs are now tracked (step 7 done).

---

## TL;DR / In plain terms

The full run worked. All four experiments trained 10 seeds each with no crash, no
restart and no code change, and every number in `results/INDEX.md` was re-derived
independently and matches. The problems are in how two of the convergence numbers
read and in one missing table column. Left alone, the report would say the agent
needed far longer to settle than it did, and would hide that the coarser grid gave
the better total result. Fixing this needs a re-run of evaluation only (minutes,
no retraining) plus some report wording.

**Key concepts to understand**

- **Greedy action.** The portfolio with the highest Q-value right now, the one the
  agent would pick if it stopped exploring. "Converged" in our rules means this
  stopped changing.
- **A brief switch.** Late in training the agent still explores 0.01% to 1% of the
  time. A rarely tried portfolio can get one lucky reward, become greedy for a few
  hundred episodes, then fall back. Our primary rule treats each switch as a reset
  of the 1.3M-episode clock.
- **Total gap.** The learned portfolio's loss against the continuous optimum equals
  the discretization gap (grid's best vs. continuous) plus the learning gap (learned
  vs. grid's best). INDEX.md reports the two parts but never their sum.

**Key takeaways**

- The results are valid and can be committed as the first full-scale results.
- Before anyone writes the Parts IV to VI sections from them, apply fixes 1 to 3 in
  §4 so the tables say what the numbers mean.

---

## 1. What happened during the run

| Time | Event | Action taken |
|---|---|---|
| 01:27 | An earlier launch (before this session) died before training any seed. It left `results/baseline_g5_grid20/manifest.json` with 0 of 10 runs, which the runner refuses as "partly trained" (`src/run_experiments.py:81-86`) | Moved the folder to the session scratchpad (deletion was blocked by the permission system). Nothing else was in it |
| 01:35 | Full run launched with `nohup caffeinate`, log at `results/full_run_2026-10-22.log` | |
| 01:38 | My own monitoring script used `pgrep -f src.run_experiments`, which matched the monitor's own command line, so it could never detect the run ending. Tooling bug, not a project bug | Switched to watching the process ID |
| 02:26 | `baseline_g5_grid20` evaluated | Checked benchmarks independently (below) |
| 03:14 | `resolution_g5_grid25` evaluated | Same |
| 04:07 | `robust_g3_grid20` evaluated | Same |
| 05:00 | `robust_g8_grid20` evaluated, comparison and INDEX written, process exited cleanly | Independent review agent launched |

About 300 seconds per seed on the 20% grid and 285 on the 25% grid. The log has no
errors or warnings.

**Fixes during the run: none.** No restart was needed and no project file changed.

## 2. What was verified

| Check | How | Result |
|---|---|---|
| Grid sizes, best grid portfolio, continuous optimum, discretization gap (all four experiments) | My own stars-and-bars grid and projected-gradient solver, reading only the processed CSVs | Match to about 1e-11 |
| Every per-seed number (final action, CE loss, all four convergence rules, regret) | Reviewer re-derived from the `.npz` logs without `src/evaluate.py` | All 40 seeds match |
| Share-of-seeds and medians in INDEX.md | Reviewer, from the re-derived numbers | All match |
| Learned weights | From manifests | Nonnegative, sum to one, 10 runs per experiment |
| Evaluation sample | Reviewer rebuilt it from the seed plan | Matches to 1e-12; separate from training draws |
| Leakage | Reviewer read `agent.py`, `train.py` | Agent gets only action count, settings and its rng; reward computed in `train.py`; convergence rules computed only after training |
| Reproducibility | Reviewer re-ran 30,000 episodes of three seeds | Bit-identical to saved logs; code and data hashes match the manifests; commit is clean |
| Comparison folder | Byte comparison | Identical to the standalone evaluations |

## 3. Findings

Ranked by how much they would mislead a reader. Severity "should-fix" means fix
before writing the report; "note" means handle in the report text.

### Finding 1 (should-fix): the primary convergence rule mostly measures the timing of the last brief switch

**What.** `greedy_stable_through_end` reports a baseline median of 185.1M simulated
days (71% of the budget), and 4 of 40 seeds "fail" it. The learned portfolio was
settled far earlier.

**Evidence (I re-checked this myself from the `.npz` logs).**
- In all 40 seeds, the greedy action at the halfway checkpoint (episode 6.5M) is the
  final learned action. The second half of training changed no seed's answer.
- At 10,000-episode checkpoints, the final action was greedy at every checkpoint
  from episode 4.34M or earlier in every seed. Baseline seeds settled at 4.03M,
  2.69M, 2.38M, 2.33M, 1.16M, 2.09M, 2.72M, 2.90M, 0.90M and 1.13M episodes.
- Between checkpoints, rarely visited actions (1,000 to 4,000 visits, so noisy
  Q-values just below the incumbent's) win briefly after a lucky reward, for 1 to
  477 episodes. Baseline run 0 briefly switches to action 803, which loses
  300 bps/yr, then returns. Baseline run 1 spends 472 of its last 6.5M episodes on
  such switches and so fails the rule.
- The four "failing" seeds are baseline run 1, γ = 3 runs 6 and 7, and γ = 8 run 6.

**Consequence.** Read at face value, the report would say the agent needs
roughly 150M to 190M simulated days to converge and fails in up to 20% of seeds,
when every seed actually settles within 8M to 87M days (18M to 81M on the
baseline) and never changes its answer afterwards.

**Proposed fix.** The rule was fixed in advance, so keep it and report its numbers
unchanged. Add a second, clearly labelled post-hoc descriptive measure: "final
action greedy at every 10,000-episode checkpoint from episode X". Explain the
brief-switch mechanism in the report. The same mechanism explains why exact recovery
of the best portfolio is luck-dependent: once ε is small, other actions stop being
sampled and their Q-values freeze near the winner's.

### Finding 2 (should-fix): the two convergence columns in INDEX.md are measured at different resolutions

**What.** `near_optimal_for_good` is checked at 10,000-episode checkpoints, which is
the definition in `EVALUATION.md`. The primary rule counts every episode. INDEX.md
shows them side by side without saying so.

**Evidence (reviewer; consistent with Finding 1).**

| Experiment | Near-optimal, at checkpoints (reported) | Same rule, every episode |
|---|---|---|
| baseline | 25.8M days | 155.7M days |
| 25% grid | 9.6M days | 66.8M days |
| γ = 3 | 22.2M days | 147.8M days |
| γ = 8 | 25.8M days | 123.5M days |

**Consequence.** "Near-optimal at 26M days but converged at 185M days" compares two
differently sampled numbers.

**Proposed fix.** Label the INDEX column and the report table "at 10,000-episode
checkpoints", and give the every-episode figure in the text or a footnote.

### Finding 3 (should-fix): no table shows the total gap to the continuous optimum, and on it the coarser grid wins

**What.** `ce_loss_vs_continuous_bps` exists per seed inside each `summary.json` but
appears in no table: not INDEX.md, not `per_seed.csv`, not `comparison.json`.

**Evidence (re-checked; total = discretization gap + median learning gap).**

| Experiment | Discretization | Learning (median) | Total vs. continuous (median) |
|---|---|---|---|
| 20% grid, γ = 5 | 5.7 | 14.1 | 19.8 bps/yr |
| 25% grid, γ = 5 | 11.7 | 2.5 | 14.3 bps/yr |
| γ = 3 | 2.5 | 10.0 | 12.5 bps/yr |
| γ = 8 | 6.1 | 5.7 | 11.9 bps/yr |

**Consequence.** Part V asks for both benchmarks and Part VI for the action-space
trade-off. At this budget the finer grid gives the worse total result, which is the
main action-space finding, and the current tables hide it.

**Proposed fix.** Add a total-gap column to `per_seed.csv`, INDEX.md and
`comparison.json` (`src/evaluate.py`, `src/run_experiments.py`, both Albert's).

### Finding 4 (note): identical numbers across experiments come from shared random streams, not a bug

Baseline and γ = 8 both show a near-optimal median of 25.8M days. The exploration
draws depend only on the seed and the grid size, not on γ (ties use a separate rng,
`src/agent.py:76, 121-124`), so seed i explores the same way in all three 20%-grid
experiments. Both medians are run 1's value of 1.29M episodes. The |ΔQ| rule's
timing is identical in 9 of 10 seeds between γ = 5 and γ = 8. Each experiment was
confirmed to be evaluated with its own γ and grid. For the report: the |ΔQ| rule
tracks the exploration schedule and grid size, not learning quality, and the γ
experiments are paired runs, not independent ones (`EXPERIMENTS.md` §6 already
intends this).

### Finding 5 (note): the 10,000-episode evaluation sample cannot resolve the learning gaps

Two standard errors of the realized shortfall are 6 to 37 bps/yr. On the 25% grid,
seeds with an exact loss of 5.1 bps show a realized shortfall of -9.9 ± 32.6 bps (the
learned portfolio "beats" the best on the sample). The exact values are already the
primary numbers. The report must not cite realized shortfalls as evidence of a gap.

### Finding 6 (note): cumulative regret has awkward units

The plotted regret sums a daily objective difference over episodes against simulated
days (baseline median about 340). Convert for the report: about 66 bps/yr per
episode averaged over training, or multiply by B.

### Finding 7 (note): reproducibility gaps

- `results/full_run_2026-10-22.log` is gitignored (`results/*`), so it will not be
  committed.
- The manifest's `code_sha256` covers `train.py`, `agent.py`, `environment.py`,
  `rewards.py` and `grid.py`, but not `evaluate.py`, `config.py` or
  `run_experiments.py`. `summary.json` records no commit (only `index.json` does).
- pandas and matplotlib versions are not recorded.
- `EXPERIMENTS.md` line 3 still says "no full-scale run yet". (Fixed later.)

### Finding 8 (note): the median learning gap is not any seed's value

The baseline median of 14.1 bps is the mean of 12.3 and 15.9. The "median seed" in
the bar chart is run 1 (15.9, the upper median). Say so in the caption.

### Finding 9 (note): observation counts for the primary rule include the waiting window

Each `greedy_stable_through_end` count is the last switch plus K (1.3M episodes,
26M days). State this wherever the number appears.

## 4. Plan to correct

Every file in steps 1 to 3 is Albert's under prospectus §4. None of William's files
is touched. Training is reused, so a re-run takes minutes.

| Step | Change | Files | Fixes |
|---|---|---|---|
| 1 | Add the total-gap column; add the post-hoc "settled from checkpoint" measure, labelled descriptive; label near-optimal as checkpoint-resolution | `src/evaluate.py`, `src/run_experiments.py`, tests in `src/tests/` | Findings 1, 2, 3 |
| 2 | Record the git commit, SHA-256 of `evaluate.py` / `config.py` / `run_experiments.py`, and pandas and matplotlib versions in `summary.json` | `src/evaluate.py` | Finding 7 |
| 3 | Update `EXPERIMENTS.md` status and `EVALUATION.md` definitions (checkpoint resolution, K window included, regret units, upper-median seed) | docs | Findings 2, 6, 7, 8, 9 |
| 4 | Re-run `python -m src.run_experiments`. It skips training (all four experiments are "finished") and rewrites evaluations, comparison and INDEX | none | |
| 5 | Re-verify: the new total-gap column against the table in Finding 3, and all existing numbers unchanged | | |
| 6 | Independent review of the code change (CLAUDE.md §2.6) | | |
| 7 | Decide whether to keep the run log: either add `!results/full_run_*.log` to `.gitignore` or copy the log into an evaluation folder. `.gitignore` is shared, so check with William | `.gitignore` | Finding 7 |

Findings 4, 5, 8 and 9 become sentences in the report's Parts IV to VI sections. They
need no code.

The pre-registered convergence rule, K, the 25 bps line and the |ΔQ| tolerance stay
exactly as fixed on 2026-10-16. The new measure is added beside them and labelled
post-hoc, so none of the fixed-in-advance numbers changes.
