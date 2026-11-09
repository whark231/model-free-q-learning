# Parts IV and V: Evaluating the Learned Policy

**Owner:** Albert · **Status:** built and tested on `dev`; rules and thresholds fixed 2026-10-16 (§7)
**Code:** `src/evaluate.py`, `src/hashing.py` · **Tests:** `src/tests/test_evaluate.py` (35 tests on 2026-11-09)
**Date:** 2026-10-16, numpy 2.3.5, matplotlib 3.10.6

---

## TL;DR / In plain terms

`evaluate.py` grades a finished training experiment. For every seed, it takes
the portfolio the agent ended up preferring and compares it with the answer key:
the best possible long-only portfolio, the best portfolio on the menu, and equal
weight. It measures how much return per year the investor gives up by holding
the learned portfolio, how far its weights are from the optimum, how fast the
agent got close, and whether it met the convergence rules. The comparisons use
fresh simulated days the agent never trained on.

**Key concepts to understand**

- **Certainty-equivalent (CE) loss.** The objective U(w) = w'μ − (γ/2)w'Σw is
  the guaranteed daily return the investor considers as good as the risky
  portfolio. So 252 × (U_best − U), in basis points, is the guaranteed return
  per year given up by holding portfolio w instead of the best one. It is the
  main gap measure because it keeps its meaning at any γ.
- **Per seed, never averaged into one portfolio.** Each seed learns its own
  portfolio. Averaging their weights builds a portfolio no seed holds, and
  because U is concave that blend can score better than any of them. Results
  are therefore reported per seed, with the median and range.
- **Agent-observable vs. benchmark-based rules.** The convergence rules use
  only the agent's own Q-values, so they could stop training without leaking
  the answer key. The near-optimal rule uses the benchmark, so it is computed
  only here, after training.

**Key takeaways**

- Every Part V comparison and every Part IV count is produced by one command.
- The rules and their thresholds are fixed (§7): K = 10% of training
  episodes, ΔQ tolerance 1e-5, near-optimal line 25 bps/yr, γ ∈ {3, 5, 8},
  B = 20.

---

## 1. How to run

```bash
python -m src.evaluate results/<experiment> --eval-episodes 10000 \
    --converge-k <K> --dq-tol <tol> --near-optimal-bps <bps>

python -m src.evaluate --compare results/<grid25> results/<grid20> --labels 25% 20% \
    --out results/<comparison> --eval-episodes 10000 \
    --converge-k <K> --dq-tol <tol> --near-optimal-bps <bps>
```

Outputs go to `<experiment>/evaluation/` (or `--out`). The script refuses to
overwrite an earlier evaluation unless `--overwrite` is given.

| File | Contents |
|---|---|
| `summary.json` | Everything below, machine-readable, with the settings used |
| `weights.csv` | Benchmark weights, each seed's learned weights, absolute differences by asset |
| `performance.csv` | Mean, volatility, U(w), CE loss; realized values and paired shortfalls with standard errors |
| `per_seed.csv` | Learned action, learning gap and total gap, distances, regret, observations to each convergence rule (pre-registered and post hoc) |
| `weights_grouped_bar.png` | Median-loss seed vs. continuous optimum vs. best on the grid, with every seed's weights as dots |
| `learning_curve.png` | Mean reward per window; CE loss of the current greedy portfolio |
| `cumulative_regret.png` | Expected regret against the best grid action |
| `ce_loss_by_seed.png` | Final CE loss of each run |
| `convergence_by_resolution.png` (with `--compare`) | Median CE loss over time for each grid |

---

## 2. Definitions

| Symbol / name | Meaning |
|---|---|
| U(w) | w'μ − (γ/2)w'Σw, the daily objective |
| w* | Continuous long-only optimum (`constrained_optimal_weights`) |
| w_disc | Best portfolio on the training grid (`best_discrete_weights`) |
| Learned portfolio | The greedy action of a seed's final Q-table, ties to the lowest index |
| w̄ | Its weights averaged over the evaluation period. The policy is held fixed with exploration off, so it picks the same portfolio every period and w̄ equals that portfolio's weights |
| CE loss vs. X | 252 × (U(X) − U(w)) × 10,000, in basis points per year |
| Discretization gap | CE loss of w_disc vs. w*: the cost of the menu |
| Learning gap | CE loss of the learned portfolio vs. w_disc: the cost of learning from noise |
| Total gap | CE loss of the learned portfolio vs. w*. Equal to discretization gap plus learning gap, per seed (`ce_loss_vs_continuous_bps`; added 2026-11-09) |
| Relative gap | (U_ref − U) / \|U_ref\|; reported as secondary, and `None` when \|U_ref\| < 1e-5 per day |
| Expected regret | Sum over episodes of U(w_disc) − U(chosen action), computed from how often each action was chosen in each window. Units are daily objective units summed over episodes: divide by the number of episodes and multiply by 252 × 10,000 to get the average loss per episode in bps/yr |
| Observations | Episodes × episode length B, in simulated days. For `greedy_stable` and `greedy_stable_through_end` the count includes the K-episode waiting window, so it is the point at which the rule could first be declared, not when the action was first picked |
| Checkpoint | The end of each `log_every`-episode window (10,000 episodes in every run so far). Rules marked "checkpoint" are read only there, so a brief switch between two checkpoints is invisible to them |
| Median (summary) | The usual median: with 10 seeds, the average of the 5th and 6th values. Every "median" in `summary.json`, `INDEX.md` and the reports means this |
| Median seed (weights chart) | The upper median, the 6th of 10 seeds ranked by learning gap. A chart needs one real seed's weights, so the two middle seeds cannot be averaged |

### Convergence and optimality rules

| Rule | Uses | Definition |
|---|---|---|
| `greedy_stable` | agent only | First episode at which the greedy action has been unchanged for K consecutive episodes |
| `greedy_stable_through_end` | agent only | The final greedy action has been held for at least K episodes, counted from when it last changed |
| `dq_below_tol_for_good` | agent only | Read at checkpoints. First checkpoint from which every later full window had max \|change in Q\| below the tolerance. A short final window is ignored, because few updates make it look artificially calm |
| `near_optimal_for_good` | benchmark (evaluation only) | Read at checkpoints. First checkpoint from which the greedy portfolio's CE loss vs. w_disc stays at or below the near-optimal line |

For each rule, the summary reports the share of seeds that met it and the
median, minimum and maximum observations needed. That answers Part IV ("how
many observations and episodes are required... how often convergence occurs
across seeds") and the Part V requirement on "the number of observations needed
to approach a clearly defined level of optimality."

The two `greedy_stable` rules use the exact episode of every greedy change
(`train.py` logs each one), so they are resolved to the episode.

### Post-hoc convergence measures (added 2026-11-09)

**In plain terms:** the primary rule asks for the final pick to be held for 10%
of training, which a seed that settles late can miss even though it ends on the
right portfolio. These two measures describe when each seed actually settled.
They were defined after the baseline results were seen (`RUN_ISSUES_2026-10-22.md`
Findings 1 and 2), so they are descriptive and never replace the
pre-registered rules. `summary.json` lists them under
`convergence.descriptive_rules`.

| Measure | Definition |
|---|---|
| `settled` | First checkpoint from which the final greedy action is the greedy action at every later checkpoint. `None` if it is not greedy at the last checkpoint. No K window, so it can be met by a seed that fails `greedy_stable_through_end`. For a tabular agent the last checkpoint's greedy action is the final one by definition, so every seed settles (at the latest, at the last checkpoint): the share of seeds is always 100% and only the timing carries information. `INDEX.md` therefore shows the median and range, not a share. |
| `near_optimal_every_episode` | First episode after which every greedy action, brief switches between checkpoints included, stays within the near-optimal line. The every-episode counterpart of `near_optimal_for_good`. A seed whose final greedy action is within the line always meets it, however late its last switch; read the episode, not only the share |

### Provenance (added 2026-11-09)

`summary.json` has a `provenance` block: the git commit the evaluation ran at
(with `-dirty` if the working tree had changes), the SHA-256 of `evaluate.py`,
`config.py`, `run_experiments.py`, `hashing.py` and the benchmark solver
`benchmarks/mean_variance.py`, and the Python, numpy,
scipy, matplotlib and pandas versions. Digests are over line-ending-normalized
bytes (`src/hashing.py`), so the same file gives the same digest on Windows and
on the Mac. Training manifests use the same digest from 2026-11-09;
manifests written on Windows before then hold the CRLF digest, which
`load_manifest` still accepts.

### Q-value fit (post hoc, added 2026-11-04)

**In plain terms:** these say whether the agent learned the whole reward
landscape or only found a good top pick. They were added for the
function-approximation comparison, after the baseline results were seen, and
are computed for tabular runs too so the two can be compared like for like.

The reference for each portfolio is its true expected reward. The reward's
variance uses ddof = 1, so with Gaussian returns the expected reward is
exactly U(w); no adjustment is needed.

| Name | Definition |
|---|---|
| `q_fit_all` | Over every portfolio on the grid: Pearson and Spearman correlation of the final Q̂ with U(w), and top-10 recall, the share of the true top 10 portfolios that are also in the learned top 10 (ties in Q̂ to the lowest index) |
| `q_fit_visited` | The same, over the portfolios the agent tried at least once. A tabular agent never updates an untried portfolio, which keeps it at Q̂ = 0. On the 20% grid every portfolio is tried within the training budget, so this equals `q_fit_all` there; the two differ on the 10% grid or with short budgets |
| `optimism_of_learned_bps` | 252 × (Q̂ − U) × 10,000 for the learned portfolio, in bps/yr. Positive means the agent overrates what it picked |

Read optimism against its noise. For the tabular agent with α = 1/N(a), Q̂ is
a sample mean, so its standard error is the reward SD divided by √(visits):
about 0.7 bps/yr at B = 63 and 1.2 bps/yr at B = 20 for a portfolio visited
about 11M times.

With a constant α (run `tabular-const`) Q̂ is no longer a sample mean. Starting
from 0, a portfolio updated N times holds about (1 − (1 − α)^N)·U. At α = 3.2e-4
the median portfolio, tried about 1,100 times during exploration, sits at about
0.3·U, so `tabular-const`'s correlations and top-10 recall mostly measure that
shrinkage (`EXPERIMENTS.md` §8.7). The FA agents start θ at 0 and shrink the
same way in directions exploration did not finish learning.
Seeds that differ only in γ share their return path, so their optimism values
move together and do not count as independent evidence.

Correlations are `None` (blank in the CSV) when fewer than two portfolios are
compared or either side is constant. The summary gives their spread across
seeds under `q_fit`; `per_seed.csv` has one column per seed-level value.

### Evaluation sample

The evaluation draws come from `SeedPlan.evaluation_rng()`, a stream that
training never touches. One sample is drawn and every portfolio of every seed is
scored on it (common random numbers). The exact values from μ and Σ are the
primary figures. The realized values are a check, and the paired shortfall
against w_disc, with its standard error, is the reliable realized comparison.

---

## 3. Decisions and assumptions

| Decision | Choice | Why | Not taken |
|---|---|---|---|
| Main gap measure | CE loss in bps/yr | Keeps sign and meaning at any γ; easy to explain | Relative gap as main measure: flips sign once U ≤ 0 (from γ ≈ 23 on this calibration) |
| Learned portfolio | Greedy action of the final Q, per seed | The policy the agent would follow after training | Averaging weights across seeds: produces an off-grid blend that can beat the best grid action |
| Ties | Lowest index | Deterministic, and matches what `train.py` logs | Random tie-break: would make evaluation depend on another random stream |
| ΔQ rule | Must hold for good, over full windows only | A single quiet window, or a short last window, is not convergence | First window below the tolerance: fired early and then reverted in the demo |
| Evaluation sample | One shared sample, 10,000 episodes in the demo | Common random numbers make paired differences precise | Separate samples per portfolio: much noisier comparisons |
| Grid check | Refuse if the grid fingerprint in the manifest differs | An action index must mean the same portfolio it meant in training | Size check only |
| Data check | Refuse if the calibration files changed since training | The benchmarks must describe the market the agent learned in | Warning only |
| Q-fit reference (2026-11-04) | U(w), the exact expected reward | ddof = 1 makes the reward unbiased for U(w) | Realized evaluation rewards: add noise to a quantity known exactly |
| Q-fit populations | All portfolios and visited portfolios, both reported | All portfolios is the natural FA question; visited portfolios keeps the tabular agent's untried zeros from deciding its score | Visited only: would hide that FA scores portfolios it never tried |

> **Assumption:** the learned policy is evaluated with exploration off. The
> assignment's "steady-state evaluation sample" is read as the policy the agent
> converged to, not the exploring behavior during training. The final ε of each
> run is reported so this can be checked.

---

## 4. Verification

| Check | Result |
|---|---|
| Pure functions (objective, CE loss, relative gap, regret, every convergence rule) | Hand-computed examples |
| Realized statistics | Recomputed directly, including standard errors and paired shortfall |
| Agent that always holds the best grid portfolio | Loss 0, regret 0, flagged as exactly best, every seed |
| Agent that holds a known worse portfolio | Loss, relative gap, L1, Euclidean, absolute differences and regret match independent calculations |
| Agent whose greedy action switches at a known episode | Every convergence episode exactly as expected |
| γ = 60, every objective negative | Worst portfolio shows a large positive loss, never near-optimal; relative gap flagged unreliable |
| Evaluation sample | Rebuilt independently from the evaluation stream; uncorrelated with training draws |
| CSV contents | Match the summary |
| Refusals | Changed data, changed grid, partial experiment, overwrite, comparison of experiments differing in more than the grid |
| Q-fit metrics (2026-11-04) | Hand examples; correlations match `scipy.stats.pearsonr` and `spearmanr` with ties; an agent holding Q = U scores 1, 1, 1 and optimism 0; Q = U + c gives optimism exactly c; a fixed agent's optimism rebuilt from its logged window rewards; CSV columns match the summary |
| Post-hoc measures and total gap (2026-11-09) | An agent whose greedy action switches at episode 1,200 settles at checkpoint 1,500 and is near-optimal every episode from 1,200; total gap equals discretization plus learning gap per seed and matches an independent U(w*) − U(w) |
| Line endings (2026-11-09) | LF and CRLF copies of a file get the same digest; a manifest whose calibration digests are the CRLF form is accepted |
| Re-evaluating existing runs (2026-11-09) | `baseline-v1`, `n26m`, `all-three`, `grid10-robustness` and `fa-spline` re-evaluated: every earlier value in `summary.json`, `comparison.json` and the CSVs identical, figures byte-identical. The total gap matches `RUN_ISSUES_2026-10-22.md` Finding 3 (19.8, 14.3, 12.5 and 11.9 bps/yr medians) |
| Re-evaluating existing runs (2026-11-04) | `b63` and `eps-decay` (6 experiments) re-evaluated with the new code: every earlier number in `summary.json`, `weights.csv` and `performance.csv` identical |
| Mutation testing | 17 deliberate bugs injected; 16 caught. The survivor changes only which weights the bar chart draws, which no test inspects |

Full suite: 225 passed when this table was written (2026-10-16); 373 after the Q-fit metrics (2026-11-04).

---

## 5. Review history

An independent agent review on 2026-10-16 found the per-seed arithmetic
correct, then found these, all fixed before commit:

| Finding | Fix |
|---|---|
| The "learned" portfolio averaged weights across seeds and scored above the best grid action | Per-seed reporting; the average kept only as a labelled descriptive extra |
| Relative gaps flip sign or blow up when U ≤ 0 | CE loss in bps/yr as the main measure; relative gap secondary and flagged |
| The ΔQ rule fired on one quiet window and on short final windows | Must hold for good, full windows only |
| Realized statistics were mostly noise | Exact values primary; paired shortfall with standard error added |
| Resolution comparison had no consistency check | Refuses experiments that differ in anything but the grid |
| Grid checked by size only; partial experiments accepted silently | Grid fingerprint (now written by `train.py`); partial runs refused unless allowed |
| Early spikes dominated the loss charts; label clashes; a reused color | Clipped y-axis with a note; separate colors for resolutions; "run i" labels |
| 18 of 21 mutations survived the first tests | Rewritten tests with known-behavior agents; 16 of 17 caught |

---

## 6. Illustrative demo (not results)

These came from William's agent with γ = 5, B = 20, 2 million episodes and 3
seeds. All of those are test values, chosen before the decisions in §7 were
made, so these numbers show the pipeline working and are not findings.

| | 25% grid (495) | 20% grid (1,287) |
|---|---|---|
| Discretization gap | 11.7 bps/yr | 5.7 bps/yr |
| Learning gap, median (range) | 5.1 (0 to 20.6) bps/yr | 10.5 (0 to 15.9) bps/yr |
| Seeds that found the exact best | 1 of 3 | 1 of 3 |

---

## 7. Settings (fixed 2026-10-16)

γ ∈ {3, 5, 8} with 5 as baseline, B = 20, and K = 10% of training episodes
are team decisions. The primary convergence rule is
`greedy_stable_through_end`: the final greedy action held for at least K
episodes. The ΔQ tolerance (1e-5) and the near-optimal line (25 bps/yr) are
adopted defaults. All live in `src/config.py`; the run procedure is in
`EXPERIMENTS.md`. The written justification for γ was added 2026-10-19
(`DATA_CLEANING.md` §8.4).

Under a constant learning rate (the §8 runs in `EXPERIMENTS.md`) the ΔQ rule
is met in the first window by every seed, because one update moves a value by
about 5e-7, far below the 1e-5 tolerance. It says nothing about convergence
there and should not be reported for those runs.

The 25 bps line is a fixed number of basis points, so it is a tighter bar at
some γ than others: it is 3.5%, 4.8% and 6.7% of the best grid portfolio's
value at γ = 3, 5 and 8, and it contains 2, 4 and 2 of the 1,287 portfolios on
the 20% grid. Near-optimal rates across γ should be compared with that in
mind.
