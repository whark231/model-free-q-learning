# Parts IV and V: Evaluating the Learned Policy

**Owner:** Albert · **Status:** built and tested on `dev`; rules and thresholds fixed 2026-10-16 (§7)
**Code:** `src/evaluate.py` · **Tests:** `src/tests/test_evaluate.py` (24 tests)
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
| `per_seed.csv` | Learned action, losses, distances, regret, observations to each convergence rule |
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
| Relative gap | (U_ref − U) / \|U_ref\|; reported as secondary, and `None` when \|U_ref\| < 1e-5 per day |
| Expected regret | Sum over episodes of U(w_disc) − U(chosen action), computed from how often each action was chosen in each window |
| Observations | Episodes × episode length B, in simulated days |

### Convergence and optimality rules

| Rule | Uses | Definition |
|---|---|---|
| `greedy_stable` | agent only | First episode at which the greedy action has been unchanged for K consecutive episodes |
| `greedy_stable_through_end` | agent only | The final greedy action has been held for at least K episodes, counted from when it last changed |
| `dq_below_tol_for_good` | agent only | First checkpoint from which every later full window had max \|change in Q\| below the tolerance. A short final window is ignored, because few updates make it look artificially calm |
| `near_optimal_for_good` | benchmark (evaluation only) | First checkpoint from which the greedy portfolio's CE loss vs. w_disc stays at or below the near-optimal line |

For each rule, the summary reports the share of seeds that met it and the
median, minimum and maximum observations needed. That answers Part IV ("how
many observations and episodes are required... how often convergence occurs
across seeds") and the Part V requirement on "the number of observations needed
to approach a clearly defined level of optimality."

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
| Mutation testing | 17 deliberate bugs injected; 16 caught. The survivor changes only which weights the bar chart draws, which no test inspects |

Full suite: 225 passed.

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
`EXPERIMENTS.md`. Still open: the written justification for γ
(`DATA_CLEANING.md` §8.4).

The 25 bps line is a fixed number of basis points, so it is a tighter bar at
some γ than others: it is 3.5%, 4.8% and 6.7% of the best grid portfolio's
value at γ = 3, 5 and 8, and it contains 2, 4 and 2 of the 1,287 portfolios on
the 20% grid. Near-optimal rates across γ should be compared with that in
mind.
