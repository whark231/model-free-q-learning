# Notes for William

**From:** Albert
**Date:** 2026-10-15 (first written 2026-10-13)
**Status:** every issue from both review rounds is fixed in the code and
independently re-checked (§1). §2 and §3 are the work split and open
decisions. **New in this update: `src/train.py` is on `dev`, built against
your interface, with four items for you in §5.** Nothing in your files has
been changed.

---

## TL;DR / In plain terms

Your fixes for A, B and C work, and your scale correction to my proposed KKT
check was right: my absolute threshold would have failed correct answers on
rescaled inputs.

For the agent, I'm proposing we split by file instead of building everything
side by side. You write `src/agent.py`, the part that picks portfolios and
updates Q-values. I write `src/train.py`, the part that runs episodes, manages
seeds and saves results. We agree the interface between them first, in a
short session that also settles the §6 decisions. Each of us then works on
our own branch and reviews the other's PR. Your next-steps note keeps the
core paired, so this is a proposal, not a done deal: tell me if you'd rather
stay fully paired.

---

## 1. Review issues: all closed

| # | Issue | Fixed in | Re-checked by Albert (2026-10-13) |
|---|---|---|---|
| 1 | SLSQP stopped early on daily moments | `51b3eb7` | Matches an exact solve to 3e-6 in every weight |
| 2 | `best_discrete_weights` hardcoded the grid | `51b3eb7` | `increment` and `grid` arguments work |
| 3 | Float drift in grid weights | `02f6a49` | Regression test |
| 4 | `n = mu.shape` tuple | `51b3eb7` | |
| 5 | `.vscode/settings.json` Windows path | `51b3eb7` | Untracked |
| A | Reward NaN for a one-day batch | `0e2201d` | Raises on 1-day, empty, 1-D and wrong-width batches |
| B | `best_discrete_weights` returned a view | `0e2201d` | Returns a copy |
| C | `nit <= 1` guard | `0e2201d` | See below |

Checks of your relative KKT guard:

- **Real calibration.** γ from 0.25 to 30 in steps of 0.25, with inputs at
  daily, annual, and 1/252-of-daily scale. One alarm, at γ = 4.25 on the
  1/252 scale, and it was justified: that solve's weights were off by 7e-4
  from a solve at annual scale. Nothing fired at realistic scales.
- **200 random 9-asset problems.** No false alarms. Every weight within 3e-6
  of an exact solution found by enumerating which assets are held.
- **Deliberately bad solves** (the old default tolerance at daily scale) at
  γ = 1, 5 and 15. All three caught.

Full suite on `dev` at `0e2201d`: 149 passed.

---

## 2. Proposal: splitting the tabular Q-learning core

### 2.1 The interface (agree first, merge first)

The only place our code touches:

```python
agent = TabularQAgent(n_actions, *, alpha_rule, epsilon0, beta, q_init, discount, rng)
a = agent.select_action()          # epsilon-greedy, eps_t = epsilon0 * exp(-beta * t)
agent.update(a, reward)            # Q(a) <- Q(a) + alpha * [R + rho * max Q - Q(a)]
agent.greedy_action                # current best action index
agent.q_values, agent.visit_counts # read-only copies, for logging
```

One episode in `train.py`:

```python
a = agent.select_action()
w = grid[a]
batch = sim.sample_episode(B)                       # environment.py
r = episodic_mean_variance_reward(w, batch, gamma)  # rewards.py
agent.update(a, r)
```

The agent receives an action index and a reward, never returns, μ, Σ or a
benchmark. The leakage tests in `test_environment.py` already guard
`src/agent.py`.

Proposed first step: a tiny PR adding the `TabularQAgent` signature and
docstrings to `agent.py`, with method bodies raising `NotImplementedError`. We
both build on it, so neither of us guesses the other's method names.

### 2.2 Who owns what

| | William | Albert |
|---|---|---|
| File | `src/agent.py` | `src/train.py`, later `src/evaluate.py` |
| Branch | `feature/agent` | `feature/train-loop` |
| Builds | Q-table, epsilon-greedy with decay, learning-rate rule, discount (ρ = 0 baseline), Q initialization, random tie-breaking from `agent_rng` | Run and episode loop using `SeedPlan` and `ReturnSimulator`; per-episode log (action, reward, ε, greedy action, largest Q change); Q checkpoints; run manifest (seeds, git commit, calibration file hashes, γ, B, episode count); outputs under `results/` |
| Tests | `test_agent.py`: toy bandits with known means where the agent must find the best arm; exact update arithmetic; ε schedule; tie-breaking; NaN rewards rejected; existing leakage guard passes | `test_train.py`: same seed gives an identical run; evaluation draws never used in training; log shapes; manifest complete; stopping rule uses only agent-observable quantities |
| Stand-in while the other half is unfinished | A short loop on fake rewards | `RandomAgent` and a sample-average agent with the same interface |
| Assignment coverage | Part III "report and justify": learning rate, ε₀ and β, discount, Q init, tie-breaking | Part IV: seeds, observation and episode counts, convergence; feeds my Part V analysis |

### 2.3 Rules that keep us out of each other's way

1. Separate files and separate branches, both cut from the latest `dev` and
   merged by PR into `dev`. I touch `agent.py` only through its interface.
2. Interface PR merges before either feature branch.
3. Shared files (README, `requirements.txt`, the prospectus) change in their
   own small PRs, with one owner per change.
4. The output format (one `.npz` log plus one `manifest.json` per run) goes in
   the interface docstring, since `evaluate.py` depends on it.
5. Once both are merged, an integration run on the 25% grid (495 actions),
   fixed seed, a few thousand episodes, rerun to confirm identical output.
6. Review swap: I review `agent.py` line by line and you review `train.py`.
   That keeps the prospectus goal that we both understand the core.

---

## 3. Decisions needed before coding

Values can sit in a config and be filled in later. The definitions cannot:
the assignment says to "define convergence before examining the final
results."

| Decision | Owner | Albert's proposal | Reason |
|---|---|---|---|
| γ | William | Your call, written down before any results (`DATA_CLEANING.md` §8.4) | Avoids tuning the experiment to its outcome |
| Episode length B | Both | Baseline 20 (a trading month); 5 and 63 as one Part VI robustness check | Interpretable as monthly rebalancing; fits "a short batch of returns"; leaves room to vary both ways; makes the transaction-cost extension natural |
| What the agent sees | Both | Rewards only | A 9-asset return batch would let it score all actions at once or estimate μ and Σ, which stops being standard Q-learning |
| Learning-rate rule | Both | Sample average, α = 1/N(a), as the baseline | With constant α the Q-values never settle: at α = 0.01 they keep jittering by about half the objective |
| Discount ρ | Both | 0 | The assignment's "clean bandit-like benchmark" for a stateless problem |
| Convergence rule | Both | Stop on agent-observable quantities only (greedy action unchanged for K episodes, or largest Q change below a tolerance). Report the gap to the best menu action afterwards, in evaluation | The gap needs μ and Σ, so letting it stop training would leak benchmark information |
| ε₀, β, episode count | Both | ε₀ = 1; set β from the episode budget so exploration fades over roughly the first half of training | Ties the schedule to the budget instead of a magic number |
| Q initialization, tie-breaking | William | Q = 0; ties broken uniformly at random with `agent_rng` | With 1/N averaging the first visit overwrites the initial value; random ties avoid a bias toward low action indices |
| Seed count | Both | 10 | The assignment asks for "multiple random seeds"; about 5 minutes per 13M-episode run at B = 20 on the 20% grid (corrected, see §5) |
| Second robustness check | Both | Episode length (5 / 20 / 63), alongside the 20% vs. 10% grid comparison | Both are on the assignment's Part VI list and need no new code |

**Prospectus edit to agree:** if we adopt the split, §4's row "Tabular
Q-learning core: William + Albert (paired)" becomes "Agent (`agent.py`):
William; training runner (`train.py`): Albert; each reviews the other's PR."

---

## 4. Numbers to keep in mind for the agent

- **Noise.** At γ = 5, B = 20, one episode's reward for the best menu
  portfolio has a standard deviation 7.5 times its true value. The best and
  second-best menu portfolios differ by 2%.
- **Budget.** In an idealized test with 1/N averaging and uniform exploration,
  about 13 million episodes (260 million simulated days) got within roughly 2%
  of the best menu portfolio on a typical seed. It found the exact best only
  about a quarter of the time.
- **Lock-in risk.** In a 3-asset toy run with fast-decaying ε, the agent
  settled on the second-best portfolio (1% short of the best) because the
  best one got an unlucky early average and was rarely retried.
- **Speed.** The draw and reward alone cost about 11 µs per episode (3.8 µs
  draw plus 7.6 µs reward call); the full loop with an agent is slower (§5).
  Each episode's mean vector and covariance can be computed for a whole block
  in advance, then reward = w'm − (γ/2)w'Sw, which is exact.
- **Observation counts** for Part IV should come from the training loop
  (episodes used × B). `ReturnSimulator.days_simulated` also counts days
  drawn in advance by block sampling.

---

## 5. `src/train.py` is on `dev`: four items for you

The runner is built against the interface in `src/agent.py` and tested with
stand-in agents (30 tests in `src/tests/test_train.py`; full suite 179
passed). Every reward and every logged number is rebuilt independently in the
tests and compared exactly. An independent review filled in your
`feature/agent` TODOs in a scratch copy and confirmed the runner produces
exactly the same Q-table as a bare five-line loop. Once your agent lands, one
command runs a real experiment:

```bash
python -m src.train --gamma 5 --episode-length 20 --episodes 200000 --master-seed 2026 --runs 2 --beta 2e-5 --out results/demo
```

γ, B, episode count, master seed, run count and β are required arguments
with no defaults, because they are still open decisions (§3).

### 5.1 The learning-rate visit count (needs your decision)

`train.py` passes `alpha_rule` as a function of n, where **n counts the update
being made**: the first update of an action uses α = 1/1, which replaces
`q_init` with the first reward. Your `_alpha` TODO leaves the order open
("does the first update use alpha_rule(0) or alpha_rule(1)?").

If `_alpha` calls the rule before incrementing the visit count, the first call
is `1 / np.int64(0)`, which numpy turns into `inf` with only a warning. Q
becomes inf or NaN and, since `np.argmax` treats NaN as the largest value,
that action becomes the permanent greedy choice.

What `train.py` now does about it: the sample-average rule raises
`ValueError` for n < 1, and the loop stops with `FloatingPointError` if any
Q-value becomes non-finite.

**Proposed:** increment `_visits[action]` first, then call
`alpha_rule(self._visits[action])`. That matches the sample-average
definition (after n rewards, Q is their mean) and the runner's convention.

### 5.2 `greedy_action` draws from the agent's random stream

Your TODO breaks ties with `self.rng.choice(...)`, which is the right rule for
choosing actions. It also means that merely reading the property consumes a
random draw, so any logging code that reads it would change the agent's later
choices. `train.py` never calls it; it logs its own lowest-index argmax of
`q_values`, for logging only. Two small proposals:

- note in the property's docstring that reading it consumes randomness;
- add a read-only `epsilon` property (the rate the next `select_action` will
  use), so the runner can log it instead of recomputing
  `epsilon0 * exp(-beta * t)` from the config. Today the two agree exactly,
  but a later change to the schedule (a floor, say) would make the recomputed
  log silently wrong.

### 5.3 Speed: my earlier estimate was wrong

The "about 2.5 minutes per 13M-episode run" in the previous version of these
notes is wrong. With your agent filled in from its TODOs, the reviewer
measured:

| Grid | Actions | Runner | Bare loop | Per 13M-episode run |
|---|---|---|---|---|
| 20% | 1,287 | 23.6 µs/episode | 19.7 µs | about 5 minutes |
| 10% | 43,758 | 91 µs/episode | 51.6 µs | about 20 minutes (3.3 hours for 10 seeds) |

Most of the runner's overhead is copying the full Q-table through
`q_values` after every update (to log the change in Q and track the greedy
action). **Proposed interface addition:** a scalar accessor such as
`q_value(action) -> float`, or have `update` return the new Q(action). The
runner could then track the greedy action incrementally and skip the copy.
Not urgent at the 20% grid; worth it before the 10% robustness runs.

### 5.4 The log format differs from §2.2

§2.2 promised a per-episode log (action, reward, ε). At 13M episodes per run
that is hundreds of MB per run, so `train.py` saves instead:

- every `log_every` episodes (default 10,000): mean reward, largest |change in
  Q|, logged greedy action, per-action choice counts, ε for the next episode,
  and a float32 Q-table snapshot;
- every change of the logged greedy action, with its episode number, starting
  at episode 0;
- a `manifest.json` with the config, seed plan, SHA-256 of the calibration
  files and the code, git commit and library versions.

That is enough for the learning curve, exact expected regret per window
(choice counts × gap to the best menu action, computed in evaluation), "greedy
unchanged for K episodes" at any K, convergence frequency across seeds, and
the final greedy portfolio to score on the evaluation stream. It is not enough
for a per-episode plot of every reward; if you need that for the report, say
so and I'll add an optional full log for short runs.

Training always runs its full budget. Nothing stops early, and the runner
never imports the benchmarks or touches the evaluation stream (both tested),
so convergence is judged afterwards from these logs.

