# 04 Agent and training (working notes)

**TL;DR / In plain terms:** the baseline agent is a stateless table with one value per portfolio, updated with step 1/N(a), no discounting, zero start and ε-greedy exploration that decays exponentially. Settings were committed on 2026-10-16 and the first full run was 2026-10-22. The values are recorded; the reasons for several of them are not written anywhere a grader can read.

Sources: `src/agent.py`, `src/train.py`, `src/config.py`, `docs/methods/EXPERIMENTS.md` §1 to §5, `docs/reviews/REVIEW_HISTORY.md` §4, §6.

## Settings (all in `src/config.py`; every manifest records them)
| Item | Value | Written reason found |
|---|---|---|
| γ | 5 (3, 8 robustness) | Yes, `DATA_CLEANING.md` §8.4 |
| B | 20 days | One phrase: "a monthly holding period" |
| Episodes | 13,000,000 per seed (260M days) | Only REVIEW_HISTORY §4: a 13M demo "got within roughly 2% of the best menu portfolio". Benchmark-based sizing, made before the settings were fixed |
| Seeds | 10, master 20261016 | "Proposal, adopted"; no reasoning |
| α | 1/N(a) | Sample-average, standard for a stationary bandit; stated |
| ρ | 0 | Assignment's "clean bandit-like benchmark"; stated |
| Q0 | 0 | Stated |
| ε | ε_t = exp(−βt), ε0 = 1, ε = 1% at halfway, β ≈ 7.08e-7 | Value stated, choice of 1% at halfway not argued |
| Tie-break | random among maxima when acting; lowest index for logging and evaluation | Stated |
| K | 10% of episodes | Not argued |
| ΔQ tolerance | 1e-5 per 10,000-episode window | Not argued |
| Near-optimal line | 25 bps/yr | EVALUATION.md gives it as 3.5 / 4.8 / 6.7% of best grid value |
| Evaluation | 10,000 fresh episodes | Stated |

Random tries per portfolio under the schedule: about 1/β ÷ actions = 1.41M ÷ 1,287 ≈ 1,097 at baseline.

## Pre-registration order (checked in git)
- `src/config.py` fixed in `07bfb64` (2026-10-16).
- Baseline trained at `80a6e9c` (2026-10-22 01:26).
- Post hoc tabular runs registered 2026-10-27, run 10-28/29.
- Constant-α and FA runs registered 2026-11-05, run at `bc311a3` on 2026-11-06.
- FA on the 10% grid registered in `c0e5494` (2026-11-11 03:31), run the same day.
- Disclosure: an illustrative demo (γ = 5, B = 20, 2M episodes) ran before the settings were fixed (EVALUATION.md §6, EXPERIMENTS.md §5).

## Leakage
- The agent module takes rewards and action indices only. Tests enforce import allowlists.
- Benchmark information did enter two human design choices: the 13M budget (above) and the 25 bps line expressed against best grid value. Neither reaches the agent. Both need one sentence in the report.

## Gaps against "Report and justify" (assignment Part III)
B, episode count, seed count, the ε schedule's 1%-at-halfway and K have values but no argument in any doc. This is finding H1 in the audit.
