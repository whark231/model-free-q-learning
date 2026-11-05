"""The experiment settings, in one place.

Team decisions (2026-10-16):

- Risk aversion γ ∈ {3, 5, 8}. γ = 5 is the baseline for every headline
  table; γ = 3 and γ = 8 are the Part VI risk-aversion robustness check.
- Episode length B = 20 trading days (a monthly holding period).
- Convergence: the greedy action is unchanged for K = 10% of the training
  episodes (the "greedy action remains unchanged for a prespecified number of
  episodes" rule, assignment Part IV). The primary reading is that the FINAL
  greedy action held for at least K episodes through the end of training
  (`greedy_stable_through_end` in evaluate.py). The looser "first time any
  action held for K" is reported as secondary.

Defaults proposed by Albert and adopted with the decisions above, all easy to
change here (REVIEW_HISTORY.md §3):

- 10 seeds, 13 million training episodes per seed (260 million simulated days).
- Exploration ε_t = exp(-β t), with β set so ε falls to 1% halfway through
  training.
- |change in Q| tolerance 1e-5 per window, and a near-optimal line of 25 bps
  per year of certainty-equivalent loss against the best grid portfolio.
- Grids: 20% (1,287 portfolios) baseline, 25% (495) as the second resolution
  Part V requires.

Every value here was fixed before any real experiment ran. Change them only
with a recorded reason, never because of how a run looked.

Post-hoc runs (added 2026-10-27, after the baseline run of 2026-10-22, tag
`baseline-v1`): each is a `RunVariant` in `POST_HOC_RUNS`, chosen with
`python -m src.run_experiments --run <run_id>`. A variant changes B, the
training budget, the ε schedule or the grid, and leaves every constant above
untouched, so running without `--run` still reproduces `baseline-v1` exactly.
Each variant's settings and expected direction are written in EXPERIMENTS.md
§7 before it runs.

Constant-learning-rate runs (registered 2026-11-05, after the professor
agreed to a constant alpha for function approximation and a separately tuned
constant alpha for the tabular comparison): a variant can also change the
agent (`agent`, `feature_map`, `l2`, `greedy_refresh_every`) and the learning
rate (`alpha_rule`). Every alpha and l2 below was chosen by src/tune_fa.py on
tuning seeds (master seed 20261103), recorded in results/tuning/fa_tuning.json,
never on the reported seeds. EXPERIMENTS.md §8 holds the full registration.
"""

import math
from dataclasses import dataclass

from src.evaluate import EvalConfig
from src.train import TrainConfig

GAMMA_BASELINE = 5.0
GAMMA_ROBUSTNESS = (3.0, 8.0)
GAMMAS = (3.0, 5.0, 8.0)

EPISODE_LENGTH = 20
N_EPISODES = 13_000_000
N_RUNS = 10
MASTER_SEED = 20261016  # the decision date; fixed and recorded in every manifest

CONVERGENCE_K_FRACTION = 0.10
PRIMARY_CONVERGENCE_RULE = "greedy_stable_through_end"
EPSILON_AT_HALFWAY = 0.01
DQ_TOL = 1e-5
NEAR_OPTIMAL_BPS = 25.0
N_EVAL_EPISODES = 10_000

# Agent settings settled earlier (REVIEW_HISTORY.md §3.3).
EPSILON0 = 1.0
ALPHA_RULE = "sample_average"  # alpha = 1/N(a)
Q_INIT = 0.0
DISCOUNT = 0.0

INCREMENT_BASELINE = 0.2
INCREMENT_COMPARISON = 0.25
LOG_EVERY = 10_000


def convergence_k(n_episodes: int) -> int:
    """K = 10% of the training episodes, rounded up."""
    return math.ceil(CONVERGENCE_K_FRACTION * n_episodes)


def beta_for(n_episodes: int, epsilon_at_halfway: float = EPSILON_AT_HALFWAY) -> float:
    """β such that ε = exp(-β t) equals epsilon_at_halfway at t = n_episodes / 2."""
    return math.log(1.0 / epsilon_at_halfway) / (n_episodes / 2)


@dataclass(frozen=True)
class Experiment:
    """One training run set: a name for its results folder, plus its settings.
    The last three default to the baseline; only post-hoc variants change them."""

    name: str
    gamma: float
    increment: float
    purpose: str
    episode_length: int = EPISODE_LENGTH
    epsilon_at_halfway: float = EPSILON_AT_HALFWAY
    n_episodes: int = N_EPISODES
    alpha_rule: str | float = ALPHA_RULE
    agent: str = "tabular"
    feature_map: str | None = None
    l2: float = 0.0
    greedy_refresh_every: int = 1


@dataclass(frozen=True)
class RunVariant:
    """A post-hoc change to the baseline, run at all three γ on one grid.

    suffix goes on every experiment name (baseline_g5_grid20_<suffix>) so its
    folders can never be mistaken for the baseline's. expected is the
    direction written down before the run, registered the date it was. It is
    the one-line summary INDEX.md prints; EXPERIMENTS.md §7 (§8 for the
    constant-learning-rate runs) holds the full expectation.
    """

    run_id: str
    suffix: str
    registered: str
    expected: str
    episode_length: int = EPISODE_LENGTH
    n_episodes: int = N_EPISODES
    epsilon_at_halfway: float = EPSILON_AT_HALFWAY
    increment: float = INCREMENT_BASELINE
    alpha_rule: str | float = ALPHA_RULE
    agent: str = "tabular"
    feature_map: str | None = None
    l2: float = 0.0
    greedy_refresh_every: int = 1

    def changes(self) -> dict:
        """The settings this variant moves away from the baseline, as {name: (baseline, variant)}."""
        return {k: (v, getattr(self, k)) for k, v in BASELINE_RUN_SETTINGS.items() if getattr(self, k) != v}


# What a variant can change, at its baseline value. The last five are the
# learning rate and the agent; tabular runs keep their defaults.
BASELINE_RUN_SETTINGS = {
    "episode_length": EPISODE_LENGTH, "n_episodes": N_EPISODES, "epsilon_at_halfway": EPSILON_AT_HALFWAY,
    "increment": INCREMENT_BASELINE, "alpha_rule": ALPHA_RULE, "agent": "tabular", "feature_map": None,
    "l2": 0.0, "greedy_refresh_every": 1,
}
AGENT_SETTINGS = ("alpha_rule", "agent", "feature_map", "l2", "greedy_refresh_every")

# Tuned on master seed 20261103 by src/tune_fa.py (results/tuning/fa_tuning.json,
# FA_INTEGRATION_NOTES.md §5): each value is a fraction of the model's stability
# bound 2 / max||phi||^2, which is 2 for tabular and quadratic and 2/45 for splines.
TUNED_ALPHA_TABULAR = 0.00032  # fraction 1.6e-4 of 2
TUNED_ALPHA_QUADRATIC = 0.00032  # fraction 1.6e-4 of 2
TUNED_ALPHA_SPLINE = 7.111111111111112e-06  # fraction 1.6e-4 of 2/45; l2 = 0 won on 6 seeds (close call)
FA_GREEDY_REFRESH = 100  # FA_INTEGRATION_NOTES.md §3


# Every post-hoc run, by run_id. Each run branch adds its own entry.
POST_HOC_RUNS: dict[str, RunVariant] = {
    "b63": RunVariant(
        "b63", "B63", "2026-10-27",
        "a smaller learning gap at every γ; exactly-best share, training regret and discretization gap about unchanged",
        episode_length=63,
    ),
    "n26m": RunVariant(
        "n26m", "N26M", "2026-10-27",
        "a slightly smaller learning gap and a slightly higher exactly-best share at every γ, likely within noise",
        n_episodes=26_000_000,
    ),
    "eps-decay": RunVariant(
        "eps-decay", "eps3", "2026-10-27",
        "learning gap and exactly-best share unchanged within noise; training regret about 15 bps/yr higher at γ = 5",
        epsilon_at_halfway=0.03,
    ),
    "all-three": RunVariant(
        "all-three", "B63_N26M_eps3", "2026-10-27",
        "the smallest learning gap of runs 1 to 4 at every γ, mostly from B; training regret a little above b63's",
        episode_length=63, n_episodes=26_000_000, epsilon_at_halfway=0.03,
    ),
    "grid10-robustness": RunVariant(
        "grid10-robustness", "B63_N26M_eps3", "2026-10-27",
        "a larger learning gap against the best grid portfolio than run all-three's at every γ, exactly-best share "
        "near zero, discretization gap 5.7 -> 1.3 bps/yr at γ = 5 and unchanged at γ = 3 and 8",
        episode_length=63, n_episodes=26_000_000, epsilon_at_halfway=0.03, increment=0.1,
    ),
    "tabular-const": RunVariant(
        "tabular-const", "TABconst", "2026-11-05",
        "a larger mean learning gap than baseline-v1 at every γ, and fewer seeds meeting the primary convergence "
        "rule at every γ (baseline-v1: 9, 8, 9 at γ = 5, 3, 8), because a constant alpha never averages the reward "
        "noise away (Q-value noise about 49 bps/yr)",
        alpha_rule=TUNED_ALPHA_TABULAR,
    ),
    "fa-quadratic": RunVariant(
        "fa-quadratic", "FAquad", "2026-11-05",
        "a smaller mean learning gap than tabular-const at every γ, and a higher median Spearman correlation and "
        "top-10 recall over all portfolios; against baseline-v1 no direction is predicted. Reference model, never "
        "the discovery result",
        alpha_rule=TUNED_ALPHA_QUADRATIC, agent="linear", feature_map="quadratic",
        greedy_refresh_every=FA_GREEDY_REFRESH,
    ),
    "fa-spline": RunVariant(
        "fa-spline", "FAspline", "2026-11-05",
        "a lower median Spearman correlation and top-10 recall over all portfolios than fa-quadratic at every γ, "
        "since most spline directions learn far more slowly than 13M episodes allow; no direction for the learning "
        "gap against fa-quadratic (that argument points to larger, the γ = 5 tuning seeds to smaller), nor against "
        "tabular-const or baseline-v1",
        alpha_rule=TUNED_ALPHA_SPLINE, agent="linear", feature_map="spline", l2=0.0,
        greedy_refresh_every=FA_GREEDY_REFRESH,
    ),
}


def experiments(run: RunVariant | None = None) -> list[Experiment]:
    """Every experiment the report needs, in the order to run them. With a
    post-hoc variant: the same three γ on the variant's grid and settings."""
    if run is not None:
        grid = f"grid{round(run.increment * 100)}"
        purpose = f"post-hoc run {run.run_id}: {run.expected}"
        agent = {k: getattr(run, k) for k in AGENT_SETTINGS}
        return [
            Experiment(f"{'baseline' if g == GAMMA_BASELINE else 'robust'}_g{int(g)}_{grid}_{run.suffix}",
                       g, run.increment, purpose, run.episode_length, run.epsilon_at_halfway, run.n_episodes,
                       **agent)
            for g in (GAMMA_BASELINE, *GAMMA_ROBUSTNESS)
        ]
    return [
        Experiment("baseline_g5_grid20", GAMMA_BASELINE, INCREMENT_BASELINE,
                   "headline results (Part V weights, performance, convergence)"),
        Experiment("resolution_g5_grid25", GAMMA_BASELINE, INCREMENT_COMPARISON,
                   "second action-space resolution (Part V), also the action-space robustness check (Part VI)"),
        *(
            Experiment(f"robust_g{int(g)}_grid20", g, INCREMENT_BASELINE, "risk-aversion robustness check (Part VI)")
            for g in GAMMA_ROBUSTNESS
        ),
    ]


def train_config(experiment: Experiment, n_episodes: int | None = None, n_runs: int = N_RUNS,
                 log_every: int = LOG_EVERY) -> TrainConfig:
    """n_episodes defaults to the experiment's own training budget."""
    n_episodes = experiment.n_episodes if n_episodes is None else n_episodes
    return TrainConfig(
        gamma=experiment.gamma,
        episode_length=experiment.episode_length,
        n_episodes=n_episodes,
        master_seed=MASTER_SEED,
        n_runs=n_runs,
        beta=beta_for(n_episodes, experiment.epsilon_at_halfway),
        increment=experiment.increment,
        alpha_rule=experiment.alpha_rule,
        epsilon0=EPSILON0,
        q_init=Q_INIT,
        discount=DISCOUNT,
        log_every=log_every,
        agent=experiment.agent,
        feature_map=experiment.feature_map,
        l2=experiment.l2,
        greedy_refresh_every=experiment.greedy_refresh_every,
    )


def eval_config(n_episodes: int = N_EPISODES, n_eval_episodes: int = N_EVAL_EPISODES) -> EvalConfig:
    return EvalConfig(
        n_eval_episodes=n_eval_episodes,
        converge_k=convergence_k(n_episodes),
        dq_tol=DQ_TOL,
        near_optimal_bps=NEAR_OPTIMAL_BPS,
    )
