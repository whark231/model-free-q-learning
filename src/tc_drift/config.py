"""Fixed settings for the transaction-cost-with-drift runs (Albert).

In plain terms: one place for the numbers every TC-drift run shares (how
long the evaluation paths are, how many starts, how often training restarts
and checkpoints, which probe states the convergence rule watches) and for the
random streams each run uses. A separate module from src/config.py, which is
fingerprinted in every baseline evaluation (TC_DRIFT_DESIGN.md C5).

Values marked PENDING are not decided yet; config.pending() lists them, and no
reported run may start while it is non-empty.

Seeds (TC_DRIFT_DESIGN.md §4):
- reported run i: SeedPlan(spec.MASTER_SEED, N_SEEDS).run_streams(i) gives
  (returns_rng, agent_rng); training start states come from
  agent_rng.spawn(1)[0], as the old TC code did;
- evaluation paths and starts: SeedPlan(...).evaluation_rng();
- tuning: the same layout under spec.TUNING_MASTER_SEED;
- the off-grid probe states: np.random.default_rng(PROBE_SEED), drawn once.
"""

import numpy as np

from src.environment import SeedPlan
from src.tc_drift import spec

LEVELS = ("zero", "aum_100m", "aum_1b", "aum_10b")  # calibration.cost_settings() keys (D9)
HEADLINE_LEVEL = "aum_1b"

# Evaluation (D10, approved 2026-11-13). Revisit M_EVAL once the oracle's lead
# over always-rebalance is measured (§13 D10 rule).
H_EVAL = 90
M_EVAL = 1000

# Oracle paths. Fixed at 2,000 by check 3, restated on the policy (William,
# 2026-11-17; TC_DRIFT_DESIGN.md §5).
M_ORACLE = 2000

# Training. Restarting every H_EVAL blocks matches the evaluation's episode
# length, so training covers the same mix of start-up and steady-state blocks
# the evaluation scores (Albert's proposal, 2026-11-17).
# Training start states (D4, as for the evaluation): uniform on the 2% grid.
N_SEEDS = 10
RESTART_EVERY = H_EVAL
N_BLOCKS = None          # PENDING tuning (William's choice among TC_DRIFT_DESIGN.md §9 options)
CHECKPOINT_EVERY = None  # PENDING: N_BLOCKS // 100, set with N_BLOCKS
HYPERPARAMETERS = None   # PENDING tuning: the DriftAgent keywords

# Probe states for the convergence rule (D12): every 2% target plus a fixed
# sample of off-grid 1% lattice states.
N_OFFGRID_PROBES = 200
PROBE_SEED = (spec.TUNING_MASTER_SEED, 1)


def train_start_states() -> np.ndarray:
    """Where training restarts: every 2% target, drawn uniformly (D4)."""
    return spec.target_grid()


def probe_states() -> np.ndarray:
    """(1,326 + N_OFFGRID_PROBES, 3): the 2% targets, then off-grid lattice
    points drawn without replacement from PROBE_SEED. The same every call."""
    points = spec.lattice()
    off_grid = np.setdiff1d(np.arange(len(points)), spec.target_lattice_index())
    pick = np.random.default_rng(PROBE_SEED).choice(off_grid, size=N_OFFGRID_PROBES, replace=False)
    return np.vstack([spec.target_grid(), points[np.sort(pick)]])


def run_rngs(run: int, master_seed: int = spec.MASTER_SEED, n_runs: int = N_SEEDS):
    """(returns_rng, agent_rng, start_rng) for training run `run`."""
    returns_rng, agent_rng = SeedPlan(master_seed, n_runs).run_streams(run)
    start_rng = agent_rng.spawn(1)[0]
    return returns_rng, agent_rng, start_rng


def evaluation_rng(master_seed: int = spec.MASTER_SEED, n_runs: int = N_SEEDS) -> np.random.Generator:
    return SeedPlan(master_seed, n_runs).evaluation_rng()


def pending() -> list[str]:
    """Names of the settings still undecided."""
    return [name for name in ("N_BLOCKS", "CHECKPOINT_EVERY", "HYPERPARAMETERS") if globals()[name] is None]
