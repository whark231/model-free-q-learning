"""Fixed settings for the transaction-cost-with-drift runs (Albert).

In plain terms: one place for the numbers every TC-drift run shares (how
long the evaluation paths are, how many starts, how often training restarts
and checkpoints, which probe states the convergence rule watches) and for the
random streams each run uses. A separate module from src/config.py, which is
fingerprinted in every baseline evaluation (TC_DRIFT_DESIGN.md C5).

config.pending() lists any value still undecided (None); no reported run may
start while it is non-empty. All were set on 2026-11-20.

Seeds (TC_DRIFT_DESIGN.md §4):
- reported run i: SeedPlan(spec.MASTER_SEED, N_SEEDS).run_streams(i) gives
  (returns_rng, agent_rng); training start states come from
  agent_rng.spawn(1)[0], as the old TC code did;
- evaluation paths and starts: SeedPlan(...).evaluation_rng();
- tuning: the same layout under spec.TUNING_MASTER_SEED;
- the off-grid probe states: np.random.default_rng(PROBE_SEED), drawn once.
"""

import math

import numpy as np

from src.environment import SeedPlan
from src.tc_drift import spec

LEVELS = ("zero", "aum_100m", "aum_1b", "aum_10b")  # calibration.cost_settings() keys (D9)
HEADLINE_LEVEL = "aum_1b"

# Evaluation (D10, approved 2026-11-13: H = 90, m = 1,000). m raised to 4,000
# by Albert on 2026-11-20, before any final agent run, as William proposed
# (AUDIT_2026-11-06.md TC-7): the D10 rule needs the agent's gap SE, so fixing m
# now keeps it from being chosen by results. 4,000 halves every SE; the
# paths take 544 MB. The D10 pilot, check 5 and the tuning pilot used 1,000.
H_EVAL = 90
M_EVAL = 4000
M_EVAL_PILOTS = 1000

# Oracle paths. Fixed at 2,000 by check 3, restated on the policy (William,
# 2026-11-17; TC_DRIFT_DESIGN.md §5).
M_ORACLE = 2000

# Training. Restarting every H_EVAL blocks matches the evaluation's episode
# length, so training covers the same mix of start-up and steady-state blocks
# the evaluation scores (Albert's proposal, 2026-11-17).
# Training start states (D4, as for the evaluation): uniform on the 2% grid.
N_SEEDS = 10
RESTART_EVERY = H_EVAL
# Decided by William, 2026-11-19 and 2026-11-20 (TC_DRIFT_DESIGN.md §9; the
# tuning pilot chose alpha_value = 3e-4, the other values were confirmed for the
# final runs). beta puts epsilon at 1% halfway through N_BLOCKS; the cost block
# switches to recursive least squares when epsilon first reaches 0.5.
N_BLOCKS = 4_000_000
# 40,050 = 445 restarts, the tuning pilot's segment (tune.SEGMENT): with train()'s
# final checkpoint at N_BLOCKS this gives the layout design §6 registered (11
# tail-average snapshots, a 12-checkpoint D12 window from block 3,564,450).
CHECKPOINT_EVERY = 40_050
_BETA = math.log(100) / (N_BLOCKS / 2)
HYPERPARAMETERS = {
    "alpha_value": 3e-4,
    "alpha_cost": 3e-4,
    "beta": _BETA,
    "target_refresh_every": 1000,
    "l2": 0.0,
    "cost_rls_after": round(math.log(2) / _BETA),
    "cost_rls_prior": 1e-3,
}

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
