"""Training loop for the transaction-cost-with-drift agent (Albert).
Phase 1: signature only.

In plain terms: start from a holding, ask the agent for an action, play out
one block (env.draw_returns, env.block_outcome), charge the cost
(spec.trade_cost), hand the agent the net reward and the drifted holdings,
repeat. Every `restart_every` blocks, restart from a holding drawn uniformly
from `start_states`, so the agent sees many starting points. Every
`checkpoint_every` blocks, record the agent's greedy action on each of the
fixed `probe_states`; the pre-registered convergence rule reads these
(docs/methods/TC_DRIFT_DESIGN.md §6). The loop holds the market and the cost
settings; the agent never does.
"""

import numpy as np

from src.tc_drift.spec import Market, TCAgent, TCSettings


def train(agent: TCAgent, market: Market, settings: TCSettings, *, n_blocks: int,
          returns_rng: np.random.Generator, start_rng: np.random.Generator, start_states: np.ndarray,
          restart_every: int, checkpoint_every: int, probe_states: np.ndarray, drift: bool = True) -> dict:
    """Train in place and return a log for the run manifest: the greedy
    actions on probe_states at each checkpoint, hold fraction and cost paid
    per checkpoint window, and the blocks used (blocks x B = simulated days)."""
    raise NotImplementedError("Phase 2 (Albert)")
