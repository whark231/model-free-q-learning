"""Training loop for the transaction-cost-with-drift agent (Albert).

In plain terms: start from a holding, ask the agent for an action, play out
one block (env.draw_returns, env.block_outcome), charge the cost
(spec.trade_cost), hand the agent the net reward and the drifted holdings,
repeat. Every `restart_every` blocks, restart from a holding drawn uniformly
from `start_states`, so the agent sees many starting points. Every
`checkpoint_every` blocks, record the agent's greedy action on each of the
fixed `probe_states`; the pre-registered convergence rule reads these
(docs/methods/TC_DRIFT_DESIGN.md §6, D12). The loop holds the market and the
cost settings; the agent never does.

Assumptions and conventions:
- A restart is not a terminal state. The agent is always updated with the
  true drifted end-of-block holdings as next_state, even on the block after
  which the loop restarts; the restart only changes where the next block
  starts (William's interface note, NEXT_STEPS_FOR_ALBERT.md 2026-11-15).
- The first block starts from a draw from start_states, and block k restarts
  when k is a positive multiple of restart_every.
- Returns are drawn in chunks of up to _CHUNK blocks from one ReturnSimulator.
  ReturnSimulator.sample_episodes gives the same numbers as consecutive
  single draws, so the chunk size does not change the run (tested).
- A trade whose target equals the current holdings (zero size) counts as a
  hold in the log, as in the evaluation (TC_DRIFT_DESIGN.md §4, "Ties").
- Turnover is one-way, 0.5 * ||u||_1, the same measure as the design's
  "median one-block drift 2.26%" (§12).
"""

import numpy as np

from src.environment import ReturnSimulator
from src.tc_drift import env
from src.tc_drift.spec import (BLOCK_LENGTH, HOLD, Market, N_ACTIONS, TCAgent, TCSettings, check_holdings,
                               net_reward, post_trade_weights, trade_cost)

_CHUNK = 1024           # blocks of returns drawn per ReturnSimulator call (about 1.5 MB)
ZERO_TRADE_TOL = 1e-12  # ||u||_1 at or below this counts as a hold

# Read-only diagnostics logged at each checkpoint when the agent has them
# (DriftAgent does; the stub agents in the tests do not). Fixed names, never
# a computed attribute.
_DIAGNOSTICS = ("cost_linear", "cost_quadratic", "epsilon", "n_updates")


def train(agent: TCAgent, market: Market, settings: TCSettings, *, n_blocks: int,
          returns_rng: np.random.Generator, start_rng: np.random.Generator, start_states: np.ndarray,
          restart_every: int, checkpoint_every: int, probe_states: np.ndarray, drift: bool = True) -> dict:
    """Train in place and return a log for the run manifest: the greedy
    actions on probe_states at each checkpoint, hold fraction and cost paid
    per checkpoint window, and the blocks used (blocks x B = simulated days).

    Log keys (arrays have one row per checkpoint):
      checkpoint_block   blocks completed at each checkpoint (the last one is n_blocks)
      probe_greedy       (n_checkpoints, n_probes) greedy action per probe state
      hold_fraction      share of blocks in the window that held (zero-size trades included)
      explicit_hold_fraction share of blocks in the window where the action was HOLD itself
      mean_cost_bps      mean cost paid per block in the window, bp of wealth
      mean_turnover      mean one-way turnover per block in the window
      mean_net_reward    mean net reward per block in the window, daily units
      diagnostics        {name: (n_checkpoints,)} for the names in _DIAGNOSTICS the agent has,
                         plus theta_norm when it exposes theta
      n_blocks, block_length, simulated_days, n_restarts, drift
    """
    n_blocks = _positive_int(n_blocks, "n_blocks")
    restart_every = _positive_int(restart_every, "restart_every")
    checkpoint_every = _positive_int(checkpoint_every, "checkpoint_every")
    for name, rng in (("returns_rng", returns_rng), ("start_rng", start_rng)):
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"{name} must be a numpy Generator (from src.environment.SeedPlan)")
    if returns_rng is start_rng:
        raise ValueError("returns_rng and start_rng must be different generators")
    if not isinstance(drift, (bool, np.bool_)):
        raise TypeError(f"drift must be True or False, got {type(drift).__name__}")
    starts = check_holdings(start_states)
    probes = check_holdings(probe_states)

    simulator = ReturnSimulator(market.mu, market.sigma, returns_rng)
    checkpoints = list(range(checkpoint_every, n_blocks + 1, checkpoint_every))
    if not checkpoints or checkpoints[-1] != n_blocks:
        checkpoints.append(n_blocks)
    n_ckpt = len(checkpoints)

    log = {
        "checkpoint_block": np.array(checkpoints, dtype=np.int64),
        "probe_greedy": np.empty((n_ckpt, len(probes)), dtype=np.int64),
        "hold_fraction": np.empty(n_ckpt),
        "explicit_hold_fraction": np.empty(n_ckpt),
        "mean_cost_bps": np.empty(n_ckpt),
        "mean_turnover": np.empty(n_ckpt),
        "mean_net_reward": np.empty(n_ckpt),
    }
    diag_names = [name for name in _DIAGNOSTICS if hasattr(agent, name)]
    diagnostics = {name: np.empty(n_ckpt) for name in diag_names}
    if hasattr(agent, "theta"):
        diagnostics["theta_norm"] = np.empty(n_ckpt)

    state = starts[start_rng.integers(len(starts))]
    n_restarts = 0
    window = np.zeros(5)   # holds, explicit HOLDs, cost, turnover, net reward
    window_start = 0
    ckpt = 0
    chunk = np.empty((0, BLOCK_LENGTH, 3))
    pos = 0

    for k in range(n_blocks):
        if k > 0 and k % restart_every == 0:
            state = starts[start_rng.integers(len(starts))]
            n_restarts += 1
        if pos == len(chunk):
            chunk = simulator.sample_episodes(min(_CHUNK, n_blocks - k), BLOCK_LENGTH)
            pos = 0

        action = agent.select_action(state.copy())
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < N_ACTIONS:
            raise ValueError(f"the agent returned an invalid action {action!r}")
        action = int(action)
        cost = trade_cost(state, action, settings)
        x = post_trade_weights(state, action)
        reward, w_end = env.block_outcome(x, chunk[pos:pos + 1], settings.gamma, drift)
        pos += 1
        reward_net = float(net_reward(reward[0], cost))
        next_state = w_end[0]
        agent.update(state.copy(), action, reward_net, next_state.copy())

        size = float(np.abs(x - state).sum())
        window += (size <= ZERO_TRADE_TOL, action == HOLD, cost, 0.5 * size, reward_net)
        state = next_state

        if k + 1 == checkpoints[ckpt]:
            n_window = k + 1 - window_start
            log["hold_fraction"][ckpt] = window[0] / n_window
            log["explicit_hold_fraction"][ckpt] = window[1] / n_window
            log["mean_cost_bps"][ckpt] = window[2] / n_window * 1e4
            log["mean_turnover"][ckpt] = window[3] / n_window
            log["mean_net_reward"][ckpt] = window[4] / n_window
            log["probe_greedy"][ckpt] = [agent.greedy_action(p.copy()) for p in probes]
            for name in diag_names:
                diagnostics[name][ckpt] = float(getattr(agent, name))
            if "theta_norm" in diagnostics:
                diagnostics["theta_norm"][ckpt] = float(np.linalg.norm(agent.theta))
            window[:] = 0.0
            window_start = k + 1
            ckpt += 1

    log.update(diagnostics=diagnostics, n_blocks=n_blocks, block_length=BLOCK_LENGTH,
               simulated_days=n_blocks * BLOCK_LENGTH, n_restarts=n_restarts, drift=bool(drift))
    return log


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)
