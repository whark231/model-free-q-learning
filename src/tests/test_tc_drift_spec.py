"""Contract tests for the transaction-cost-with-drift interface (Phase 1).

In plain terms: these pin down what docs/methods/TC_DRIFT_DESIGN.md freezes,
so William's and Albert's code can be built separately and still fit: the
action and lattice numbering, the cost formula and its calibration, the
reward and annualization conventions, every stub's signature, and the rule
that the agent never receives mu, Sigma, kappa, lambda, gamma or oracle output.
Numbers are re-derived independently (brute force or by hand), not by
re-running the implementation.
"""

import ast
import inspect
import os
import subprocess
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

from src.data.universe import TICKERS
from src.tc_drift import agent, env, evaluate, features, oracle, spec, train

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Constants, actions and the lattice
# ---------------------------------------------------------------------------

def test_counts_follow_the_stars_and_bars_formula():
    assert spec.N_TARGETS == comb(50 + 2, 2) == len(spec.target_grid()) == 1326
    assert spec.N_LATTICE == comb(100 + 2, 2) == len(spec.lattice()) == 5151
    assert len(spec.lattice(0.005)) == comb(200 + 2, 2) == 20301
    assert spec.HOLD == 1326 and spec.N_ACTIONS == 1327


def test_assets_are_the_assignments_original_three_in_tickers_order():
    assert spec.ASSETS == ("SPY", "IEF", "GLD")
    assert spec.ASSET_INDEX == (0, 3, 6) == tuple(TICKERS.index(t) for t in spec.ASSETS)


def test_every_target_is_a_lattice_point():
    np.testing.assert_allclose(spec.lattice()[spec.target_lattice_index()], spec.target_grid(), atol=1e-12)


@pytest.mark.parametrize("grid", [spec.target_grid, spec.lattice])
def test_shared_grids_are_read_only_and_stay_that_way(grid):
    g = grid()
    with pytest.raises(ValueError):
        g[0, 0] = 0.5
    with pytest.raises(ValueError):
        g.flags.writeable = True


def test_holdings_must_sum_to_one_within_1e9_and_actions_must_be_integers():
    with pytest.raises(ValueError):
        spec.check_holdings(np.array([0.5, 0.3, 0.2 + 2e-9]))
    spec.check_holdings(np.array([0.5, 0.3, 0.2 + 5e-10]))
    s = np.array([0.5, 0.3, 0.2])
    for bad in (True, 1.0, -1, spec.N_ACTIONS):
        with pytest.raises(ValueError):
            spec.post_trade_weights(s, bad)
    with pytest.raises(ValueError):
        spec.trade_cost(np.array([2.0, 0.0, 0.0]), 0, spec.TCSettings(kappa=1e-4, lam=0.0))


def test_lattice_points_round_to_themselves():
    pts = spec.lattice()
    np.testing.assert_array_equal(spec.nearest_lattice_index(pts), np.arange(len(pts)))


@pytest.mark.parametrize("increment", [0.01, 0.02, 0.005])
def test_rounding_finds_the_euclidean_nearest_lattice_point(increment):
    """Brute force over every lattice point, for random holdings anywhere on the simplex."""
    rng = np.random.default_rng(20261112)
    w = rng.dirichlet(np.ones(3), size=2000)
    pts = spec.lattice(increment)
    got = spec.nearest_lattice_index(w, increment)
    dist = ((w[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(dist[np.arange(len(w)), got], dist.min(axis=1), atol=1e-15)


def test_rounding_accepts_one_holding_and_rejects_bad_ones():
    assert isinstance(spec.nearest_lattice_index(np.array([0.333, 0.333, 0.334])), int)
    for bad in ([0.5, 0.5, 0.5], [1.2, -0.2, 0.0], [np.nan, 0.5, 0.5], [0.5, 0.5]):
        with pytest.raises(ValueError):
            spec.nearest_lattice_index(np.array(bad))


def test_hold_keeps_the_drifted_holdings_and_a_trade_goes_to_its_target():
    s = np.array([0.471, 0.129, 0.4])
    np.testing.assert_array_equal(spec.post_trade_weights(s, spec.HOLD), s)
    np.testing.assert_array_equal(spec.post_trade_weights(s, 7), spec.target_grid()[7])
    states = np.repeat(s[None], 2, 0)
    out = spec.post_trade_weights(states, np.array([spec.HOLD, 7]))
    np.testing.assert_array_equal(out, [s, spec.target_grid()[7]])
    with pytest.raises(ValueError):
        spec.post_trade_weights(s, spec.N_ACTIONS)


# ---------------------------------------------------------------------------
# Costs and their calibration
# ---------------------------------------------------------------------------

def target_index(weights):
    return int(np.flatnonzero(np.all(np.isclose(spec.target_grid(), weights), axis=1))[0])


def test_hold_is_free_and_trading_to_where_you_already_are_is_free():
    settings = spec.TCSettings(kappa=5e-4, lam=2e-3)
    s = np.array([0.471, 0.129, 0.4])
    assert spec.trade_cost(s, spec.HOLD, settings) == 0.0
    start = spec.target_grid()[100]
    assert spec.trade_cost(start, 100, settings) == 0.0


def test_the_l1_norm_counts_both_legs_of_a_transfer():
    """Moving 20 points from SPY to IEF: |+0.2| + |-0.2| = 0.4, and 0.2^2 + 0.2^2 = 0.08."""
    s = np.array([0.6, 0.2, 0.2])
    a = target_index([0.4, 0.4, 0.2])
    assert spec.trade_cost(s, a, spec.TCSettings(kappa=1.0, lam=0.0)) == pytest.approx(0.4)
    assert spec.trade_cost(s, a, spec.TCSettings(kappa=0.0, lam=2.0)) == pytest.approx(0.08)


@pytest.mark.parametrize("c20, c40", [(2.0, 6.0), (2.0, 4.0), (2.0, 8.0), (10.0, 25.0)])
def test_coefficients_reproduce_both_reference_trades(c20, c40):
    kappa, lam = spec.coefficients_from_reference_trades(c20, c40)
    settings = spec.TCSettings(kappa=kappa, lam=lam)
    s = np.array([0.6, 0.2, 0.2])
    cost20 = spec.trade_cost(s, target_index([0.4, 0.4, 0.2]), settings)
    cost40 = spec.trade_cost(s, target_index([0.2, 0.6, 0.2]), settings)
    assert cost20 * 1e4 == pytest.approx(c20) and cost40 * 1e4 == pytest.approx(c40)


def test_pure_linear_and_pure_quadratic_are_the_ends_of_the_feasible_range():
    assert spec.coefficients_from_reference_trades(2.0, 4.0)[1] == pytest.approx(0.0, abs=1e-15)  # kappa only
    assert spec.coefficients_from_reference_trades(2.0, 8.0)[0] == pytest.approx(0.0, abs=1e-15)  # lambda only
    for bad in ((2.0, 3.9), (2.0, 8.1), (-1.0, -2.0)):
        with pytest.raises(ValueError):
            spec.coefficients_from_reference_trades(*bad)


def test_vectorized_cost_matches_one_at_a_time():
    rng = np.random.default_rng(1)
    settings = spec.TCSettings(kappa=3e-4, lam=1e-3)
    states = rng.dirichlet(np.ones(3), size=50)
    actions = rng.integers(0, spec.N_ACTIONS, size=50)
    one = [spec.trade_cost(s, int(a), settings) for s, a in zip(states, actions)]
    np.testing.assert_allclose(spec.trade_cost(states, actions, settings), one, rtol=1e-14)


@pytest.mark.parametrize("kwargs", [{"kappa": -1e-4, "lam": 0.0}, {"kappa": 0.0, "lam": np.inf},
                                    {"kappa": 0.0, "lam": 0.0, "gamma": 0.0}])
def test_settings_reject_impossible_values(kwargs):
    with pytest.raises(ValueError):
        spec.TCSettings(**kwargs)


# ---------------------------------------------------------------------------
# Reward and reporting conventions
# ---------------------------------------------------------------------------

def test_cost_is_spread_over_the_blocks_days():
    assert spec.net_reward(1e-3, 63e-4) == pytest.approx(1e-3 - 1e-4)


def test_annualizing_a_constant_daily_reward_returns_it_in_bps_per_year():
    """A reward of x per day forever is worth x / (1 - rho); over H blocks, x * sum rho^t."""
    x, rho = 2e-4, spec.DISCOUNT
    assert spec.annualized_bps(x / (1 - rho)) == pytest.approx(x * 252e4)
    assert spec.annualized_bps(x * sum(rho**t for t in range(100)), horizon=100) == pytest.approx(x * 252e4)


@pytest.mark.parametrize("kwargs", [{"discount": 1.0}, {"discount": -0.1}, {"horizon": 0}, {"horizon": 2.5}])
def test_annualizing_rejects_impossible_inputs(kwargs):
    with pytest.raises(ValueError):
        spec.annualized_bps(1.0, **kwargs)


# ---------------------------------------------------------------------------
# Frozen signatures (change only by a PR both approve)
# ---------------------------------------------------------------------------

FROZEN = {
    env.load_market: ["processed_dir"],
    env.draw_returns: ["market", "rng", "m", "block_length"],
    env.block_outcome: ["w0", "excess_returns", "gamma", "drift"],
    env.simulate_block: ["w0", "rng", "B", "drift", "market", "gamma"],
    oracle.solve: ["market", "settings", "excess_paths", "block_outcome", "discount", "drift", "lattice_increment"],
    oracle.OracleSolution.policy: ["self", "state"],
    oracle.OracleSolution.value_of: ["self", "policy_table", "discount"],
    features.spline_features: ["W", "n_basis", "degree"],
    agent.DriftAgent.__init__: ["self", "rng", "hyperparameters"],
    agent.DriftAgent.select_action: ["self", "state"],
    agent.DriftAgent.update: ["self", "state", "action", "reward_net", "next_state"],
    agent.DriftAgent.greedy_action: ["self", "state"],
    train.train: ["agent", "market", "settings", "n_blocks", "returns_rng", "start_rng", "start_states",
                  "restart_every", "checkpoint_every", "probe_states", "drift"],
    evaluate.evaluate_policy: ["policy", "starts", "excess_paths", "settings", "discount", "drift"],
    evaluate.agent_policy: ["agent"],
    evaluate.oracle_policy: ["solution"],
    evaluate.always_rebalance_policy: ["best_target"],
    evaluate.never_trade_policy: ["first_target"],
}


@pytest.mark.parametrize("func", list(FROZEN), ids=lambda f: f.__qualname__)
def test_signatures_are_frozen(func):
    assert list(inspect.signature(func).parameters) == FROZEN[func]


def test_oracle_solution_carries_what_the_checks_need():
    fields = set(oracle.OracleSolution.__dataclass_fields__)
    assert {"V", "G", "policy_table", "r_hat", "cv_correction", "next_index"} <= fields


def test_the_agent_class_implements_the_protocol_method_names():
    for name in ("select_action", "update", "greedy_action"):
        proto = list(inspect.signature(getattr(spec.TCAgent, name)).parameters)
        assert list(inspect.signature(getattr(agent.DriftAgent, name)).parameters) == proto


def test_reported_seed_matches_the_project_seed_and_tuning_differs():
    from src.config import MASTER_SEED
    assert spec.MASTER_SEED == MASTER_SEED
    assert spec.TUNING_MASTER_SEED not in (MASTER_SEED, 20261103)


# ---------------------------------------------------------------------------
# Leakage: the agent gets its own rng, its hyperparameters and the action
# helpers in spec.AGENT_SAFE_NAMES, nothing else (assignment Part I: mu and
# Sigma "are reserved for benchmark construction and evaluation"). An
# allowlist, so new routes fail by default. These tests catch accidental
# leaks; they cannot stop a deliberate one. src/tests/test_environment.py
# also scans these modules for data paths and benchmark imports.
# ---------------------------------------------------------------------------

TC_AGENT_MODULES = ["src.tc_drift.agent", "src.tc_drift.features"]
ALLOWED_IMPORTS = {"numpy", "math", "dataclasses", "typing", "functools", "__future__",
                   "src.features", "src.tc_drift.features"}
SPEC = "src.tc_drift.spec"
BANNED_CALLS = {"__import__", "open", "eval", "exec", "globals", "vars", "compile"}
BANNED_ATTRS = {"load", "loadtxt", "genfromtxt", "fromfile", "memmap", "read_csv", "read_text", "read_bytes",
                "__dict__", "import_module"}
FORBIDDEN_PARAMS = {"mu", "sigma", "market", "settings", "kappa", "lam", "lambda_", "gamma", "oracle", "solution",
                    "G_init", "V", "policy_table"}
ALLOWED_SRC_MODULES = {"src", "src.tc_drift", "src.tc_drift.spec", "src.tc_drift.features", "src.tc_drift.agent",
                       "src.features", "src.action_space", "src.action_space.grid", "src.data", "src.data.universe"}


def tc_leakage_violations(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # relative import from inside src/tc_drift/
                module = f"src.tc_drift.{module}" if module else "src.tc_drift"
            if any(a.name == "*" for a in node.names):
                found.append(f"{module}.*")
            elif module == SPEC:
                found += [a.name for a in node.names if a.name not in spec.AGENT_SAFE_NAMES]
            elif module not in ALLOWED_IMPORTS:
                found.append(module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                found.append(node.func.id)
            if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) > 1 \
                    and not isinstance(node.args[1], ast.Constant):
                found.append("getattr with a computed name")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            found.append(node.attr)
    return found


@pytest.mark.parametrize("module", TC_AGENT_MODULES)
def test_tc_agent_source_uses_only_allowed_inputs(module):
    path = REPO / Path(*module.split(".")).with_suffix(".py")
    assert tc_leakage_violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("module", TC_AGENT_MODULES)
def test_tc_agent_import_graph_stays_inside_the_allowlist(module):
    code = f"import {module}\nimport sys\nprint('\\n'.join(sys.modules))"
    env_vars = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    loaded = set(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env_vars,
                                check=True, cwd=REPO).stdout.split())
    assert module in loaded
    assert sorted(m for m in loaded if (m == "src" or m.startswith("src.")) and m not in ALLOWED_SRC_MODULES) == []


def test_agent_safe_names_exist_and_exclude_costs_market_and_gamma():
    assert all(hasattr(spec, n) for n in spec.AGENT_SAFE_NAMES)
    assert not spec.AGENT_SAFE_NAMES & {"trade_cost", "coefficients_from_reference_trades", "TCSettings", "Market",
                                        "GAMMA_HEADLINE", "REFERENCE_TRADES", "net_reward", "annualized_bps"}


def test_no_agent_method_or_hyperparameter_takes_market_cost_or_oracle_inputs():
    for name in ("__init__", "select_action", "update", "greedy_action"):
        params = set(inspect.signature(getattr(agent.DriftAgent, name)).parameters)
        assert not params & FORBIDDEN_PARAMS, name
    assert not agent.DriftAgent.HYPERPARAMETERS & FORBIDDEN_PARAMS


def test_the_agent_rejects_a_keyword_outside_its_hyperparameters():
    """Phase 2 contract; skipped while the agent is a stub."""
    try:
        with pytest.raises(TypeError):
            agent.DriftAgent(np.random.default_rng(0), mu=np.zeros(3))
    except NotImplementedError:
        pytest.skip("DriftAgent is a Phase 1 stub")


@pytest.mark.parametrize("snippet", [
    "from src.tc_drift.spec import trade_cost",
    "from src.tc_drift.spec import GAMMA_HEADLINE",
    "from src.tc_drift.spec import *",
    "import src.tc_drift.spec as spec",
    "from src.tc_drift import oracle",
    "from . import env",
    "from .oracle import OracleSolution",
    "from src.tc_drift.config import SETTINGS",
    "import importlib\nm = importlib.import_module('src.tc_drift.oracle')",
    "m = __import__('src.tc_drift.oracle')",
    "f = getattr(spec, 'trade_' + 'cost')",
    "G = np.load('results/tc-drift/oracle.npz')",
    "x = json.loads(open('results/oracle.json').read())",
    "d = spec.__dict__['trade_cost']",
    "from src.environment import ReturnSimulator",
    "import pandas as pd",
])
def test_tc_leakage_check_catches_each_route(snippet):
    """The agent is a stub today, so the scan passes trivially; prove it would fail a real leak."""
    assert tc_leakage_violations(snippet)


@pytest.mark.parametrize("snippet", [
    "import numpy as np\nfrom src.tc_drift.spec import HOLD, N_TARGETS, target_grid, nearest_lattice_index",
    "from src.tc_drift.features import spline_features",
    "from src.features import bspline_basis",
    "x = getattr(self, 'theta')",
])
def test_tc_leakage_check_allows_what_the_agent_needs(snippet):
    assert tc_leakage_violations(snippet) == []
