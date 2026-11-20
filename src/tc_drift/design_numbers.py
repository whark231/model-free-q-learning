"""Reproduce the numbers quoted in docs/methods/TC_DRIFT_DESIGN.md §12.

In plain terms: before building anything, size the problem. How noisy is one
63-day block's reward, how far do holdings drift in a block, how much does the
risk-free rate matter for drift, and how much does the control variate help?
This uses mu and Sigma to size the environment only; nothing here is handed to
an agent, and it is not part of the agent's import graph.

    python -m src.tc_drift.design_numbers

Seed DESIGN_SEED is used for these sizing draws only (not the reported seed
20261016 and not the tuning seed 20261112).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.action_space.grid import generate_weight_grid
from src.benchmarks.mean_variance import constrained_optimal_weights
from src.data.universe import TICKERS
from src.environment import PROCESSED_DIR, load_calibration
from src.features import bspline_basis
from src.rewards import episodic_mean_variance_reward
from src.tc_drift.spec import ASSET_INDEX, BLOCK_LENGTH, GAMMA_HEADLINE

DESIGN_SEED = 20261113
BPS_YR = 252 * 1e4


def buy_and_hold_block(w0, excess, gamma, rf_daily):
    """Reward and end weights when shares are fixed for the block. Each day's
    portfolio excess return uses that day's weights; weights then move with
    gross returns 1 + rf + excess. (Reference loop for sizing, not env.py.)"""
    w = np.repeat(w0[None, :], len(excess), 0)
    port = np.empty(excess.shape[:2])
    for t in range(excess.shape[1]):
        port[:, t] = np.einsum("mi,mi->m", w, excess[:, t])
        grown = w * (1.0 + rf_daily + excess[:, t])
        w = grown / grown.sum(axis=1, keepdims=True)
    return port.mean(axis=1) - gamma / 2 * port.var(axis=1, ddof=1), w


def main(n_paths: int = 20_000) -> dict:
    mu9, sigma9 = load_calibration()
    out = {"seed": DESIGN_SEED, "n_paths": n_paths, "block_length": BLOCK_LENGTH, "gamma": GAMMA_HEADLINE}
    out["nine_asset_optimum_support"] = {
        str(g): {TICKERS[i]: round(float(w), 4) for i, w in enumerate(constrained_optimal_weights(mu9, sigma9, g))
                 if w > 1e-6} for g in (3.0, 5.0, 8.0)}
    idx = list(ASSET_INDEX)
    mu, sigma = mu9[idx], sigma9[np.ix_(idx, idx)]
    w0 = constrained_optimal_weights(mu, sigma, GAMMA_HEADLINE)
    out["three_asset_optimum_gamma5"] = [round(float(x), 4) for x in w0]

    rf = pd.read_csv(Path(PROCESSED_DIR) / "rf_daily.csv")["RF"]
    out["rf_daily_mean"] = float(rf.mean())

    rng = np.random.default_rng(DESIGN_SEED)
    excess = rng.multivariate_normal(mu, sigma, size=(n_paths, BLOCK_LENGTH))
    r_const = np.array([episodic_mean_variance_reward(w0, e, GAMMA_HEADLINE) for e in excess])
    r_bh, w_end = buy_and_hold_block(w0, excess, GAMMA_HEADLINE, float(rf.mean()))
    _, w_end_no_rf = buy_and_hold_block(w0, excess, GAMMA_HEADLINE, 0.0)
    diff = r_bh - r_const
    out["block_reward_sd_bps_yr"] = {"constant_weights": float(r_const.std() * BPS_YR),
                                     "buy_and_hold": float(r_bh.std() * BPS_YR),
                                     "buy_and_hold_minus_constant": float(diff.std() * BPS_YR)}
    out["control_variate_correction_bps_yr"] = {"mean": float(diff.mean() * BPS_YR),
                                                "se": float(diff.std() / np.sqrt(n_paths) * BPS_YR)}
    turnover = 0.5 * np.abs(w_end - w0).sum(axis=1)
    out["drift_one_way_turnover_pct"] = {"median": float(np.median(turnover) * 100),
                                         "mean": float(turnover.mean() * 100)}
    gap = np.abs(w_end - w_end_no_rf).max(axis=1)
    out["rf_effect_on_end_weights"] = {"median": float(np.median(gap)), "max": float(gap.max())}

    # How big are the decisions the agent must resolve? Expected-utility loss
    # from letting the optimum drift, and the 2% grid's own discretization gap.
    def utility(w):
        return w @ mu - GAMMA_HEADLINE / 2 * np.einsum("...i,ij,...j->...", w, sigma, w)
    grid2 = generate_weight_grid(3, 0.02)
    out["continuous_minus_best_2pct_target_bps_yr"] = float((utility(w0) - utility(grid2).max()) * BPS_YR)
    held = np.repeat(w0[None, :], n_paths, 0)
    drift_loss = {}
    for k in range(1, 21):
        block = rng.multivariate_normal(mu, sigma, size=(n_paths, BLOCK_LENGTH))
        for t in range(BLOCK_LENGTH):
            grown = held * (1.0 + float(rf.mean()) + block[:, t])
            held = grown / grown.sum(axis=1, keepdims=True)
        if k in (1, 4, 20):
            drift_loss[f"after_{k}_blocks"] = float((utility(w0) - utility(held).mean()) * BPS_YR)
    out["utility_lost_to_drift_from_optimum_bps_yr"] = drift_loss

    # Three-asset spline map, built the way src/features.py builds the nine-asset one.
    W = generate_weight_grid(3, 0.02)
    B = bspline_basis(W.ravel()).reshape(len(W), 3, 5)
    blocks = [B.reshape(len(W), 15)] + [(B[:, i, :, None] * B[:, j, None, :]).reshape(len(W), 25)
                                        for i in range(3) for j in range(i + 1, 3)]
    F = np.hstack(blocks)
    F = F[:, F.max(axis=0) > 0]
    out["spline_features_3_assets_2pct_grid"] = {"columns": int(F.shape[1]),
                                                 "rank": int(np.linalg.matrix_rank(F)),
                                                 "max_row_norm_sq": float((F**2).sum(axis=1).max())}
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
