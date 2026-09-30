import numpy as np


def episodic_mean_variance_reward(w: np.ndarray, returns_batch: np.ndarray, gamma: float) -> float:
    """
    Episodic realized mean-variance reward (recommended default).

    Given a chosen portfolio `w` and a batch of B simulated realized returns
    (returns_batch, shape (B, n_assets)), computes the portfolio's realized
    return on each of the B days, then:

        R = mean(portfolio_returns) - (gamma/2) * var(portfolio_returns)

    ddof=1 (Bessel's correction) makes the variance term an UNBIASED estimator
    of the true portfolio variance which causes expected return to converge to mean-variance benchmark.
    """
    portfolio_returns = returns_batch @ w    # size B
    realized_mean = portfolio_returns.mean()
    realized_var = portfolio_returns.var(ddof=1)

    return realized_mean - 0.5 * gamma * realized_var