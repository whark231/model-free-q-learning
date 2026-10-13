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
    if returns_batch.ndim != 2 or returns_batch.shape[1] != w.shape[0]:
        raise ValueError(
            f"returns_batch must be (B, {w.shape[0]}), got {returns_batch.shape}"
        )
    if returns_batch.shape[0] < 2:
        raise ValueError("episodic reward needs at least 2 days (ddof=1 variance)")

    portfolio_returns = returns_batch @ w    # size B
    realized_mean = portfolio_returns.mean()
    realized_var = portfolio_returns.var(ddof=1)

    return realized_mean - 0.5 * gamma * realized_var