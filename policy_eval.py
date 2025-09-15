from __future__ import annotations
import numpy as np

from bellman import bellman_update


def iterative_policy_evaluation(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    eps: float = 1e-6,
    max_iters: int = 100000,
) -> np.ndarray:
    """
    Iterative (incremental) policy evaluation via repeated Bellman updates.
    Stop when ||v_new - v||_inf < eps * (1 - gamma) / gamma

    Parameters
    ----------
    P : (S,S) array (row-stochastic)
    r : (S,) array
    gamma : float in (0,1]
    eps : float
        Target error tolerance (on true error via slide bound).
    max_iters : int

    Returns
    -------
    v : (S,) array
    """
    S = P.shape[0]
    v = np.zeros(S, dtype=float)

    n = P.shape[0]
    w = np.zeros(n, dtype=float)
    thr = eps if gamma == 1.0 else eps * (1.0 - gamma) / gamma
    k = 0
    while k < max_iters:
        nxt = bellman_update(w, P, r, gamma)
        if np.max(np.abs(nxt - w)) < thr:
            w = nxt
            break
        w = nxt
        k += 1

    return w # Returns  v : (S,) array
