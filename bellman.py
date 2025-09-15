from __future__ import annotations
import numpy as np

ArrayLike = np.ndarray


def bellman_update(v: ArrayLike, P: ArrayLike, r: ArrayLike, gamma: float) -> ArrayLike:
    """
    Apply one Bellman update for a fixed policy.
    Vectorized: v_new = r + gamma * (P @ v)

    Parameters
    ----------
    v : (S,) array
        Current value estimates.
    P : (S,S) array
        Policy-induced transition matrix (row-stochastic).
    r : (S,) array
        Reward-on-entry vector aligned to the state indexing.
    gamma : float in (0,1]
        Discount factor.

    Returns
    -------
    v_new : (S,) array
    """
  
    return r + gamma * (P @ v) # Returns v_new : (S,) array


def exact_policy_evaluation(P: ArrayLike, r: ArrayLike, gamma: float) -> ArrayLike:
    """
    Solve (I - gamma P) v = r for v.

    Parameters
    ----------
    P : (S,S) array
    r : (S,) array
    gamma : float in (0,1]

    Returns
    -------
    v : (S,) array
    """
    n = P.shape[0]
    A = np.eye(n, dtype=float) - gamma * P

    return np.linalg.solve(A, r) # Returns v : (S,) array
