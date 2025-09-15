from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np

from my_policy import MyPolicy
from mdp_utils import enumerate_states, build_policy_Pr
from plot_utils import plot_policy
from bellman import exact_policy_evaluation
from policy_eval import iterative_policy_evaluation
from mdp import MDP


def run(
    mdp: MDP,
    gamma: float,
    rng: Optional[np.random.Generator] = None,
    method: str = "exact",
):
    """
    Build a value-free policy for the given MDP, evaluate it, and report fitness.

    Parameters
    ----------
    mdp : MDP
        Environment implementing the abstract interface in mdp.py
        (reward-on-entry convention).
    gamma : float
        Discount factor in (0,1].
    rng : numpy.random.Generator, optional
        Randomness source used inside the policy constructor (if needed).
    method : {"exact","iterative"}
        Which policy evaluation method to use.

    Returns
    -------
    (pi, v, f_pi) :
        pi   : the constructed deterministic policy (callable Policy)
        v    : (S,) state-value vector aligned with the internal state ordering
        f_pi : scalar fitness f^{\hat{π}} = v^{\hat{π}}(s_0)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    pi = MyPolicy(mdp, rng)
    S = enumerate_states(mdp)
    P, r = build_policy_Pr(mdp, pi, S)

    if method == "exact":
        v = exact_policy_evaluation(P, r, gamma)
    elif method == "iterative":
        v = iterative_policy_evaluation(P, r, gamma)
    else:
        raise ValueError("method must be one of {'exact', 'iterative'}")

    f = float(v[0])
    
    return pi, v, f