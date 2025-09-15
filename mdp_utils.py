from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from lake_mdp import ABSORB
from policy import Policy


def enumerate_states(mdp) -> List[object]:
    """
    Deterministically enumerate all states reachable from the start state.

    States are treated as opaque hashables and can be shaped like:
      - ((i, j), ch) for grid cells, where ch ∈ {'S','F','H','G'}
      - (ch, ch) for the absorbing state

    We rely only on the environment's own transitions.
    """
    s0 = mdp.start_state()
    seen = {s0}
    out: List[object] = [s0]
    stk = [s0]
    while stk:
        s = stk.pop()
        for a in mdp.actions(s):
            for ns, _p in mdp.transition(s, a):
                if ns not in seen:
                    seen.add(ns)
                    out.append(ns)
                    stk.append(ns)

    return out

def build_policy_Pr(
    mdp, policy: Policy, states: List[object]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the policy-induced transition matrix P and reward vector r for a fixed policy.

    P[i, j] = Pr(S_{t+1} = states[j] | S_t = states[i], A_t ~ π(S_t))
    r[i]    = reward(states[i]) using the environment's reward convention (on entry).

    Terminal/absorbing states are detected via actions(s) == [ABSORB] and given a self-loop.
    """
    
    n = len(states)
    idx: Dict[object, int] = {s: i for i, s in enumerate(states)}
    P = np.zeros((n, n), dtype=float)

    for s in states:
        i = idx[s]
        acts = list(mdp.actions(s))
        if acts == [ABSORB]:
            P[i, idx[(ABSORB, ABSORB)]] = 1.0
            continue

        ap = getattr(policy, "action_probs", None)
        probs = ap(s) if callable(ap) else None
        if probs:
            for a, pa in probs.items():
                if pa <= 0.0:
                    continue
                for ns, p in mdp.transition(s, a):
                    P[i, idx[ns]] += float(pa) * float(p)
        else:
            a = policy(s)
            for ns, p in mdp.transition(s, a):
                P[i, idx[ns]] += float(p)

        rs = P[i].sum()
        if rs > 0 and abs(rs - 1.0) > 1e-8:
            P[i] /= rs

    rew = np.array([mdp.reward(t) for t in states], dtype=float)
    r = P @ rew

    return P, r