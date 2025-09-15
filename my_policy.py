from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from policy import Policy
from mdp import MDP, State, Action
from mdp_utils import enumerate_states

try:
    from lake_mdp import UP, RIGHT, DOWN, LEFT, ABSORB
except Exception:
    UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "⊥"

class MyPolicy(Policy):
    """
    Value-free constructive policy based on shortest-path over the MDP's
    most-likely-successor graph (no v^pi, no returns, no evaluation calls).

    Steps:
      1) Enumerate reachable states.
      2) Build a directed graph: for each (state, action), connect to the
         most likely successor under mdp.transition(s, a).
      3) Reverse-BFS from all goals and the absorbing state to compute
         a discrete distance d(s).
      4) For each non-terminal state s, choose the action minimizing d(next).
         Tie-break with a fixed action order.

    Notes:
      • Holes are treated as terminals and are not seeded in BFS, so d(H)=∞,
        which naturally discourages stepping into holes unless unavoidable.
      • Absorbing and goals get d=0.
    """

    def __init__(
        self,
        mdp: MDP,
        rng: np.random.Generator,
        tie_break: Tuple[Action, ...] = (RIGHT, DOWN, LEFT, UP),
    ):
        super().__init__(mdp, rng)
        self._tb: Tuple[Action, ...] = tie_break
        self._S: List[State] = []
        self._idx: Dict[State, int] = {}
        self._rev: Dict[State, List[Tuple[State, Action]]] = {}
        self._D: Dict[State, float] = {}
        self._act: Dict[State, Action] = {}
        self._prepare()

    def _decision(self, s: State) -> Action:
        aa = list(self.mdp.actions(s))
        if aa == [ABSORB]:
            return ABSORB
        z = self._act.get(s)
        return z if z is not None else self._tb[0]

    def _ml_succ(self, s: State, a: Action) -> Optional[State]:
        dist = self.mdp.transition(s, a)
        if not dist:
            return None
        j, pj = dist[0]
        for t, pt in dist[1:]:
            if pt > pj:
                j, pj = t, pt
        return j

    def _prepare(self) -> None:
        self._S = enumerate_states(self.mdp)
        self._idx = {s: i for i, s in enumerate(self._S)}
        self._rev = {s: [] for s in self._S}

        for s in self._S:
            aa = list(self.mdp.actions(s))
            if aa == [ABSORB]:
                continue
            for a in (UP, RIGHT, DOWN, LEFT):
                if a not in aa:
                    continue
                ns = self._ml_succ(s, a)
                if ns is not None:
                    self._rev.setdefault(ns, []).append((s, a))

        from collections import deque

        INF = float("inf")
        self._D = {s: INF for s in self._S}
        seeds: List[State] = []
        for s in self._S:
            try:
                if s[0] == ABSORB or s[1] == ABSORB or s[1] == "G":
                    seeds.append(s)
            except Exception:
                pass

        q = deque(seeds)
        for t in seeds:
            self._D[t] = 0.0

        while q:
            x = q.popleft()
            dx = self._D[x]
            for (p, _a) in self._rev.get(x, []):
                nd = dx + 1.0
                if self._D[p] > nd:
                    self._D[p] = nd
                    q.append(p)

        for s in self._S:
            aa = list(self.mdp.actions(s))
            if aa == [ABSORB]:
                continue
            best = None
            bd = float("inf")
            for a in self._tb:
                if a not in aa:
                    continue
                ns = self._ml_succ(s, a)
                if ns is None:
                    continue
                d = self._D.get(ns, float("inf"))
                if d < bd:
                    bd, best = d, a
            self._act[s] = best if best is not None else self._tb[0]