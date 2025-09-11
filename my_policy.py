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
        self._tie_break: Tuple[Action, ...] = tie_break
        self._states: List[State] = []
        self._index: Dict[State, int] = {}
        self._preds: Dict[State, List[Tuple[State, Action]]] = {}
        self._dist: Dict[State, float] = {}
        self._choice: Dict[State, Action] = {}
        self._build()

    # -- Policy API ----------------------------------------------------------
    def _decision(self, s: State) -> Action:
        # Si es terminal o absorbente, devuelve ABSORB
        acts = list(self.mdp.actions(s))
        if acts == [ABSORB]:
            return ABSORB
        # Acción precomputada; fallback: tie-break inicial
        a = self._choice.get(s)
        return a if a is not None else self._tie_break[0]

    # -- Internals -----------------------------------------------------------
    def _most_likely_successor(self, s: State, a: Action) -> Optional[State]:
        """Devuelve el sucesor más probable de (s,a)."""
        dist = self.mdp.transition(s, a)
        if not dist:
            return None
        # Elegir argmax por probabilidad; desempate estable por orden de aparición
        best_ns, best_p = dist[0]
        for ns, p in dist[1:]:
            if p > best_p:
                best_ns, best_p = ns, p
        return best_ns

    def _build(self) -> None:
        # 1) Estados alcanzables
        self._states = enumerate_states(self.mdp)
        self._index = {s: i for i, s in enumerate(self._states)}

        # 2) Grafo por sucesor más probable: construimos predecesores
        self._preds = {s: [] for s in self._states}
        for s in self._states:
            acts = list(self.mdp.actions(s))
            if acts == [ABSORB]:
                continue  # no salientes desde terminal/absorbente
            for a in (UP, RIGHT, DOWN, LEFT):
                if a not in acts:
                    continue
                ns = self._most_likely_successor(s, a)
                if ns is None:
                    continue
                # Guardamos (predecesor, acción) para el reverso
                self._preds.setdefault(ns, []).append((s, a))

        # 3) Distancias por BFS inverso desde objetivos
        from collections import deque

        INF = float("inf")
        self._dist = {s: INF for s in self._states}

        # Seeds: todos los estados con símbolo 'G' y el absorbente (ABSORB, ABSORB)
        seeds: List[State] = []
        for s in self._states:
            try:
                if s[0] == ABSORB or s[1] == ABSORB:
                    seeds.append(s)
                elif s[1] == "G":
                    seeds.append(s)
            except Exception:
                # Estados no conformes al LakeMDP; ignorar
                pass

        q = deque()
        for t in seeds:
            self._dist[t] = 0.0
            q.append(t)

        while q:
            x = q.popleft()
            dx = self._dist[x]
            for (pred, _) in self._preds.get(x, []):
                if self._dist[pred] > dx + 1.0:
                    self._dist[pred] = dx + 1.0
                    q.append(pred)

        # 4) Elegimos acción que minimiza distancia del sucesor
        for s in self._states:
            acts = list(self.mdp.actions(s))
            if acts == [ABSORB]:
                continue
            best_a = None
            best_d = float("inf")
            # desempate por orden fijo de acciones
            for a in self._tie_break:
                if a not in acts:
                    continue
                ns = self._most_likely_successor(s, a)
                if ns is None:
                    continue
                d = self._dist.get(ns, float("inf"))
                if d < best_d:
                    best_d = d
                    best_a = a
            # Si todo es inf (ej. rodeado por hoyos), usar desempate fijo
            if best_a is None:
                best_a = self._tie_break[0]
            self._choice[s] = best_a
