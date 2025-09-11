# plot_utils.py
from __future__ import annotations
import matplotlib.pyplot as plt

# Import env action constants (works if they are defined; otherwise we compare by string)
try:
    from lake_mdp import UP, RIGHT, DOWN, LEFT, ABSORB
except Exception:
    UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "ABSORB"

# Map action names -> arrow glyphs (ASCII-friendly fallbacks provided)
ACTION_TO_ARROW = {
    "UP": "↑",
    "RIGHT": "→",
    "DOWN": "↓",
    "LEFT": "←",
}


def _action_name(a):
    """Normalize an action object to 'UP'/'RIGHT'/'DOWN'/'LEFT' when possible."""
    if a == UP:
        return "UP"
    if a == RIGHT:
        return "RIGHT"
    if a == DOWN:
        return "DOWN"
    if a == LEFT:
        return "LEFT"
    # Fallback for string-like actions
    if isinstance(a, str):
        s = a.strip().upper()
        if s in ACTION_TO_ARROW:
            return s
    return None  # unknown / ABSORB / other


def plot_policy(policy, ax=None):
    """
    Plot the LakeMDP board using only the policy:
      - Extract mdp via policy.mdp
      - Color S (orange), H (blue), G (green); all other cells white
      - For every non-(S/H/G) cell, draw the arrow dictated by policy(s)

    Assumptions:
      - policy is callable on a *state* s and exposes `policy.mdp`
      - `policy.mdp.grid` is a list[list[str]] with characters in {'S','F','H','G'}
      - states are addressed as (i, j) tuples when calling policy((i, j))
    """
    mdp = policy.mdp
    grid = mdp.grid
    m = len(grid)
    n = len(grid[0]) if m else 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
        ax.set_facecolor("white")

    # Iterate rows/cols once: draw cell + add character (letter or arrow)
    for i in range(m):
        for j in range(n):
            ch = grid[i][j]

            # choose cell color
            if ch == "S":
                color, txt, txt_color = (1.0, 0.6, 0.0), "S", "white"  # orange
            elif ch == "H":
                color, txt, txt_color = (0.2, 0.4, 1.0), "H", "white"  # blue
            elif ch == "G":
                color, txt, txt_color = (0.0, 0.7, 0.0), "G", "white"  # green
            else:
                # Free cell: white, text decided by the policy
                color, txt, txt_color = (1.0, 1.0, 1.0), None, "black"

            # draw the box
            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1, facecolor=color, edgecolor="lightgray", linewidth=1.0
                )
            )

            # decide the character to place
            if txt is None:
                # ask the policy for this state's action, then map to arrow
                s = (i, j)
                try:
                    a = policy((s, ch))
                    print(f"Policy action at state {s}: {a}")
                except Exception:
                    print(f"Policy action at state {s}: <error>")
                    a = None
                name = _action_name(a)
                txt = ACTION_TO_ARROW.get(name, "·")  # dot if unknown / ABSORB

            # put the character in the center
            ax.text(
                j + 0.5,
                i + 0.55,
                txt,
                ha="center",
                va="center",
                fontsize=14,
                color=txt_color,
                weight="bold",
            )

    # tidy axes
    ax.set_xlim(0, n)
    ax.set_ylim(m, 0)  # flip to match grid layout
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return ax
