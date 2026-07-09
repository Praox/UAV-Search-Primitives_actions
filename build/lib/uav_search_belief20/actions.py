from __future__ import annotations

STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

ACTION_NAMES = {
    STAY: "stay",
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
}

MOVES = {
    STAY: (0, 0),
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

ACTION_DIM = 5
