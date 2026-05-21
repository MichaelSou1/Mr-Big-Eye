from __future__ import annotations

import random


USERNAME_POOL: list[str] = [
    "Witch",
    "Wizard",
    "Oracle",
    "Phantom",
    "Specter",
    "Queen",
    "King",
    "Duke",
    "Knight",
    "Baron",
    "Phoenix",
    "Dragon",
    "Falcon",
    "Wolf",
    "Raven",
    "Ace",
    "Hero",
    "Sage",
    "Bard",
    "Scout",
]


def pick_username(taken: set[str]) -> str:
    """Pick a memorable username that is not already taken."""
    available = [name for name in USERNAME_POOL if name not in taken]
    if available:
        return random.choice(available)

    base = random.choice(USERNAME_POOL)
    suffix = 2
    while f"{base}_{suffix}" in taken:
        suffix += 1
    return f"{base}_{suffix}"
