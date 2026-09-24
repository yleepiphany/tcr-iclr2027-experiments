"""Boundary propagation shared by matched RegMean / Block RegMean++ replay."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")


def advance_by_prefix(
    states: Mapping[str, Sequence[T]],
    mode: str,
    assign_expert: Callable[[str], None],
    restore_merged: Callable[[], None],
    advance: Callable[[T], None],
) -> None:
    """Advance each state once; expert mode never leaves expert weights installed."""
    if mode not in ("merged", "expert"):
        raise ValueError(f"Unknown replay prefix: {mode}")
    try:
        for name, group in states.items():
            if mode == "expert":
                assign_expert(name)
            for state in group:
                advance(state)
    finally:
        if mode == "expert":
            restore_merged()
