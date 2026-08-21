from __future__ import annotations

# Adapted from SeqE-Guard.R at commit e9cd07d71a29b7f61b9c9706ce480823497af277.
# Copyright (c) 2024 fischer23. Distributed under the MIT License.

import math
from collections.abc import Iterable


def seqe_guard(
    e_values: Iterable[float],
    queried_indices: Iterable[int],
    *,
    alpha: float,
) -> tuple[int, ...]:
    """Return the official SeqE-Guard bound using its one-based index contract."""
    values = tuple(float(value) for value in e_values)
    queried = frozenset(queried_indices)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("e-values must be a non-empty finite nonnegative sequence")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be within (0, 1)")
    if any(
        type(index) is not int or not 1 <= index <= len(values) for index in queried
    ):
        raise ValueError("queried indices must use the one-based e-value range")

    active: list[int] = []
    unused: list[int] = []
    bounds: list[int] = []
    for index, value in enumerate(values, start=1):
        bound = bounds[-1] if bounds else 0
        if index in queried:
            active.append(index)
            product = math.prod(values[item - 1] for item in (*active, *unused))
            if product >= 1 / alpha:
                bound += 1
                largest = max(values[item - 1] for item in active)
                remove_at = max(
                    position
                    for position, item in enumerate(active)
                    if values[item - 1] == largest
                )
                active.pop(remove_at)
        elif value < 1:
            unused.append(index)
        bounds.append(bound)
    return tuple(bounds)
