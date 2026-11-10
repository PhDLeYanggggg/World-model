"""Uniform prefix allocation under a fixed count and chronological credits."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrefixSpace:
    indices: np.ndarray
    options: tuple
    suffix: list
    total: int

    @property
    def size(self):
        return self.suffix[0][0][self.total]

    def unrank(self, rank):
        """Return sparse (query, prefix length) arrays for one exact rank."""
        if not 0 <= rank < self.size:
            raise ValueError("Allocation rank outside support")
        left, balance = self.total, 0
        queries, lengths = [], []
        for i, choices in enumerate(self.options):
            if not left:
                break
            credit = min(10, balance + 1)
            for k in choices:
                spend = 5*k
                if spend <= credit and k <= left:
                    ways = self.suffix[i+1][credit-spend][left-k]
                    if rank < ways:
                        if k:
                            queries.append(self.indices[i])
                            lengths.append(k)
                        balance, left = credit-spend, left-k
                        break
                    rank -= ways
        if left:
            raise AssertionError("Invalid suffix counts")
        return np.asarray(queries, dtype=int), np.asarray(lengths, dtype=int)


def prefix_space(legal, total, individually_under_cap):
    """Count exactly before the aggregate movement-cap rejection step."""
    if (legal.ndim != 2 or legal.shape[1] != 3 or not legal[:, 0].all()
            or (legal[:, 2] & ~legal[:, 1]).any() or total < 0
            or individually_under_cap.shape != legal.shape):
        raise ValueError("Expected a nested zero/one/two menu")
    indices = np.flatnonzero(legal[:, 1])
    choices = legal & individually_under_cap
    # Eligible queries still earn credit when none of their edits fit the cap.
    options = tuple((0, *(k for k in (1, 2) if choices[i, k])) for i in indices)
    suffix = [None]*len(options) + [[[1] + [0]*total for _ in range(11)]]
    for i in range(len(options)-1, -1, -1):
        tail = suffix[i+1]
        layer = []
        for b in range(11):
            credit = min(10, b+1)
            ways = tail[credit].copy()
            for k in options[i][1:]:
                if credit >= 5*k:
                    add = tail[credit-5*k]
                    for t in range(k, total+1):
                        ways[t] += add[t-k]
            layer.append(ways)
        suffix[i] = layer
    return PrefixSpace(indices, options, suffix, total)
