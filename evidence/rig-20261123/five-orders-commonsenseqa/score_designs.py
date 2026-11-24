"""Exact weighted risks and paired intervals for the fixed five-order samplers."""
from collections import Counter
from fractions import Fraction
from itertools import permutations, product
from math import lcm
import json
from pathlib import Path

import numpy as np

from verify_position import BLOCK, blocks_with_identity, compose, CYCLIC
from verify_real_adjacency import adjacency, all_maximal_blocks, orbit

ROOT = Path(__file__).resolve().parent
ORDERS = tuple(permutations(range(5)))
INDEX = {row: i for i, row in enumerate(ORDERS)}
POSITION, MAXIMAL = "position-minimax", "maximum-adjacency"


def blocks_with_option_relabelings(rows):
    return {tuple(sorted(INDEX[compose(p, row)] for row in rows)) for p in ORDERS}


def supports():
    position = Counter(tuple(sorted(INDEX[compose(left, compose(row, right))] for row in BLOCK))
                       for left, right in product(ORDERS, repeat=2))
    solution = json.loads((ROOT / "solutions.json").read_text())
    reps = [[tuple(map(int, row)) for row in rep] for rep in solution["representatives"]]
    orbits = [orbit(rep) for rep in reps]
    assert set.union(*orbits) == all_maximal_blocks()
    assert sum(map(len, orbits)) == len(set.union(*orbits))
    base = lcm(*(len(group) for group in orbits))
    max_adj = {block: (base // len(group)) * weight
               for group, weight in zip(orbits, solution["primal_weight_numerators"], strict=True)
               for block in group}
    cyclic = dict.fromkeys(blocks_with_option_relabelings(CYCLIC), 1)
    all_balanced = dict.fromkeys(set.union(*(blocks_with_option_relabelings(block) for block in blocks_with_identity())), 1)
    laws = {POSITION: position, MAXIMAL: max_adj, "ordinary-cyclic": cyclic, "uniform-position": all_balanced}
    assert len(all_balanced) == 1344
    assert {adjacency(block) for block in max_adj} == {15}
    assert {adjacency(block) for block in position} == {11, 12, 14, 15}
    for law in laws.values():
        inclusions = Counter()
        for block, weight in law.items():
            rows = tuple(ORDERS[index] for index in block)
            assert all(len(set(col)) == 5 for col in zip(*rows, strict=True))
            for index in block:
                inclusions[index] += weight
        assert len(inclusions) == 120 and set(inclusions.values()) == {sum(law.values()) // 24}
    return laws


def measure(gap, laws):
    n = len(gap)
    total = gap.sum(axis=1)[:, None]
    risks, entries = {}, {}
    for design, law in laws.items():
        blocks = list(law)
        weights = np.array(list(law.values()), dtype=np.int64)
        errors = 24 * gap[:, blocks].sum(axis=2) - total
        numerator = np.square(errors) @ weights
        denominator = 14400 * sum(law.values())
        loss = numerator / denominator
        exact = Fraction(int(numerator.sum()), denominator*n*n)
        # Independently apply weighted outer products of the centered block incidences.
        centered = np.full((len(blocks), 120), -1, dtype=np.int64)
        centered[np.arange(len(blocks))[:, None], blocks] += 24
        gram = centered.T @ (centered * weights[:, None])
        assert np.array_equal(((gap @ gram) * gap).sum(axis=1), numerator)
        risks[design] = loss
        entries[design] = {"mse": float(exact), "exact_mse": str(exact), "support": len(law),
                           "mean_adjacency": sum(adjacency(b) * w for b,w in law.items()) / sum(law.values())}
    base, alternative = (risks[x] for x in (POSITION, MAXIMAL))
    point = 1 - alternative.mean() / base.mean()
    generator = np.random.default_rng(22112305)
    intervals = []
    for _ in range(100):
        selection = generator.integers(n, size=(500, n))
        baseline = base[selection].mean(axis=1)
        assert np.all(baseline > 0)
        intervals.extend(1 - alternative[selection].mean(axis=1) / baseline)
    lo, hi = np.quantile(intervals, [0.025, 0.975]).tolist()
    return {"designs": entries, "relative_gain": float(point), "paired_bootstrap_95": [lo, hi],
            "practical_margin": 0.05,
            "practical_decision": "supports" if lo > .05 else "excludes" if hi < .05 else "inconclusive"}
