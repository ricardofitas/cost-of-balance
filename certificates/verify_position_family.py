"""Exact character spectra of randomized Latin blocks, using only the standard library."""

import json
from collections import Counter
from fractions import Fraction
from functools import cache
from itertools import combinations
from math import factorial, prod
from pathlib import Path


@cache
def partitions(n, largest=None):
    if n == 0:
        return ((),)
    return tuple((part,) + tail for part in range(min(n, largest or n), 0, -1)
                 for tail in partitions(n - part, part))


@cache
def character(shape, cycles):
    """Murnaghan--Nakayama, removing a connected border strip without a 2x2."""
    if not cycles:
        return int(not shape)
    total = 0
    for inner in partitions(sum(shape) - cycles[0]):
        if len(inner) > len(shape) or any(a > b for a, b in zip(inner, shape)):
            continue
        cells = {(i, j) for i, width in enumerate(shape)
                 for j in range(inner[i] if i < len(inner) else 0, width)}
        if any((i+1, j) in cells and (i, j+1) in cells and (i+1, j+1) in cells
               for i, j in cells):
            continue
        reached = {next(iter(cells))}
        stack = list(reached)
        while stack:
            i, j = stack.pop()
            for neighbor in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if neighbor in cells and neighbor not in reached:
                    reached.add(neighbor)
                    stack.append(neighbor)
        if reached == cells:
            height = len({i for i, _ in cells}) - 1
            total += (-1)**height * character(inner, cycles[1:])
    return total


def relative_cycles(p, q):
    inverse = {v: i for i, v in enumerate(p)}
    relative = [inverse[v] for v in q]
    remaining = set(range(len(p)))
    sizes = []
    while remaining:
        j = next(iter(remaining))
        size = 0
        while j in remaining:
            remaining.remove(j)
            j = relative[j]
            size += 1
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))


def signature(rows):
    n = len(rows)
    symbols = set(range(n))
    assert all(len(row) == n and set(row) == symbols for row in rows)
    assert all(set(column) == symbols for column in zip(*rows))
    pairs = Counter(relative_cycles(a, b) for a, b in combinations(rows, 2))
    assert all(1 not in cycles for cycles in pairs)
    return pairs


def class_size(cycles):
    divisor = prod(length**count * factorial(count)
                   for length, count in Counter(cycles).items())
    return factorial(sum(cycles)) // divisor


def spectrum(n, pairs):
    values = {}
    for shape in partitions(n):
        rank = character(shape, (1,)*n)
        off_diagonal = sum(count * character(shape, cycles) for cycles, count in pairs.items())
        values[shape] = Fraction(n*rank + 2*off_diagonal, n*n*rank)
    assert values.pop((n,)) == 1
    assert sum(value * character(shape, (1,)*n)**2 for shape, value in values.items()) == factorial(n)//n-1
    assert min(values.values()) >= 0
    return values


def check_table(n):
    shapes = partitions(n)
    sizes = {c: class_size(c) for c in shapes}
    for i, a in enumerate(shapes):
        for j, b in enumerate(shapes[:i+1]):
            inner = sum(sizes[c]*character(a, c)*character(b, c) for c in shapes)
            assert inner == (factorial(n) if i == j else 0)


def verify():
    templates = json.loads(Path(__file__).with_name("position-templates.json").read_text())
    results = {}
    for key, support in templates.items():
        n = int(key)
        check_table(n)
        law = {shape: Fraction() for shape in partitions(n) if shape != (n,)}
        total = Fraction()
        for atom in support:
            weight = Fraction(atom["weight"])
            assert weight > 0
            total += weight
            rows = [tuple(row) for row in atom["rows"]]
            assert len(rows) == n
            for shape, value in spectrum(n, signature(rows)).items():
                law[shape] += weight * value
        assert total == 1
        bound = Fraction(n-2, n*n-3*n+1)
        assert max(law.values()) == bound
        assert law[n-2, 2] == law[n-2, 1, 1] == bound
        results[key] = {"minimax": str(bound), "templates": len(support),
                        "nontrivial_eigenvalues_by_partition": {
                            ",".join(map(str, shape)): str(value) for shape, value in law.items()}}
    return results


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
