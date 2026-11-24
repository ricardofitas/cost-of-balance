"""Certify rational bounds on the maximum-adjacency five-order minimax risk."""
import json
from fractions import Fraction
from itertools import permutations, product
from math import lcm
from pathlib import Path


ORDERS = tuple(permutations(range(5)))
INDEX = {order: i for i, order in enumerate(ORDERS)}
N = len(ORDERS)
K = 5


def orbit(rows):
    return {tuple(sorted(INDEX[tuple(p[v] for v in row)] for row in rows)) for p in ORDERS}


def adjacency(block):
    return len({pair for i in block for pair in zip(ORDERS[i], ORDERS[i][1:])})


def all_maximal_blocks():
    buckets = {
        a: [p for p in ORDERS if p[0] == a and all(i != x for i, x in enumerate(p))]
        for a in range(1, K)
    }
    blocks = [(ORDERS[0], *rows) for rows in product(*(buckets[i] for i in range(1, K)))
              if all(len(set(col)) == K for col in zip(ORDERS[0], *rows))]
    assert len(blocks) == 56
    coverages = {tuple(INDEX[row] for row in block): adjacency(tuple(INDEX[row] for row in block))
                 for block in blocks}
    assert max(coverages.values()) == 15
    return set().union(*(orbit([ORDERS[i] for i in b])
                         for b, value in coverages.items() if value == 15))


def certify_psd(matrix):
    """An exact integer Schur-complement elimination."""
    previous = 1
    for column in range(len(matrix)):
        pivot = matrix[column][column]
        assert pivot >= 0
        if pivot == 0:
            assert not any(matrix[column][column+1:])
            continue
        for row in range(column+1, len(matrix)):
            for j in range(row, len(matrix)):
                numerator = matrix[row][j]*pivot - matrix[row][column]*matrix[column][j]
                value, remainder = divmod(numerator, previous)
                assert remainder == 0
                matrix[row][j] = matrix[j][row] = value
        previous = pivot


def lower_risks(orbits, numerators, vectors, denominator):
    assert sum(numerators) == denominator and min(numerators) >= 0
    risks = [Fraction(0) for _ in orbits]
    for weight, raw in zip(numerators, vectors, strict=True):
        assert len(raw) == N and all(type(x) is int for x in raw)
        centered = [N*x-sum(raw) for x in raw]
        square_norm = sum(x*x for x in centered)
        assert square_norm > 0 and sum(centered) == 0
        for index, one_orbit in enumerate(orbits):
            total = sum(sum(centered[i] for i in block)**2 for block in one_orbit)
            risks[index] += Fraction(weight*N*total, denominator*K*K*len(one_orbit)*square_norm)
    return risks


def upper_bound(orbits, weights, denominator, bound):
    assert sum(weights) == denominator and min(weights) >= 0
    multiple = lcm(*(len(item) for item in orbits))
    total_weight = denominator * multiple
    inclusions = [[0 for _ in ORDERS] for _ in ORDERS]
    for orbit_blocks, weight in zip(orbits, weights, strict=True):
        scaled = weight * (multiple // len(orbit_blocks))
        for block in orbit_blocks:
            for a, b in product(block, repeat=2):
                inclusions[a][b] += scaled
    assert all(inclusions[i][i]*N == total_weight*K for i in range(N))
    scale = lcm(K*K*total_weight, N, bound.denominator)
    matrix = [
        [(int(i == j)*bound.numerator*(scale//bound.denominator))
         - N*inclusions[i][j]*(scale//(K*K*total_weight)) + scale//N
         for j in range(N)] for i in range(N)
    ]
    certify_psd(matrix)


def main():
    pins = json.loads(Path(__file__).with_name("solutions.json").read_text())
    denominator = pins["denominator"]
    orbits = [orbit([tuple(map(int, row)) for row in representative])
              for representative in pins["representatives"]]
    assert len(orbits) == 5
    admitted = all_maximal_blocks()
    assert len(admitted) == 216 and sum(map(len, orbits)) == 216
    assert set().union(*orbits) == admitted
    risks = lower_risks(orbits, pins["dual_weight_numerators"], pins["dual_score_vectors"], denominator)
    lower, upper = Fraction(47084, 100000), Fraction(47087, 100000)
    assert min(risks) >= lower
    upper_bound(orbits, pins["primal_weight_numerators"], denominator, upper)
    print(json.dumps({"real_score_normalized_minimax_interval": [str(lower), str(upper)],
                      "maximum_adjacency_blocks": len(admitted), "option_relabeling_orbits": len(orbits),
                      "bound_width": str(upper-lower), "exact_lower_pass": True,
                      "exact_upper_pass": True, "benchmark_results_opened": False}, indent=2))


if __name__ == "__main__":
    main()
