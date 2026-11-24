"""Exact minimax certificate for five equal-weight position-balanced orders."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations, product
import json


PERMS = tuple(permutations(range(5)))
IDENTITY = PERMS[0]
BLOCK = tuple(tuple(map(int, p)) for p in ("01234", "12340", "24013", "30421", "43102"))
CYCLIC = tuple(IDENTITY[i:]+IDENTITY[:i] for i in range(5))
F = Fraction


def compose(p, q):
    return tuple(p[i] for i in q)


def inverse(p):
    return tuple(p.index(i) for i in range(len(p)))


def balanced_block(block):
    return len(block) == 5 and all(len(set(column)) == 5 for column in zip(*block))


def sign(p):
    return (-1) ** sum(p[i] > p[j] for i, j in combinations(range(5), 2))


def blocks_with_identity():
    # Sorting rows by their first element gives a unique normalized block.
    candidates = {i: [p for p in PERMS if p[0] == i and all(j != v for j, v in enumerate(p))]
                  for i in range(1, 5)}
    for rows in product(*(candidates[i] for i in range(1, 5))):
        block = (IDENTITY, *rows)
        if balanced_block(block):
            yield block


def cycle_type(p):
    unseen = set(range(5))
    lengths = []
    while unseen:
        current = next(iter(unseen))
        size = 0
        while current in unseen:
            unseen.remove(current)
            current = p[current]
            size += 1
        lengths.append(size)
    return tuple(sorted(lengths))


def covariance_kernel(block):
    """Form the exact covariance by separately applying all 120^2 relabelings."""
    counts = Counter()
    for left, right in product(PERMS, repeat=2):
        selected = tuple(compose(left, compose(order, right)) for order in block)
        assert balanced_block(selected)
        counts.update(product(selected, repeat=2))
    assert sum(counts.values()) == 360000
    # The normalized covariance kernel at the identity is count/3000 - 1/120.
    co = {p: F(counts[IDENTITY, p], 3000)-F(1, 120) for p in PERMS}
    assert all(F(value, 3000)-F(1, 120) == co[compose(inverse(a), b)]
               for (a,b), value in counts.items())
    assert all(counts[p,p] == 600 for p in PERMS)
    assert sum(co.values()) == 0
    assert all(co[p] == co[inverse(p)] for p in PERMS)
    values = dict()
    for p, value in co.items():
        shape = cycle_type(p)
        if shape not in values:
            values[shape] = value
        assert values[shape] == value
    return co


def check_spectrum(co, factors):
    # Convolution is matrix composition for this right-translation kernel.
    # The polynomial certificate bounds every eigenvalue without using characters.
    kernel = {p: F(int(p == IDENTITY)) for p in PERMS}
    composition = {p: [compose(inverse(q), p) for q in PERMS] for p in PERMS}
    for root in factors:
        kernel = {p: sum(kernel[q]*co[q_inv_p] for q,q_inv_p in zip(PERMS, composition[p]))
                  - root*kernel[p] for p in PERMS}
    assert all(value == 0 for value in kernel.values())
    assert sum(co[p]*sign(p) for p in PERMS) == max(factors)
    return [str(x) for x in factors]


def main():
    blocks = tuple(blocks_with_identity())
    counts = Counter(sum(sign(p) < 0 for p in block) for block in blocks)
    assert counts == {0: 6, 1: 40, 4: 10}

    def distinct_adjacencies(block):
        return len({pair for order in block for pair in zip(order, order[1:])})

    max_adjacencies = max(map(distinct_adjacencies, blocks))
    assert distinct_adjacencies(BLOCK) == max_adjacencies == 15
    assert balanced_block(BLOCK) and sum(sign(p) < 0 for p in BLOCK) == 1
    # Multiplication by the inverse of any selected row preserves position
    # balance and the absolute parity imbalance, and brings identity into the block.
    minimum_squared_sign = min(F((5-2*odd)**2, 25) for odd in counts)
    assert minimum_squared_sign == F(9, 25)
    optimal_kernel = covariance_kernel(BLOCK)
    cyclic_kernel = covariance_kernel(CYCLIC)
    certificate = check_spectrum(optimal_kernel, [F(x, 125) for x in (0, 17, 20, 33, 35, 45)])
    check_spectrum(cyclic_kernel, [0, F(1, 5), F(1, 3), 1])

    # Two position-balanced blocks with the same absolute parity imbalance
    # have the same counts by relative permutation conjugacy class. The
    # only possible off-diagonal classes are odd (2,3) and even (5).
    # Uniform reduced blocks followed by a common random output relabeling
    # is uniform over all blocks, with five preimages per full block.
    uniform_kernel = {p: (50*optimal_kernel[p]+6*cyclic_kernel[p])/56 for p in PERMS}
    uniform_certificate = check_spectrum(
        uniform_kernel, [0, F(1, 7), F(9, 35), F(2, 7), F(3, 7)]
    )
    print(json.dumps({
        "order_count": len(PERMS), "blocks_with_identity": sum(counts.values()),
        "position_balanced_blocks": len(PERMS)*sum(counts.values())//len(BLOCK),
        "reduced_block_counts_by_number_of_odd_rows": dict(sorted(counts.items())),
        "base_block": ["".join(map(str, p)) for p in BLOCK],
        "covariance_annihilator_roots": certificate,
        "normalized_minimax_risk": str(minimum_squared_sign),
        "uniform_position_balanced_worst_risk": "3/7",
        "uniform_covariance_annihilator_roots": uniform_certificate,
        "maximum_directed_adjacency_coverage": [max_adjacencies, 20],
        "ordinary_cyclic_worst_risk": "1",
        "benchmark_scores_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
