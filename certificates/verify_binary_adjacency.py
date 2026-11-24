"""Independent, integer-only lower bound for maximal realized adjacency."""
from collections import Counter
from fractions import Fraction
from itertools import permutations, product
import json

ORDERS = tuple(permutations("01234"))
IDENTITY = ORDERS[0]
WITNESS = (
    "-++-+--+-+-++-++++---+---++-+--+----+-+-+-+++++--+-+-+-+++++-----"
    "+-+-+-++-----+-+-+-++++-++-+--+++-+++----+--+-+-++-+--+"
)


def parity(order):
    current = set(order)
    cycles = 0
    while current:
        cycles += 1
        value = min(current)
        while value in current:
            current.remove(value)
            value = order[int(value)]
    return (-1) ** (5 - cycles)


def canonical(block):
    # For any selected row r, apply r^{-1} to option identities to make it the
    # identity row. The lexicographic minimum identifies its full option orbit.
    representatives = [
        tuple(sorted(tuple(str(row.index(label)) for label in order) for order in block))
        for row in block
    ]
    return min(representatives)


def get_maximum_coverage():
    by_first = {
        letter: [order for order in ORDERS if order[0] == letter
                 and all(a != b for a, b in zip(order, IDENTITY))]
        for letter in IDENTITY[1:]
    }
    coverage = Counter()
    maximum_orbits = {}
    for rows in product(*by_first.values()):
        block = (IDENTITY, *rows)
        if any(set(column) != set(IDENTITY) for column in zip(*block)):
            continue
        directed = len({pair for row in block for pair in zip(row, row[1:])})
        coverage[directed] += 1
        if directed == 15:
            maximum_orbits[canonical(block)] = abs(sum(map(parity, block)))
    return coverage, maximum_orbits


def risk(block, scores):
    contrasts = [
        Fraction(sum(scores[tuple(labels[int(item)] for item in order)] for order in block), 5)
        for labels in ORDERS
    ]
    mean = Fraction(sum(scores.values()), len(scores))
    population = sum((score - mean) ** 2 for score in scores.values()) / len(scores)
    return (sum((value - mean) ** 2 for value in contrasts) / len(contrasts)) / population


def main():
    assert len(WITNESS) == len(ORDERS)
    assert set(WITNESS) == {"+", "-"}
    sign_witness = dict(zip(ORDERS, (1 if value == "+" else -1 for value in WITNESS), strict=True))
    coverage, orbits = get_maximum_coverage()
    assert coverage == {5: 1, 10: 1, 11: 20, 12: 5, 14: 20, 15: 9}
    assert sorted(orbits.values()) == [3, 5, 5, 5, 5]
    base_risk, = [risk(block, sign_witness) for block, odd in orbits.items() if odd == 3]
    alternatives = [risk(block, sign_witness) for block, odd in orbits.items() if odd == 5]
    alternative_minimum = min(alternatives)
    assert base_risk == Fraction(11, 25)
    # Any symmetrized design with maximal coverage puts weight w on the four
    # parity-homogeneous orbits. Balance the two affine lower bounds in w.
    intersection = (base_risk - Fraction(9, 25)) / (Fraction(16, 25) + base_risk - alternative_minimum)
    assert 0 <= intersection <= 1
    lower_bound = Fraction(9, 25) + Fraction(16, 25) * intersection
    assert lower_bound > Fraction(9, 25)
    print(json.dumps({
        "reduced_coverage_histogram": dict(sorted(coverage.items())),
        "maximal_coverage_option_orbits": len(orbits),
        "maximal_coverage_orbits_by_absolute_parity_imbalance": dict(Counter(orbits.values())),
        "binary_witness_risks": [str(value) for value in [base_risk, *sorted(alternatives)]],
        "worst_risk_lower_bound_for_maximal_coverage": str(lower_bound),
    }, indent=2))


if __name__ == "__main__":
    main()
