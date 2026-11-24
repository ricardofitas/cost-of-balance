"""Check the four-order appendix's finite-support and minimax certificates."""

from fractions import Fraction
from itertools import combinations, permutations, product


ORDERS = tuple(permutations(range(4)))
INDEX = {order: index for index, order in enumerate(ORDERS)}
PAIRS = tuple(combinations(range(4), 2))


def orbit(block, slot_maps):
    return {
        tuple(sorted(INDEX[tuple(slot[ORDERS[index][j]] for j in option)]
                     for index in block))
        for option, slot in product(ORDERS, slot_maps)
    }


def supports():
    balanced = {
        block for block in combinations(range(24), 4)
        if all(len({ORDERS[k][j] for k in block}) == 4 for j in range(4))
    }
    precedence = {
        block for block in balanced
        if all(sum(ORDERS[k][i] < ORDERS[k][j] for k in block) == 2
               for i, j in PAIRS)
    }
    adjacent = {
        block for block in precedence
        if all(sum(ORDERS[k][j] - ORDERS[k][i] == 1 for k in block) == 1
               for i, j in permutations(range(4), 2))
    }
    williams_seed = tuple(INDEX[tuple(map(int, row))]
                          for row in ("0123", "1302", "2031", "3210"))
    assert adjacent == orbit(williams_seed, ORDERS[:1])
    cyclic = orbit(williams_seed, ORDERS)
    klein = balanced - cyclic
    assert klein == orbit(min(klein), ORDERS)
    other = precedence - adjacent - klein
    assert other == orbit(min(other), ORDERS[:1])
    assert klein == orbit(min(klein), ORDERS[:1])
    assert tuple(map(len, (balanced, precedence, cyclic, klein, adjacent, other))) == (
        24, 18, 18, 6, 6, 6
    )
    return balanced, cyclic, klein, adjacent, other


def centered_gram(blocks):
    gram = [[sum(i in block and j in block for block in blocks)
             for j in range(24)] for i in range(24)]
    row_sums = {sum(row) for row in gram}
    assert len(row_sums) == 1
    row_sum = row_sums.pop()
    return [[24 * value - row_sum for value in row] for row in gram]


def spectral_bound(blocks, roots):
    gram = centered_gram(blocks)
    polynomial = [[int(i == j) for j in range(24)] for i in range(24)]
    for root in roots:
        assert any(any(row) for row in polynomial)
        polynomial = [
            [sum(row[k] * gram[k][j] for k in range(24)) - 24 * root * row[j]
             for j in range(24)]
            for row in polynomial
        ]
    assert all(not any(row) for row in polynomial)
    return Fraction(24 * roots[-1], 16 * len(blocks))


def witness_risk(blocks, positive, negative):
    assert len(positive) == len(negative) and set(positive).isdisjoint(negative)
    profile = [int(k in positive) - int(k in negative) for k in range(24)]
    error = sum(sum(profile[k] for k in block) ** 2 for block in blocks)
    return Fraction(24 * error, 16 * len(blocks) * sum(x * x for x in profile))


def main():
    balanced, cyclic, klein, williams, other = supports()
    for name, blocks, roots, expected in [
        ("all balanced", balanced, (0, 4, 10), Fraction(5, 8)),
        ("position minimax", cyclic, (0, 4, 6), Fraction(1, 2)),
        ("precedence minimax", williams | other, (0, 2, 4, 6), Fraction(3, 4)),
        ("Williams", williams, (0, 4), Fraction(1)),
    ]:
        bound = spectral_bound(blocks, roots)
        assert bound == expected
        print(f"{name}: {len(blocks)} blocks, normalized risk {bound}")

    position = ((0, 7, 16, 23), (3, 11, 12, 20))
    precedence = ((0, 2, 7, 10, 13, 16, 21, 23), (3, 5, 9, 11, 12, 14, 18, 20))
    assert witness_risk(cyclic, *position) == Fraction(1, 2)
    assert witness_risk(klein, *position) == 1
    assert witness_risk(williams, *precedence) == Fraction(3, 4)
    assert witness_risk(other, *precedence) == Fraction(3, 4)
    assert witness_risk(klein, *precedence) == 1
    assert witness_risk(williams, min(williams), max(williams)) == 1
    print("All ternary witnesses verified.")


if __name__ == "__main__":
    main()
