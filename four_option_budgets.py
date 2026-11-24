"""Certified order samplers for paired four-option evaluations at fixed budgets."""

import argparse
from collections import defaultdict
from fractions import Fraction
from functools import cache
from itertools import permutations, product
import json
from math import lcm
from random import Random

from design_templates import FOUR_OPTION_TEMPLATES


ORDERS = tuple(permutations(range(4)))  # A row lists semantic labels in display order.
INDEX = {row: i for i, row in enumerate(ORDERS)}
ALL = tuple(range(len(ORDERS)))
BALANCES = ("position", "precedence", "full")
BUDGETS = (4, 8, 12, 16, 20, 24)


@cache
def templates():
    return FOUR_OPTION_TEMPLATES


def check_balanced(budget, balance):
    if budget not in BUDGETS or balance not in BALANCES:
        raise ValueError("Choose four to 24 calls in multiples of four, and position, precedence or full balance.")


def transform(indices, labels, slots):
    return tuple(sorted(INDEX[tuple(labels[ORDERS[i][s]] for s in slots)] for i in indices))


def complement(block):
    return tuple(i for i in ALL if i not in block)


@cache
def law(budget, balance):
    """Map each eligible support point of the certified sampler to its exact probability."""
    check_balanced(budget, balance)
    if budget == 24:
        return {ALL: Fraction(1)}
    if budget > 12:
        return {complement(b): w for b, w in law(24-budget, balance).items()}
    result = defaultdict(Fraction)
    slots = ORDERS if balance == "position" else (ORDERS[0], ORDERS[-1])
    for atom in templates()[str(budget)][balance]:
        orbit = {transform(atom["orders"], g, h) for g, h in product(ORDERS, slots)}
        mass = Fraction(atom["weight"]) / len(orbit)
        for block in orbit:
            result[block] += mass
    return dict(sorted(result.items()))


def sample_indices(budget, balance, rng):
    if balance == "unrestricted":
        if not 1 <= budget <= 24:
            raise ValueError("An unrestricted block uses between one and 24 distinct orders.")
        return tuple(sorted(rng.sample(ALL, budget)))
    check_balanced(budget, balance)
    if budget == 24:
        return ALL
    if budget > 12:
        return complement(sample_indices(24-budget, balance, rng))
    atoms = templates()[str(budget)][balance]
    weights = [Fraction(atom["weight"]) for atom in atoms]
    denominator = lcm(*(w.denominator for w in weights))
    (atom,) = rng.sample(atoms, 1, counts=[int(w*denominator) for w in weights])
    labels = rng.choice(ORDERS)
    slots = rng.choice(ORDERS if balance == "position" else (ORDERS[0], ORDERS[-1]))
    return transform(atom["orders"], labels, slots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("budget", type=int)
    parser.add_argument("balance", choices=("unrestricted", *BALANCES))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    block = sample_indices(args.budget, args.balance, Random(args.seed))
    print(json.dumps([ORDERS[i] for i in block]))
