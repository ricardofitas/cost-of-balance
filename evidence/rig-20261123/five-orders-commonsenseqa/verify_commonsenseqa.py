"""Reproduce exact CommonsenseQA point results from the 120-bit correctness masks."""
from collections import Counter
from fractions import Fraction
from itertools import product
import json
from pathlib import Path

from verify_position import BLOCK, CYCLIC, PERMS, blocks_with_identity

ROOT = Path(__file__).resolve().parent
INDEX = {p: i for i, p in enumerate(PERMS)}


def relabel_options(block, perm):
    return tuple(sorted(INDEX[tuple(perm[x] for x in row)] for row in block))


def main():
    data = json.loads((ROOT / "correctness-masks.json").read_text())
    expected = json.loads((ROOT / "results.json").read_text())
    frozen = json.loads((ROOT / "frozen.json").read_text())
    assert data["study_sha256"] == expected["study"]
    assert [row[0] for row in data["rows"]] == [row["id"] for row in frozen["rows"]]
    n = len(data["rows"])
    assert n == expected["questions"]
    position = Counter(relabel_options([tuple(row[x] for x in slots) for row in BLOCK], options)
                       for slots, options in product(PERMS, repeat=2))
    uniform = {relabel_options(block, options) for block, options in product(blocks_with_identity(), PERMS)}
    cyc = {relabel_options(CYCLIC, options) for options in PERMS}
    solution = json.loads((ROOT / "solutions.json").read_text())
    adjacency = {}
    for rep, weight in zip(solution["representatives"], solution["primal_weight_numerators"], strict=True):
        rows = [tuple(map(int, row)) for row in rep]
        blocks = {relabel_options(rows, options) for options in PERMS}
        assert not blocks.intersection(adjacency)
        adjacency.update(dict.fromkeys(blocks, Fraction(weight, 100_000*len(blocks))))
    laws = {"position-minimax": dict(position), "ordinary-cyclic": dict.fromkeys(cyc, 1),
            "uniform-position": dict.fromkeys(uniform, 1), "maximum-adjacency": adjacency}
    count = {name: Counter() for name in laws}
    accuracy = [0, 0]
    for _, left, right in data["rows"]:
        a, b = int(left, 16), int(right, 16)
        assert 0 <= a < 2**120 and 0 <= b < 2**120
        accuracy[0] += a.bit_count()
        accuracy[1] += b.bit_count()
        contrast = [((a >> order) & 1) - ((b >> order) & 1) for order in range(120)]
        total = sum(contrast)
        for name, support in laws.items():
            for block in support:
                count[name][block] += (24*sum(contrast[i] for i in block) - total)**2
    errors = {}
    for name, law in laws.items():
        error = sum(Fraction(count[name][block] * weight) for block, weight in law.items())
        error /= 14400 * n*n * sum(law.values())
        assert error == Fraction(expected["designs"][name]["exact_mse"])
        errors[name] = str(error)
    reference = Fraction(accuracy[0]-accuracy[1], 120*n)
    assert float(reference) == expected["reference_gap"]
    print(json.dumps({"exact_reference_gap": str(reference), "exact_mse": errors}, indent=2))


if __name__ == "__main__":
    main()
