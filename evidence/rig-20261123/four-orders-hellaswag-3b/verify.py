"""Independently enumerate exact paired MSE from the scored correctness masks."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations
import json
from math import isclose
from pathlib import Path


def catalog(orders):
    # Orders here are displayed option indices, deliberately not the worker's inverse maps.
    out = {k: [] for k in ("ordinary-cyclic", "all-balanced", "position-minimax", "precedence-minimax", "williams")}
    for idx in combinations(range(24), 4):
        rows = [orders[i] for i in idx]
        if any(len(set(col)) != 4 for col in zip(*rows, strict=True)):
            continue
        out["all-balanced"].append(idx)
        rotate = {rows[0][k:] + rows[0][:k] for k in range(4)}
        if set(rows) == rotate:
            out["ordinary-cyclic"].append(idx)
        # A double-transposition block has every nonidentity relative map of order two.
        involutions = all(all(row[rows[0].index(row[pos])] == rows[0][pos] for pos in range(4))
                          for row in rows[1:])
        if not involutions:
            out["position-minimax"].append(idx)
        if any(sum(row.index(a) < row.index(b) for row in rows) != 2 for a, b in combinations(range(4), 2)):
            continue
        if not involutions:
            out["precedence-minimax"].append(idx)
        adjacencies = [edge for row in rows for edge in zip(row, row[1:])]
        if len(set(adjacencies)) == 12:
            out["williams"].append(idx)
    assert [len(blocks) for blocks in out.values()] == [6, 24, 18, 12, 6]
    return out


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "correctness-masks.json").read_text())
    expected = json.loads((root / "results.json").read_text())
    assert data["roster_sha256"] == expected["manifest_sha256"]
    rows = data["rows"]
    n = len(rows)
    assert n == expected["panel"]["items"]
    designs = catalog(list(permutations(range(4))))
    totals = dict.fromkeys(designs, 0)
    origins = Counter(data["origin_by_item"])
    assert origins == expected["cohort_origins"]
    origin_totals = {name: dict.fromkeys(designs, 0) for name in origins}
    accuracy = [0, 0]
    for (_, left, right), origin in zip(rows, data["origin_by_item"], strict=True):
        a, b = int(left, 16), int(right, 16)
        assert 0 <= a < (1 << 24) and 0 <= b < (1 << 24)
        accuracy[0] += a.bit_count()
        accuracy[1] += b.bit_count()
        diff = [((a >> i) & 1) - ((b >> i) & 1) for i in range(24)]
        all_orders = sum(diff)
        for name, blocks in designs.items():
            value = sum((6 * sum(diff[i] for i in block) - all_orders)**2 for block in blocks)
            totals[name] += value
            origin_totals[origin][name] += value
    fractions = {name: Fraction(totals[name], 576 * len(blocks) * n**2) for name, blocks in designs.items()}
    for name, mse in fractions.items():
        assert isclose(float(mse), expected["panel"]["designs"][name]["mse"], rel_tol=1e-12)
    gap = Fraction(accuracy[0] - accuracy[1], 24 * n)
    assert isclose(float(gap), expected["panel"]["reference_gap"], rel_tol=1e-12, abs_tol=1e-16)
    group_mse = {origin: {name: Fraction(value, 576 * len(designs[name]) * origins[origin]**2)
                          for name, value in sums.items()} for origin, sums in origin_totals.items()}
    p, w = "precedence-minimax", "williams"
    sensitivity = {origin: {"items": origins[origin], "relative_williams_gain_vs_precedence": (
        float(1 - values[w] / values[p]) if values[p] else None)}
                   for origin, values in group_mse.items()}
    print(json.dumps({"full_order_accuracy_gap": str(gap),
                      "exact_mse": {k: str(v) for k, v in fractions.items()},
                      "descriptive_origin_sensitivity": sensitivity}, indent=2))


if __name__ == "__main__":
    main()
