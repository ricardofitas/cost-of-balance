"""Verify support connectivity and correctness lower bounds for adjacency constraints."""

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, permutations
import json
from math import factorial
from pathlib import Path
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
CATALOGUE = "https://users.monash.edu.au/~iwanless/data/RCLS/"


def compose(a, b):
    return tuple(a[x] for x in b)


def coverage(rows):
    return len({pair for row in rows for pair in zip(row, row[1:])})


def is_latin(rows, n):
    labels = set(range(n))
    return (len(rows) == n and all(len(r) == n and set(r) == labels for r in rows)
            and all(set(col) == labels for col in zip(*rows)))


def reduced_latin(n):
    identity = tuple(range(n))
    candidates = [[] for _ in identity]
    for row in permutations(identity):
        mask = sum(1 << (n*i+x) for i, x in enumerate(row))
        candidates[row[0]].append((row, mask))

    def extend(rows, used):
        if len(rows) == n:
            yield rows
            return
        for row, mask in candidates[len(rows)]:
            if not mask & used:
                yield from extend((*rows, row), used | mask)

    return extend((identity,), sum(1 << (n*i+i) for i in identity))


def group_size(n, generators):
    reached = {tuple(range(n))}
    pending = list(reached)
    while pending:
        a = pending.pop()
        for b in generators:
            v = compose(a, b)
            if v not in reached:
                reached.add(v)
                pending.append(v)
    return len(reached)


def mean_swaps(block):
    n = len(block)
    totals = [sum(a[i] == b[j] and a[j] == b[i] for i,j in combinations(range(n),2))
              for a,b in combinations(block,2)]
    return Fraction(sum(totals),len(totals))


def small_supports():
    checked = {}
    for n, expected in [(4, (4,12,1,4)), (5, (56,15,9,120)), (6, (9408,30,2,24))]:
        counts = Counter()
        best, blocks = -1, []
        for block in reduced_latin(n):
            value = coverage(block)
            counts[value] += 1
            if value > best:
                best, blocks = value, []
            if value == best:
                blocks.append(block)
        generators = {row for b in blocks for row in b}
        subgroup = group_size(n, generators)
        assert (sum(counts.values()), best, len(blocks), subgroup) == expected
        if n in (4, 6):
            assert all(row[i] + row[n-1-i] == n-1 for row in generators for i in range(n))
        if n == 4:
            assert {mean_swaps(b) for b in reduced_latin(4)} == {Fraction(2,3), Fraction(2)}
        if n == 6:
            assert {mean_swaps(b) for b in blocks} == {Fraction(3,5)}
        checked[n] = {"reduced_latin": sum(counts.values()), "maximum_coverage": best,
                      "reduced_maximum_blocks":len(blocks), "subgroup_order":subgroup,
                      "components":factorial(n)//subgroup}
    return checked


def five_correctness():
    n = 5
    orders = tuple(permutations(range(n)))
    index = {p:i for i,p in enumerate(orders)}
    pins = json.loads((HERE/"solutions.json").read_text())
    profiles = json.loads((HERE/"five-correctness-witnesses.json").read_text())
    weights = list(map(Fraction, profiles["weights"]))
    assert min(weights) > 0 and sum(weights) == 1
    orbit_blocks = []
    for rep in pins["representatives"]:
        rows = [tuple(map(int, r)) for r in rep]
        orbit_blocks.append({tuple(sorted(index[compose(g,r)] for r in rows)) for g in orders})
    # The separate upper-bound verifier enumerates these orbits too.
    all_blocks = set().union(*orbit_blocks)
    assert len(all_blocks) == sum(map(len,orbit_blocks)) == 216
    assert {b for b in all_blocks if 0 in b} == {
        tuple(sorted(index[r] for r in b)) for b in reduced_latin(n) if coverage(b) == 15}
    risks = []
    for signed in profiles["profiles"]:
        assert len(signed) == len(orders) and set(signed) <= set("-0+")
        z = ["-0+".index(x)-1 for x in signed]
        assert sum(z) == 0
        norm = sum(v*v for v in z)
        risks.append([Fraction(len(orders)*sum(sum(z[i] for i in b)**2 for b in orbit),
                               n*n*len(orbit)*norm) for orbit in orbit_blocks])
    lower = [sum(w*v for w,v in zip(weights, col, strict=True)) for col in zip(*risks)]
    assert lower == [Fraction(7,15)]*5
    return {"weights":list(map(str,weights)), "risk_by_witness_and_orbit":[list(map(str,r)) for r in risks],
            "correctness_lower":str(min(lower))}


def check_connected(n, pins):
    blocks = [[tuple(map(int, r)) for r in b] for b in pins["connected_templates"]]
    assert all(is_latin(b,n) and coverage(b) == n*(n-1) and tuple(range(n)) in b for b in blocks)
    generators = [r for b in blocks for r in b]
    # A 2-transitive group containing a transposition contains every transposition.
    reached = {(0,1)}
    pending = list(reached)
    while pending:
        i,j = pending.pop()
        for row in generators:
            image = (row[i],row[j])
            if image not in reached:
                reached.add(image)
                pending.append(image)
    assert len(reached) == n*(n-1)
    i,j,exponent = pins["transposition_product"]
    identity = tuple(range(n))
    word, value = compose(generators[i],generators[j]), identity
    for _ in range(exponent):
        value = compose(value, word)
    assert sum(a != b for a,b in zip(value,identity)) == 2


def larger_case(n, pins, catalogue_dir):
    check_connected(n,pins)
    name = f"catalogue.redRCLS{n}"
    if catalogue_dir is None:
        with urlopen(CATALOGUE+name, timeout=30) as response:
            raw = response.read()
    else:
        raw = (catalogue_dir/name).read_bytes()
    assert sha256(raw).hexdigest() == pins["catalogue_sha256"]
    fields = raw.decode("ascii").split()
    count = pins["reduced_count"]
    assert list(map(int,fields[:3])) == [n,n,count] and len(fields)-3 == n*count
    rows = [tuple(ord(c)-ord("a") for c in s) for s in fields[3:]]
    blocks = [tuple(rows[i:i+n]) for i in range(0,len(rows),n)]
    assert len(set(blocks)) == count
    weights, contrasts = [], []
    for w in pins["witnesses"]:
        slots = tuple(w["slots"])
        symmetric = w["kind"] == "cycle"
        assert w["kind"] in ("cycle","triangle") and len(slots) == (4 if symmetric else 3)
        assert len(set(slots)) == len(slots) and set(slots) <= set(range(n))
        h = {}
        for j,(a,b) in enumerate(zip(slots,slots[1:]+slots[:1])):
            sign = (-1)**j if symmetric else 1
            h[a,b], h[b,a] = sign, sign if symmetric else -sign
        assert sum(h.values()) == 0
        weights.append(Fraction(w["weight"]))
        contrasts.append(h)
    assert min(weights) > 0 and sum(weights) == 1
    lower = Fraction(1)
    for b in blocks:
        assert is_latin(b,n) and coverage(b) == n*(n-1)
        assert b[0] == tuple(range(n)) and tuple(r[0] for r in b) == b[0]
        positions = [tuple(r.index(i) for i in range(n)) for r in b]
        risk = Fraction()
        for h, weight in zip(contrasts,weights):
            squared = sum(sum(h.get((r[i],r[j]),0) for r in positions)**2
                          for i,j in permutations(range(n),2))
            risk += weight * Fraction(squared, n*n*len(h))
        lower = min(lower,risk)
    expected = {8:Fraction(69,280), 9:Fraction(43,243), 10:Fraction(12427,99080)}
    assert lower == expected[n] and lower > Fraction(n-2,n*n-3*n+1)
    return {"catalogue_reduced_blocks":count, "correctness_lower":str(lower),
            "connected_upper_strictly_below_one":True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue-dir", type=Path, help="use pinned catalogue files already on disk")
    parser.add_argument("--small", action="store_true", help="check four to six without a catalogue")
    args = parser.parse_args()
    result = {"small_supports":small_supports(), "five_options":five_correctness()}
    if not args.small:
        pinned = json.loads((HERE/"adjacency-witnesses.json").read_text())
        result["larger_adjacency"] = {k:larger_case(int(k),v,args.catalogue_dir) for k,v in pinned.items()}
    print(json.dumps(result, indent=2))
