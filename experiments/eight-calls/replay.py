"""Integrate order-sampler risks exactly and replay paired, stratified item intervals."""

import argparse
from collections import Counter
from fractions import Fraction as F
from functools import cache
import gzip
from hashlib import sha256
from itertools import combinations
import json
from math import gcd, lcm
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from four_option_budgets import ALL, BALANCES, INDEX, ORDERS, law


PANELS = ("hellaswag-3b", "mmlu-3b", "logiqa-3b", "hellaswag-1.5b-2b")
REPLICATES = 50_000
SEED = 2026112308


def disjoint_pairs(balance):
    """Condition two draws from the four-order sampler on disjointness."""
    counts = Counter()
    for (a, wa), (b, wb) in combinations(law(4, balance).items(), 2):
        if set(a).isdisjoint(b):
            counts[tuple(sorted(a+b))] += wa*wb
    total = sum(counts.values())
    return {b: w/total for b, w in counts.items()}


def dihedral_blocks():
    counts = Counter()
    for labels in ORDERS:
        block = [INDEX[row[j:] + row[:j]] for row in (labels, labels[::-1]) for j in range(4)]
        counts[tuple(sorted(block))] += 1
    assert len(counts) == 3
    return {b: F(c, 24) for b, c in counts.items()}


def mask(indices):
    return sum(1 << i for i in indices)


def discrete_sampler(k, weights):
    den = lcm(*(w.denominator for w in weights.values()))
    assert sum(weights.values()) == 1
    return k, den, [(mask(b), int(w*den)) for b, w in weights.items()]


DESIGNS = {f"{k}-{c}": discrete_sampler(k, law(k, c)) for k in (4, 8) for c in BALANCES}
DESIGNS.update({f"8-pair-{c}": discrete_sampler(8, disjoint_pairs(c)) for c in BALANCES})
DESIGNS["8-rotation-reversal"] = discrete_sampler(8, dihedral_blocks())


def primitive(columns):
    """Integer orthogonal basis of a feature span via exact Gram--Schmidt."""
    basis = []
    for column in columns:
        u = column
        for v, norm2 in basis:
            dot = sum(x*y for x, y in zip(u, v))
            if dot:
                u = [norm2*x - dot*y for x, y in zip(u, v)]
                scale = gcd(*u)
                if scale:
                    u = [x//scale for x in u]
        if any(u):
            basis.append((u, sum(x*x for x in u)))
    return basis


POSITION = [[1]*24] + [[int(row[s] == a) for row in ORDERS] for a in range(4) for s in range(4)]
PRECEDENCE = [[int(row.index(a) < row.index(b)) for row in ORDERS] for a,b in combinations(range(4), 2)]
BASIS = primitive(POSITION + PRECEDENCE)
assert len(primitive(POSITION)) == 10 and len(BASIS) == 13
WILLIAMS = tuple(mask(b) for b in law(4, "full"))


@cache
def item_risks(a, b):
    assert 0 <= a < 2**24 and 0 <= b < 2**24 and not a & b
    total = a.bit_count() - b.bit_count()
    squares = a.bit_count() + b.bit_count()
    variance = F(24*squares - total*total, 576)
    risk = {f"{k}-unrestricted": variance*F(24-k, 23*k) for k in (4, 8)}
    for name, (k, den, support) in DESIGNS.items():
        unit = 24//k
        error = sum(w*(unit*((a&m).bit_count() - (b&m).bit_count()) - total)**2 for m,w in support)
        risk[name] = F(error, 576*den)

    # Independent eight-call computation from feature projections and six block means.
    d = [((a >> i) & 1) - ((b >> i) & 1) for i in ALL]
    explained = [F(sum(x*y for x,y in zip(d,v))**2, norm2) for v,norm2 in BASIS]
    for name, dimension, ratio in (("position",10,F(1,7)), ("precedence",13,F(2,11))):
        energy = (squares - sum(explained[:dimension])) / 24
        assert risk["8-"+name] == ratio*energy
    energy_full = sum((6*((a&m).bit_count()-(b&m).bit_count()) - total)**2 for m in WILLIAMS)
    assert risk["8-full"] == F(2*energy_full, 5*6*576)
    assert risk["8-full"] == F(2,5)*risk["4-full"] == risk["8-pair-full"]
    return risk


def relative(full, comparator):
    return 100*(1-full/comparator)


def analyze_point(groups):
    risk_sums = Counter()
    total, gap_numerator = 0, 0
    for profiles in groups.values():
        for pa, pb, count in profiles:
            a, b = int(pa,16), int(pb,16)
            assert count > 0
            for key, val in item_risks(a,b).items():
                risk_sums[key] += count*val
            total += count
            gap_numerator += count*(a.bit_count()-b.bit_count())
    mse = {key: v/total**2 for key,v in sorted(risk_sums.items())}
    gain = {k: relative(mse[f"{k}-full"], mse[f"{k}-precedence"]) for k in (4,8)}
    energy_share = (mse["8-full"]/F(2,5))/(mse["8-precedence"]/F(2,11))
    assert (energy_share < F(5,11)) == (gain[8] > 0)
    return {"items":total, "strata":len(groups), "reference_gap":str(F(gap_numerator, 24*total)),
            "mse":{key:str(v) for key,v in mse.items()}, "full_gain_percent":{str(k):float(x) for k,x in gain.items()},
            "gain_change_pp":float(gain[8]-gain[4]), "retained_energy_share":str(energy_share),
            "four_to_eight_reduction_percent":{c:float(relative(mse["8-"+c], mse["4-"+c]))
                                                 for c in ("unrestricted", *BALANCES)},
            "optimized_vs_disjoint_pairs_percent":{c:float(relative(mse["8-"+c], mse["8-pair-"+c]))
                                                     for c in BALANCES}}


def bootstrap(groups, seed):
    import numpy as np
    names = sorted(item_risks(0,0))
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.zeros((REPLICATES,len(names)))
    for key, profiles in sorted(groups.items()):
        counts = np.array([row[2] for row in profiles], dtype=int)
        risks = np.array([[float(item_risks(int(a,16),int(b,16))[name]) for name in names] for a,b,_ in profiles])
        n = counts.sum()
        for start in range(0, REPLICATES, 128):
            stop = min(REPLICATES,start+128)
            freq = rng.multinomial(n, counts/n, size=stop-start)
            samples[start:stop] += freq @ risks
    columns = {name:samples[:,i] for i,name in enumerate(names)}
    gains = [relative(columns[f"{k}-full"], columns[f"{k}-precedence"]) for k in (4,8)]
    def interval(values):
        return np.percentile(values, [2.5,97.5]).tolist()
    return {"seed":seed, "replicates":REPLICATES, "numpy":np.__version__,
            "full_gain_percent":{"4":interval(gains[0]), "8":interval(gains[1])},
            "gain_change_pp":interval(gains[1]-gains[0]),
            "optimized_vs_disjoint_pairs_percent":{c:interval(relative(columns["8-"+c],columns["8-pair-"+c]))
                                                  for c in BALANCES}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", action="store_true", help="also replay all 50,000 item resamples (NumPy)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="write the result to a JSON file")
    mode.add_argument("--check", action="store_true", help="compare with the published result")
    args = parser.parse_args()
    path = HERE/"profiles.json.gz"
    assert sha256(path.read_bytes()).hexdigest() == json.loads((HERE/"source-receipts.json").read_text())["profile_sha256"]
    data = json.loads(gzip.decompress(path.read_bytes()))
    result = {}
    for i,panel in enumerate(PANELS):
        result[panel] = analyze_point(data[panel])
        if args.intervals:
            result[panel]["bootstrap"] = bootstrap(data[panel], SEED+i)
    payload = json.dumps(result, indent=2) + "\n"
    if args.check:
        published = json.loads((HERE/"results.json").read_text())
        if not args.intervals:
            for panel in published.values():
                del panel["bootstrap"]
        assert result == published
        print("All selected panel results matched.")
    elif args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
