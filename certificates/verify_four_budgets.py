"""Enumerate balanced blocks and verify exact minimax certificates on all four-option budgets."""

from collections import Counter
from fractions import Fraction as F
from itertools import combinations, permutations, product
import json
from math import gcd, lcm
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"certificates")]
from four_option_budgets import ALL, BALANCES, BUDGETS, INDEX, ORDERS, complement, law
from verify_four import supports


def mask(indices):
    return sum(1 << i for i in indices)


def catalog(k):
    labels = range(4)
    pos = [mask(i for i,row in enumerate(ORDERS) if row[s] == a) for s in labels for a in labels]
    pre = [mask(i for i,row in enumerate(ORDERS) if row.index(a) < row.index(b))
           for a,b in combinations(labels,2)]
    adj = [mask(i for i,row in enumerate(ORDERS) if row.index(b)-row.index(a) == 1)
           for a,b in permutations(labels,2)]
    by_label = [[mask(b) for b in combinations([i for i,row in enumerate(ORDERS) if row[0] == a], k//4)]
                for a in labels]
    families = {c:set() for c in BALANCES}
    for chunks in product(*by_label):
        m = sum(chunks)
        if all((m & f).bit_count() == k//4 for f in pos):
            b = tuple(i for i in ALL if m & (1 << i))
            families["position"].add(b)
            if all((m & f).bit_count() == k//2 for f in pre):
                families["precedence"].add(b)
                if all((m & f).bit_count() == k//4 for f in adj):
                    families["full"].add(b)
    return families


def covariance(weights,k):
    d = lcm(*(w.denominator for w in weights.values()))
    h = [[0]*24 for _ in ALL]
    assert sum(weights.values()) == 1
    for block,w in weights.items():
        assert len(block) == len(set(block)) == k and w > 0
        for i,j in product(block, repeat=2):
            h[i][j] += int(w*d)
    assert all(24*h[i][i] == k*d for i in ALL)
    den = 24*k*k*d
    a = [[576*x-k*k*d for x in row] for row in h]
    g = gcd(den, *(x for row in a for x in row))
    a = [[x//g for x in row] for row in a]
    den //= g
    assert all(sum(row) == 0 for row in a)
    assert F(sum(a[i][i] for i in ALL),den) == F(24-k,k)
    return a,den


def multiply(a,b):
    cols = list(zip(*b))
    return [[sum(x*y for x,y in zip(row,col)) for col in cols] for row in a]


def certify_frame(kernel,ratio,catalogue,k):
    a,den = kernel
    p,q = ratio.as_integer_ratio()
    a2 = multiply(a,a)
    assert all(q*a2[i][j] == p*den*a[i][j] for i,j in product(ALL,repeat=2))
    for block in catalogue:
        for i,row in enumerate(a):
            assert 24*q*sum(row[j] for j in block) == p*den*(24*int(i in block)-k)
    rank = F(24-k,k)/ratio
    assert rank.denominator == 1
    return int(rank)


def proportional(kernel1,kernel2,scale):
    a,da = kernel1
    b,db = kernel2
    assert all(scale.denominator*db*a[i][j] == scale.numerator*da*b[i][j]
               for i,j in product(ALL,repeat=2))


def verify_fixed_law(weights, roots, witness):
    a,den = covariance(weights,8)
    multipliers = [[int(i == j) for j in ALL] for i in ALL]
    for root in roots:
        p,q = root.as_integer_ratio()
        factor = [[q*a[i][j] - p*den*int(i == j) for j in ALL] for i in ALL]
        multipliers = multiply(multipliers,factor)
    assert all(not any(row) for row in multipliers)
    d = [{"0":0,"+":1,"-":-1}[x] for x in witness]
    assert len(d) == 24 and any(d)
    p,q = roots[-1].as_integer_ratio()
    assert all(q*24*sum(x*y for x,y in zip(row,d)) == p*den*(24*d[i]-sum(d)) for i,row in enumerate(a))
    return str(roots[-1])


def check():
    families = {k:catalog(k) for k in (4,8,12)}
    _, cyclic, _, williams, other = supports()
    # Historical certificate encodes option-to-slot maps. Convert to display rows.
    inv = [INDEX[tuple(row.index(j) for j in range(4))] for row in ORDERS]
    for c,old in zip(BALANCES,(cyclic, williams|other, williams)):
        actual = law(4,c)
        assert {tuple(sorted(inv[i] for i in b)) for b in old} == set(actual)
        assert set(actual.values()) == {F(1,len(old))}
    opt = {4:(F(1,2),F(3,4),F(1)), 8:(F(1,7),F(2,11),F(2,5)), 12:(F(1,14),F(1,11),F(1,6))}
    kernels, records = {}, {}
    for k in BUDGETS:
        if k in (16,20):
            opt[k] = tuple(F((24-k)**2,k*k)*v for v in opt[24-k])
        if k == 24:
            opt[k] = (F(0),)*3
        record = {"unrestricted":str(F(24-k,23*k))}
        for c,ratio in zip(BALANCES,opt[k]):
            weights = law(k,c)
            kernels[k,c] = covariance(weights,k)
            if k <= 12:
                allowed = families[k][c]
            elif k < 24:
                allowed = {complement(b) for b in families[24-k][c]}
                proportional(kernels[k,c], kernels[24-k,c], F((24-k)**2,k*k))
            else:
                allowed = {ALL}
            assert set(weights) <= allowed
            entry = {"risk":str(ratio), "admissible_blocks":len(allowed), "sampler_support":len(weights)}
            if k in (8,12,16):
                entry["frame_rank"] = certify_frame(kernels[k,c],ratio,allowed,k)
            record[c] = entry
        records[str(k)] = record
    four = tuple(law(4,"full"))
    assert len(set().union(*four)) == sum(map(len,four)) == 24
    doublets = {tuple(sorted(a+b)) for a,b in combinations(four,2)}
    assert len(doublets) == 15 and doublets == families[8]["full"]
    assert set(law(8,"full").values()) == {F(1,15)}
    proportional(kernels[8,"full"],kernels[4,"full"],F(2,5))
    triplets = {tuple(sorted(a+b+c)) for a,b,c in combinations(four,3)}
    odd = tuple(i for i,row in enumerate(ORDERS) if sum(a>b for a,b in combinations(row,2)) % 2)
    parity = {odd,complement(odd)}
    assert len(triplets) == 20 and families[12]["full"] == triplets | parity
    assert law(12,"full") == {**{b:F(1,24) for b in triplets}, **{b:F(1,12) for b in parity}}
    roots = {"position":(F(0),F(4,27),F(1,6)), "precedence":(F(0),F(1,7),F(3,14)), "full":(F(0),F(2,5))}
    witnesses = {"position":"+000-00+-000000-+00-000+", "precedence":"+--++-00-0+000+0-000-0+0",
                 "full":"+++----++-+--+--++-+-+-+"}
    pairs = {}
    for c in BALANCES:
        sums = Counter(tuple(sorted(a+b)) for a,b in combinations(law(4,c),2) if set(a).isdisjoint(b))
        total = sum(sums.values())
        wt = {b:F(count,total) for b,count in sums.items()}
        pairs[c] = {"support":len(wt), "risk":verify_fixed_law(wt,roots[c],witnesses[c])}
    return {"balanced_budgets":records, "uniform_disjoint_four_call_pairs_at_eight":pairs}


if __name__ == "__main__":
    print(json.dumps(check(),indent=2))
