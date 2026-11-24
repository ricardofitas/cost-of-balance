"""Independent exact replay via Frobenius coefficients and pair-position frame matrices."""

from collections import Counter
from fractions import Fraction
from functools import cache
from itertools import combinations, permutations
from math import comb, lcm
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent


def shapes(total, cap=None):
    if total == 0:
        yield ()
    for k in range(1, min(total, cap or total) + 1):
        for tail in shapes(total-k,k):
            yield (k,*tail)


@cache
def frobenius(part, cycle_lengths):
    # Transpose shortens the determinant. The conjugate character differs by sign.
    if part[0] < len(part):
        conjugate = tuple(sum(v > i for v in part) for i in range(part[0]))
        return (-1)**(sum(part)-len(cycle_lengths))*frobenius(conjugate,cycle_lengths)
    m = len(part)
    target = tuple(v+m-1-i for i,v in enumerate(part))
    monomials = {(0,)*m:1}
    for k in cycle_lengths:
        nxt = Counter()
        for powers, count in monomials.items():
            for j in range(m):
                if powers[j]+k <= target[j]:
                    item = list(powers)
                    item[j] += k
                    nxt[tuple(item)] += count
        monomials = nxt
    result = 0
    for p in permutations(range(m-1,-1,-1)):
        sign = (-1)**sum(a<b for a,b in combinations(p,2))
        result += sign*monomials.get(tuple(a-b for a,b in zip(target,p)),0)
    return result


def cycles_from_rows(a,b):
    # Count cycles of b o a^{-1} on option labels.
    move = dict(zip(a,b,strict=True))
    sizes = []
    while move:
        p = next(iter(move))
        size = 0
        while p in move:
            p = move.pop(p)
            size += 1
        sizes.append(size)
    return tuple(sorted(sizes))


def replay_spectra(n, atoms):
    frequencies = Counter()
    for atom in atoms:
        rows = atom["rows"]
        for first in rows:
            for second in rows:
                frequencies[cycles_from_rows(first,second)] += Fraction(atom["weight"]) / n**2
    assert sum(frequencies.values()) == 1
    output = {}
    for part in shapes(n):
        dimension = frobenius(part,(1,)*n)
        value = sum(w*frobenius(part,c) for c,w in frequencies.items())/dimension
        if part != (n,):
            output[",".join(map(str,part))] = str(value)
    swaps = sum(w*c.count(2) for c,w in frequencies.items()) * Fraction(n,n-1)
    s, a = n*(n-3)//2, (n-1)*(n-2)//2
    assert Fraction(output[f"{n-2},2"]) == Fraction(1,n)+Fraction(n-1,n)*swaps/s
    assert Fraction(output[f"{n-2},1,1"]) == Fraction(1,n)+Fraction(n-1,n)*(1-swaps)/a
    return output


def check_frame(n):
    dyads = list(permutations(range(n), 2))
    index = {p:i for i,p in enumerate(dyads)}
    # Integer matrix equal to scale times the candidate orthogonal projector.
    symmetric_dim = n*(n-3)//2
    skew_dim = (n-1)*(n-2)//2
    sym_coeff = Fraction(symmetric_dim,3*comb(n,4)*8)
    skew_coeff = Fraction(skew_dim,comb(n,3)*6)
    scale = lcm(sym_coeff.denominator,skew_coeff.denominator)
    matrix = [[0 for _ in dyads] for _ in dyads]
    for length, coeff in [(4,sym_coeff),(3,skew_coeff)]:
        for subset in combinations(range(n), length):
            for tail in permutations(subset[1:]):
                if tail[0] > tail[-1]:
                    continue
                tour=(subset[0],*tail)
                edges = {}
                for j,(a,b) in enumerate(zip(tour,tour[1:]+tour[:1])):
                    weight = (-1)**j if length == 4 else 1
                    edges[index[a,b]] = weight
                    edges[index[b,a]] = weight if length == 4 else -weight
                for i,u in edges.items():
                    for j,v in edges.items():
                        matrix[i][j] += int(scale*coeff) * u*v
    assert sum(row[i] for i,row in enumerate(matrix)) == scale*(symmetric_dim+skew_dim)
    for i,row in enumerate(matrix):
        assert all(sum(row[index[a,b]] for b in range(n) if b != a) == 0 for a in range(n))
        assert all(sum(row[index[a,b]] for a in range(n) if a != b) == 0 for b in range(n))
        for j,other in enumerate(matrix):
            assert sum(x*y for x,y in zip(row, other)) == scale * row[j]
    return symmetric_dim+skew_dim


if __name__ == "__main__":
    verified = json.loads((ROOT/"position-family-certificate.json").read_text())
    templates = json.loads((ROOT/"position-templates.json").read_text())
    checked_frames = {str(n):check_frame(n) for n in range(4,11)}
    for k, atoms in templates.items():
        computed = replay_spectra(int(k),atoms)
        assert computed == verified[k]["nontrivial_eigenvalues_by_partition"]
        assert max(map(Fraction, computed.values())) == Fraction(verified[k]["minimax"])
    print(json.dumps({"exact_frame_dimensions":checked_frames,
                      "independent_full_spectra_matched":list(templates)},indent=2))
