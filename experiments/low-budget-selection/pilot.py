"""Prespecified sparse eight-order pilots and exact expectation certificates."""
from collections import defaultdict
from fractions import Fraction as F
from functools import cache
from hashlib import sha256
from itertools import combinations
from math import lcm
from pathlib import Path
from random import Random
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from four_option_budgets import ALL, law

PANELS=("arc-challenge","openbookqa","cosmosqa","race")
SIZES=dict(zip(PANELS,(256,256,256,128)))
METHODS=("precedence-pilot","residual-pilot")
P=law(8,"precedence")
W=law(8,"full")
SIX=tuple(law(4,"full"))
# Six orthogonal signed directions spanning the part of the precedence residual
# perpendicular to the centered Williams block means (display indices, lexicographic).
G=(
    ((0,7,16,23),(2,10,13,21)), ((0,7,17,22),(1,6,16,23)),
    ((1,6,17,22),(4,8,15,19)), ((2,11,13,20),(3,10,12,21)),
    ((3,11,12,20),(5,9,14,18)), ((4,9,14,19),(5,8,15,18)),
)
F_EDGES=combinations(SIX,2)


def index_directions(pairs):
    return {tuple(sorted(a+b)):(a,b) for a,b in pairs}


F_DIR=index_directions(F_EDGES)
G_DIR=index_directions(G)
R={b:F(int(b in F_DIR),30)+F(int(b in G_DIR),12) for b in sorted(F_DIR.keys()|G_DIR.keys())}
assert len(R)==18 and sum(R.values())==1


def pair_marginals(weights):
    out=defaultdict(F)
    for b,p in weights.items():
        for edge in combinations(b,2):
            out[edge]+=p
    return dict(out)


P_EDGE=pair_marginals(P)
W_EDGE=pair_marginals(W)
assert len(P_EDGE)==len(W_EDGE)==276
EDGE={e:(W_EDGE[e]/p-1)/64 for e,p in P_EDGE.items()}
EDGE_DEN=lcm(*(v.denominator for v in EDGE.values()))
EDGE_INT={e:int(w*EDGE_DEN) for e,w in EDGE.items()}


def role_key(panel,group):
    return sha256(("eight-pilot-roles-v1|"+panel+"|"+group).encode()).hexdigest()


def roles(panel,groups):
    assert len(groups)==len(set(groups))==SIZES[panel]
    ordered=sorted(groups,key=lambda group:role_key(panel,group))
    count=len(groups)//4
    return ordered[:count],ordered[count:]


def weights(method):
    return P if method=="precedence-pilot" else R


@cache
def tickets(method):
    distribution=weights(method)
    den=lcm(*(v.denominator for v in distribution.values()))
    return tuple(b for b,w in distribution.items() for _ in range(int(w*den)))


def chosen_block(method,panel,group):
    key="eight-pilot-orders-v1|"+method+"|"+panel+"|"+group
    return Random(int.from_bytes(sha256(key.encode()).digest(),"big")).choice(tickets(method))


def signed_square(direction,values):
    a,b=direction
    return (sum(values[j] for j in a)-sum(values[j] for j in b))**2


def sparse_score(method,b,values):
    """Only read the eight pairs of correctness outcomes exposed to the pilot."""
    if set(values)!=set(b) or any(x not in (-1,0,1) for x in values.values()):
        raise ValueError("An eight-order pilot needs exactly its assigned ternary contrasts.")
    if method=="precedence-pilot":
        return F(sum(EDGE_INT[i,j]*(values[i]-values[j])**2 for i,j in combinations(b,2)),EDGE_DEN)
    if method!="residual-pilot" or b not in R:
        raise ValueError("Unknown pilot method or block.")
    pos=F(signed_square(G_DIR[b],values),6) if b in G_DIR else 0
    neg=F(signed_square(F_DIR[b],values),15) if b in F_DIR else 0
    return (pos-neg)/(176*R[b])


def choose(score):
    return "full" if score>0 else "precedence"


def exact_risk(d,weights):
    mu=F(sum(d),24)
    return sum(p*(F(sum(d[i] for i in b),len(b))-mu)**2 for b,p in weights.items())


def verify():
    # Quadratic forms coincide iff their diagonal and two-coordinate probes coincide.
    assert all(sum(w for b,w in R.items() if j in b)==F(1,3) for j in ALL)
    for k,gen in ((1,((i,) for i in ALL)),(2,combinations(ALL,2))):
        for coords in gen:
            d=tuple(int(i in coords) for i in ALL)
            target=exact_risk(d,P)-exact_risk(d,W)
            for method in METHODS:
                actual=sum(w*sparse_score(method,b,{i:d[i] for i in b}) for b,w in weights(method).items())
                assert actual==target
    # Check the companion signed directions also annihilate the common feature span.
    from four_option_budgets import ORDERS
    pos=[[int(row[i]==a) for row in ORDERS] for i in range(4) for a in range(4)]
    pre=[[int(row.index(a)<row.index(b)) for row in ORDERS] for a,b in combinations(range(4),2)]
    for dirs in (F_DIR,G_DIR):
        for direction in dirs.values():
            assert all(signed_square(direction,v)==0 for v in pos+pre)
    print("All 300 quadratic probes agree for both eight-order estimators.")


if __name__=="__main__":
    verify()
