"""Check the six residual directions and both eight-order transfer scores using exact arithmetic."""
from itertools import combinations
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/low-budget-selection"))
import pilot
from four_option_budgets import ORDERS


def check():
    vectors=[]
    for plus,minus in pilot.G:
        assert len(set(plus+minus))==8
        v=[int(i in plus)-int(i in minus) for i in pilot.ALL]
        assert sum(x*x for x in v)==8
        assert all(sum(v[i] for i in b)==0 for b in pilot.SIX)
        for u in vectors:
            assert sum(x*y for x,y in zip(u,v))==0
        vectors.append(v)
    assert len(vectors)==6
    for b in pilot.R:
        assert all(sum(ORDERS[j][s]==a for j in b)==2 for a in range(4) for s in range(4))
        assert all(sum(ORDERS[j].index(a)<ORDERS[j].index(c) for j in b)==4 for a,c in combinations(range(4),2))
    pilot.verify()


if __name__=="__main__":
    check()
