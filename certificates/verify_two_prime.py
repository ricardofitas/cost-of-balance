"""Independently enumerate relative cycles of both implemented Latin templates."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_position import sample_position

class FixedDraw:
    def __init__(self, cyclic):
        self.cyclic = cyclic

    def randrange(self, denominator):
        self.denominator = denominator
        return 0 if self.cyclic else denominator - 1

    def sample(self, population, count):
        assert len(population) == count
        return list(population)

def counts(n, cyclic):
    draw = FixedDraw(cyclic)
    square = sample_position(n, draw)
    target = set(range(n))
    assert all(set(row) == target for row in square)
    assert all(set(column) == target for column in zip(*square))
    result = Counter()
    for left, right in combinations(square, 2):
        shift = dict(zip(left, right))
        cycles = []
        while shift:
            x = next(iter(shift))
            length = 0
            while x in shift:
                x = shift.pop(x)
                length += 1
            cycles.append(length)
        result[tuple(sorted(cycles))] += 1
    return result

if __name__ == '__main__':
    n = 262
    p = n//2
    w = Fraction((n-1)*(n-3), n*n-3*n+1)
    cyclic, other = counts(n, True), counts(n, False)
    assert cyclic == {(2,)*p:p, (p,p):p*(p-1), (n,):p*(p-1)}
    assert other == {(p,p):p*(p-1), (n,):p*p}
    swaps = w*p*cyclic[(2,)*p]/(n*(n-1)//2)
    assert swaps == Fraction(n*(n-3),2*(n*n-3*n+1))
    print('Both Latin templates and all relative pairs verified at n=262.')
