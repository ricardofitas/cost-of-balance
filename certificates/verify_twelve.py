"""Check the twelve-option mixture, the degree cutoff, and the runner-up risk."""
from collections import Counter
from fractions import Fraction
from math import comb, factorial
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from verify_position_family import class_size, signature

atoms = json.loads((ROOT / 'position-templates.json').read_text())['12']
mix = Counter()
for atom, expected_swaps in zip(atoms, (54, 27), strict=True):
    counts = signature(atom['rows'])
    assert sum(k.count(2)*v for k, v in counts.items()) == expected_swaps
    for cycle, count in counts.items():
        mix[cycle] += Fraction(atom['weight']) * count / comb(12, 2)
assert sum(mix.values()) == 1
assert sum(k.count(2)*v for k, v in mix.items()) == Fraction(54, 109)
square = sum(q*q*factorial(12)/class_size(k) for k, q in mix.items())
assert 186**2 < 109**2 * square < 187**2
risks = json.loads((ROOT / 'position-family-certificate.json').read_text())['12']
values = {shape: Fraction(s) for shape, s in risks['nontrivial_eigenvalues_by_partition'].items()}
assert len(values) == 76
assert values.pop('10,2') == values.pop('10,1,1') == Fraction(10, 109)
assert max(values.values()) == Fraction(1873, 21582)
print('Twelve-option histogram and risk checks passed.')
