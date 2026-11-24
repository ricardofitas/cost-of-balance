"""Replay the paper's finite certificates using the Python standard library."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


BASE = Path(__file__).resolve().parent / "certificates"


def run(script, *args):
    return subprocess.check_output([sys.executable, "-I", "-B", str(BASE/script), *args], text=True)


def matched(script, receipt):
    assert json.loads(run(script)) == json.loads((BASE/receipt).read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    opt = parser.add_mutually_exclusive_group()
    opt.add_argument("--catalogue-dir",type=Path,help="local directory with the three pinned catalogue files")
    opt.add_argument("--small",action="store_true",help="omit catalogue-dependent adjacency bounds at 8, 9 and 10")
    args=parser.parse_args()
    run("verify_four.py")
    run("verify_sparse_pilot.py")
    run("verify_two_prime.py")
    run("verify_twelve.py")
    matched("verify_four_budgets.py","four-budget-certificate.json")
    matched("verify_position.py","position-certificate.json")
    matched("verify_binary_adjacency.py","binary-adjacency-certificate.json")
    matched("verify_real_adjacency.py","real-adjacency-certificate.json")
    matched("verify_position_family.py","position-family-certificate.json")
    run("crosscheck_spectra.py")
    args_adj = ["--small"] if args.small else []
    if args.catalogue_dir:
        args_adj = ["--catalogue-dir",str(args.catalogue_dir.resolve())]
    observed = json.loads(run("verify_adjacency.py",*args_adj))
    expected = json.loads((BASE/"adjacency-certificate.json").read_text())
    if args.small:
        del expected["larger_adjacency"]
    assert observed == expected
    print("All selected exact certificates matched.")
