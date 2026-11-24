"""Replay the prespecified development screen on already scored local panels."""

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import file_digest
from itertools import combinations
import json
from pathlib import Path
import runpy

import numpy as np
import pyarrow.parquet as pq

W = "williams"
P = "precedence-minimax"
MARGIN = 0.05
PAIRS = ("Qwen0.5B–Smol360M", "Qwen1.5B–Granite2B", "Qwen3B–Smol3B")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def load_panels(current, prior, replay):
    def panel(name, bench, pair, prompt, labels, traces, strata=None):
        traces = np.asarray(traces, dtype=int).reshape(2, len(labels), 24)
        assert np.all((traces >= 0) & (traces < 4))
        correct = np.asarray(replay["ORDERS"])[np.arange(24), traces] == np.asarray(labels)[None, :, None]
        gap = correct.astype(int)[0] - correct.astype(int)[1]
        return name, bench, pair, prompt, gap[:, replay["DISPLAY_INDEX"]], strata

    for bench, group in (("sciq-test", "SciQ"), ("arc-easy", "ARC")):
        for style, directory, pair in [
            ("standard", "autonomous", 1),
            *((style, "extended", 0) for style in ("standard", "direct", "answer-cue")),
        ]:
            state = current / directory / "state"
            key = bench if directory == "autonomous" else f"{bench}--{style}"
            roster = state / f"{key}-roster.json"
            rows = read(roster)["items"]
            assert (state / f"{key}-item-loss.json").is_file()
            traces = [[], []]
            for m in range(2):
                for offset in range(0, len(rows), 8):
                    record = read(state / key / f"model-{m}-{offset:04d}.json")
                    assert record["roster_sha256"] == digest(roster)
                    traces[m].extend(record["displayed_choices"])
            yield panel(f"{key}/{directory}", group, pair, style, [r["answer"] for r in rows], traces)

        root = current / "github-followup"
        rows = read(current / "autonomous/state" / f"{bench}-roster.json")["items"][:32]
        for style in ("standard", "direct"):
            traces = [read(root / "run-pilot" / f"{bench}-{style}" / f"{tag}.json")["choices"]
                      for tag in ("qwen", "smol")]
            yield panel(f"{bench}--{style}/3B", group, 2, style, [r["answer"] for r in rows], traces)

    confirmation = current / "confirmation-metadata"
    for directory, key, bench, pair, tags, source in (
        ("near-tie", "arc-challenge", "ARC", 1, ("qwen", "granite"), confirmation / "arc-challenge-validation.parquet"),
        ("near-tie", "openbookqa", "OpenBookQA", 1, ("qwen", "granite"), current / "near-tie/openbookqa-validation.parquet"),
        ("confirmation-metadata", "mmlu-validation", "MMLU", 2, ("qwen", "smol"), confirmation / "mmlu-validation.parquet"),
        ("confirmation-metadata", "logiqa-test", "LogiQA", 2, ("qwen", "smol"), confirmation / "logiqa-test.jsonl"),
    ):
        root = current / directory / key
        manifest = root / "frozen.json"
        plan = read(manifest)
        pin = digest(manifest)
        assert pin == read(root / "inference.json")["manifest_sha256"]
        assert digest(source) == plan["source_sha256"]
        rows = plan["rows"]
        if directory == "near-tie":
            raw = pq.read_table(source, columns=["id", "choices", "answerKey"]).to_pylist()
            selected = [raw[r["index"]] for r in rows]
            assert [r["id"] for r in selected] == [r["id"] for r in rows]
            labels = [r["choices"]["label"].index(r["answerKey"]) for r in selected]
        elif key == "mmlu-validation":
            raw = pq.read_table(source, columns=["answer"]).column(0).to_pylist()
            labels = [raw[r["index"]] for r in rows]
        else:
            raw = [json.loads(line) for line in source.read_bytes().splitlines() if line.strip()]
            labels = [raw[r["index"]]["answer"] for r in rows]
        traces = []
        for tag in tags:
            choices = []
            for i, fixture in enumerate(plan["fixtures"][tag]):
                record = read(root / tag / f"pred-{i:04d}.json")
                assert record["roster"] == pin and record["fixture"] == fixture["sha256"]
                choices.extend(record["choices"])
            traces.append(choices)
        strata = [r["subject"] for r in rows] if key == "mmlu-validation" else None
        yield panel(f"{key}/{directory}", bench, pair, "standard", labels, traces, strata)

    collection = prior / "hellaswag-complete-collection"
    approval = read(collection / "label-approval.json")
    export = collection / "prediction-export.json"
    assert approval["authorize_confirmation_label_read"] and approval["independent_review"]
    assert digest(export) == approval["predictions_sha256"]
    rows = read(export)["rows"]
    stage = prior / "hellaswag-scoring-capsule/package/stage"
    staging = replay["load_committed"](stage / "staging-receipt.json", approval["staging_sha256"])
    roster = replay["load_committed"](stage / "roster.json", staging["cohort_file_sha256"])
    assert list(replay["keys"](roster)) == list(replay["keys"](rows))
    publisher = prior / "hellaswag-order-admission-runtime/validation.parquet"
    labels = replay["load_publisher"](publisher, roster)
    traces = [[int(digit) - 1 for row in rows for digit in row["displayed_digits"][m]] for m in (0, 1)]
    yield panel("hellaswag-expanded", "HellaSwag", 1, "hellaswag-numeric",
                [labels[r["ind"], r["source_id"]] for r in rows], traces, [r["stratum"] for r in roster])


def projections(orders):
    position = [[int(p == slot) for p in order for slot in range(4)] for order in orders]
    precedence = [[int(order[i] < order[j]) for i, j in combinations(range(4), 2)] for order in orders]
    adjacency = [[int(order[j] == order[i] + 1) for i in range(4) for j in range(4) if i != j]
                 for order in orders]
    basis = np.ones((24, 1)) / np.sqrt(24)
    spaces = {}
    for name, features in (("position", position), ("precedence", precedence), ("adjacency", adjacency)):
        features = np.asarray(features)
        u, s, _ = np.linalg.svd(features - basis @ (basis.T @ features), full_matrices=False)
        extra = u[:, s > 1e-10]
        spaces[name] = extra @ extra.T
        basis = np.column_stack((basis, extra))
    identity = np.eye(24)
    reversal = identity[[orders.index(tuple(3 - x for x in order)) for order in orders]]
    adjacent = spaces.pop("adjacency")
    spaces["even_adjacency"] = adjacent @ ((identity + reversal) / 2)
    spaces["odd_adjacency"] = adjacent @ ((identity - reversal) / 2)
    spaces["residual"] = identity - basis @ basis.T
    assert np.allclose(sum(spaces.values()), identity - 1/24)
    for name, rank in zip(spaces, (9, 3, 2, 3, 6), strict=True):
        assert np.linalg.matrix_rank(spaces[name], tol=1e-10) == rank
    return spaces


def lower_bound(values, strata):
    groups = defaultdict(list)
    for value, group in zip(values, strata, strict=True):
        groups[group].append(value)
    variance = sum(len(v) * np.var(v, ddof=1) for v in groups.values() if len(v) > 1)
    return float(values.mean() - 1.95996398454 * np.sqrt(variance) / len(values))


def main(current, prior):
    replay_file = prior / "hellaswag-scoring-capsule/package/replay.py"
    replay = runpy.run_path(str(replay_file))
    supports = replay["support_catalog"]()
    matrices = {}
    spaces = projections(replay["ORDERS"])
    for design, blocks in supports.items():
        incidence = np.zeros((len(blocks), 24))
        for row, block in zip(incidence, blocks, strict=True):
            row[list(block)] = 6
        incidence -= 1
        matrices[design] = incidence.T @ incidence / (576 * len(blocks))
    contrast = matrices[P] - matrices[W]
    panels, risks = {}, {}
    max_operator_error, max_decomposition_error = 0.0, 0.0
    for name, benchmark, pair, prompt, gap, strata in load_panels(current, prior, replay):
        n = len(gap)
        risk = {}
        exact = {}
        for design, blocks in supports.items():
            block_sums = gap[:, blocks].sum(axis=2)
            numerator = ((6 * block_sums - gap.sum(axis=1)[:, None]) ** 2).sum(axis=1)
            denominator = 576 * len(blocks)
            risk[design] = numerator / denominator
            exact[design] = str(Fraction(int(numerator.sum()), denominator * n * n))
            operator = ((gap @ matrices[design]) * gap).sum(axis=1)
            max_operator_error = max(max_operator_error, float(np.max(abs(operator - risk[design]))))
        assert max_operator_error < 1e-12
        profiles = {key: gap @ matrix for key, matrix in spaces.items()}
        terms = {}
        for left, a in profiles.items():
            terms[left] = ((a @ contrast) * a).sum(axis=1)
        for left, right in combinations(profiles, 2):
            terms[f"{left} x {right}"] = 2 * ((profiles[left] @ contrast) * profiles[right]).sum(axis=1)
        error = float(np.max(abs(sum(terms.values()) - (risk[P] - risk[W]))))
        max_decomposition_error = max(max_decomposition_error, error)
        assert max_decomposition_error < 1e-12
        strata = strata or ["all"] * n
        gains = {key: float(1 - risk[W].mean() / item.mean()) if item.mean() else None
                 for key, item in risk.items() if key != W}
        lower = {key: lower_bound((1 - MARGIN) * item - risk[W], strata)
                 for key, item in risk.items() if key != W}
        panels[name] = {"benchmark": benchmark, "pair": PAIRS[pair], "prompt": prompt, "items": n,
                        "both_at_least_3B": pair == 2, "exact_panel_mse": exact,
                        "relative_williams_gain": gains, "normal_lower_practical_margin_per_item": lower,
                        "precedence_minus_williams_components_per_item":
                            {key: float(values.mean()) for key, values in terms.items()}}
        risks[name] = risk
    assert len(panels) == 17

    def clears(panel, baseline):
        gain = panels[panel]["relative_williams_gain"].get(baseline)
        return gain is not None and gain > MARGIN and panels[panel]["normal_lower_practical_margin_per_item"][baseline] > 0

    def distinct(names, field):
        return len({panels[name][field] for name in names})

    rules = {"any": lambda p: True, "both_at_least_3B": lambda p: p["both_at_least_3B"]}
    for prompt in sorted({p["prompt"] for p in panels.values()}):
        rules[prompt] = lambda p, template=prompt: p["prompt"] == template
        rules[f"{prompt}/both_at_least_3B"] = (
            lambda p, template=prompt: p["prompt"] == template and p["both_at_least_3B"]
        )
    folds, successes, failures = {}, defaultdict(list), defaultdict(list)
    for held, panel in panels.items():
        train = [key for key, val in panels.items()
                 if val["benchmark"] != panel["benchmark"] and val["pair"] != panel["pair"]]
        baseline = min(supports, key=lambda design: np.mean([risks[key][design].mean() / risks[key][P].mean()
                                                             for key in train]))
        trained, qualified = [], []
        for rule, condition in rules.items():
            matched = [key for key in train if condition(panels[key])]
            if (condition(panel) and distinct(matched, "benchmark") >= 2 and distinct(matched, "pair") >= 2
                    and all(clears(key, P) for key in matched)):
                trained.append(rule)
                if clears(held, P) and clears(held, baseline):
                    qualified.append(rule)
                    successes[rule].append(held)
                else:
                    failures[rule].append(held)
        folds[held] = {"train": train, "fixed_baseline": baseline, "recommendations": trained,
                       "successful_recommendations": qualified}

    accepted = [rule for rule in rules if not failures[rule] and distinct(successes[rule], "benchmark") >= 2
                and distinct(successes[rule], "pair") >= 2]
    return {"replay_sha256": digest(replay_file), "panels": panels, "folds": folds,
            "rules": list(rules), "accepted_rules": accepted,
            "decision": "advance to prospective freeze" if accepted else "stop before new GPU inference",
            "maximum_operator_error": max_operator_error, "maximum_decomposition_error": max_decomposition_error,
            "new_inference_calls": 0, "sealed_panels_read": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path, help="Previous evaluation work directory (23 November)")
    parser.add_argument("prior", type=Path, help="Previous evaluation work directory (22 November)")
    args = parser.parse_args()
    print(json.dumps(main(args.current, args.prior), indent=2, sort_keys=True, allow_nan=False))
