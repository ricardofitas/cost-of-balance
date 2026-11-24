"""Dispatch, score and replay the frozen five-option rig evaluation."""
import argparse
import json
from math import isfinite
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

from stage import (ROOT, OLD, SOURCE, SOURCE_SHA, SOURCE_REV, GPU, QUEUE, COUNT, SHARD,
                   SLOTS, ORDERS, MODELS, NATIVE, BINARY, CODE, ROSTER, COMMITMENT,
                   digest, module, read, signature, write)
from score_designs import measure, supports

CHECKPOINT = ROOT / "inference.json"
PROTOCOL = ROOT / "PROTOCOL.md"
RUNNER = OLD / "extended/run"
PIPE = CODE / "hellaswag-gpu-admission/native_pipe.py"
ORDER_CALLS = len(ORDERS)
FILES = [PROTOCOL, ROSTER, NATIVE, BINARY, PIPE, SOURCE,
         *(ROOT / n for n in ("stage.py", "worker.py", "score_designs.py", "static_decode_five.cpp",
                              "verify_position.py", "verify_binary_adjacency.py", "verify_real_adjacency.py", "solutions.json"))]


def commit():
    roster = read(ROSTER)
    assert len(roster["rows"]) == COUNT and len({row["concept_sha256"] for row in roster["rows"]}) == COUNT
    assert roster["source_sha256"] == SOURCE_SHA and digest(SOURCE) == SOURCE_SHA
    committed = {"source_revision": SOURCE_REV, "source_sha256": SOURCE_SHA,
                 "questions": COUNT, "reference_orders": ORDER_CALLS, "design_calls": 5,
                 "models": {name: {"revision": pin[1], "sha256": pin[2]} for name, pin in MODELS.items()},
                 "files": {path.name: digest(path) for path in FILES}}
    if COMMITMENT.exists() and read(COMMITMENT) != committed:
        raise RuntimeError("Frozen study commitment changed")
    write(COMMITMENT, committed)
    return digest(COMMITMENT)


def check_frozen(pin):
    frozen = read(COMMITMENT)
    if pin != digest(COMMITMENT) or any(digest(path) != frozen["files"][path.name] for path in FILES):
        raise RuntimeError("Frozen study commitment changed")
    return read(ROSTER)


def prepare_job(run_id):
    pin = commit()
    write(ROOT / f"{run_id}.json", {"run_id": run_id, "campaign": "evaluation", "cwd": str(ROOT),
          "argv": [str(RUNNER), str(Path(__file__).resolve()), "infer", pin],
          "max_seconds": 4800, "protocol": f"{PROTOCOL} sha256={digest(PROTOCOL)} commitment_sha256={pin}",
          "checkpoint": str(CHECKPOINT), "gpu_count": 1, "eligible_gpu_uuids": [GPU]})
    print("Committed", pin, flush=True)


def decode(pipe, name, path, pin, limit):
    if digest(path) != pin:
        raise ValueError("Fixture bytes differ")
    lines = path.read_text().splitlines()
    assert len(lines) == 1 + SHARD * ORDER_CALLS
    lengths = [int(line.split(maxsplit=1)[0]) for line in lines[1:]]
    log = path.with_suffix(".native.log")
    model = MODELS[name][0]
    picks = []
    with pipe.session(NATIVE, BINARY, model, path, log) as child:
        live = subprocess.check_output(["nvidia-smi", "-i", GPU, "--query-compute-apps=pid",
                                         "--format=csv,noheader"], text=True)
        if str(child.pid) not in live.split():
            raise RuntimeError("Native worker did not attach to assigned GPU")
        for start in range(0, limit, SLOTS):
            child.stdin.write(f"{start}\n")
            child.stdin.flush()
            response = json.loads(child.stdout.readline())
            probs = response["probabilities"]
            if (response["token_count"] != sum(lengths[start:start+SLOTS]) or len(probs) != SLOTS
                    or any(len(p) != 5 or not all(isfinite(x) and 0 <= x <= 1 for x in p)
                           or abs(sum(p)-1) > 1e-9 for p in probs)):
                raise ValueError("Invalid five-choice CUDA decode")
            picks.extend(max(range(5), key=p.__getitem__) for p in probs)
    return picks


def valid_checkpoint(file, fixture, pin):
    data = read(file)
    if (data["study"] != pin or data["fixture"] != fixture["sha256"]
            or len(data["choices"]) != SHARD * ORDER_CALLS or any(type(x) is not int or x not in range(5)
                                                                 for x in data["choices"])):
        raise ValueError("Checkpoint does not match committed fixture")
    return data["choices"]


def inference(pin):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU or not os.environ.get("RIG_GPU_RUN_ID"):
        raise RuntimeError("Shared queue lease on the RTX 4060 Ti required")
    plan = check_frozen(pin)
    pipe = module("native_pipe", PIPE)
    qualification = {}
    for name, (file, _, model_pin) in MODELS.items():
        if digest(file) != model_pin:
            raise ValueError("Model bytes differ")
        entry = plan["qualification"][name]
        synthetic = ROOT / name / entry["file"]
        first = decode(pipe, name, synthetic, entry["sha256"], ORDER_CALLS)
        second = decode(pipe, name, synthetic, entry["sha256"], ORDER_CALLS)
        if first != second:
            raise RuntimeError(f"{name} synthetic cold replay disagrees")
        qualification[name] = len(first)
    for name in MODELS:
        folder = ROOT / name
        for i, entry in enumerate(plan["fixtures"][name]):
            out = folder / f"pred-{i:03d}.json"
            if out.exists():
                valid_checkpoint(out, entry, pin)
                continue
            preds = decode(pipe, name, folder / entry["file"], entry["sha256"], SHARD * ORDER_CALLS)
            write(out, {"study": pin, "fixture": entry["sha256"], "choices": preds})
            print(name, (i + 1) * SHARD, "/", COUNT, flush=True)
    write(CHECKPOINT, {"run_id": os.environ["RIG_GPU_RUN_ID"], "study": pin,
                       "synthetic_choices_verified_per_model": qualification,
                       "label_columns_read": 0, "reference_calls": COUNT * ORDER_CALLS * len(MODELS)})


def score():
    inference_receipt = read(CHECKPOINT)
    pin = inference_receipt["study"]
    plan = check_frozen(pin)
    state = QUEUE / "state" / inference_receipt["run_id"]
    release = read(state / "release.json")
    started = read(state / "started.json")
    if (release["status"] != "completed" or not release["eligible_checkpoint"]
            or started["gpus"] != [GPU] or inference_receipt["label_columns_read"] != 0
            or inference_receipt["reference_calls"] != COUNT*ORDER_CALLS*len(MODELS)):
        raise RuntimeError("Successful full release required before reading answer keys")

    vectors = []
    for name in MODELS:
        choices = [x for i, entry in enumerate(plan["fixtures"][name])
                   for x in valid_checkpoint(ROOT / name / f"pred-{i:03d}.json", entry, pin)]
        if len(choices) != COUNT * ORDER_CALLS:
            raise ValueError("Incomplete predictions")
        vectors.append(choices)
    scoped = [row["id"] for row in plan["rows"]]
    labels = pq.read_table(SOURCE, columns=["id", "question", "question_concept", "choices.label", "answerKey"],
                           filters=[("id", "in", scoped)]).to_pylist()
    by_id = {r["id"]: r for r in labels}
    assert len(labels) == COUNT and set(by_id) == set(scoped)
    key = []
    for frozen in plan["rows"]:
        source = by_id[frozen["id"]]
        assert (frozen["stem_sha256"] == signature(source["question"])
                and frozen["concept_sha256"] == signature(source["question_concept"]))
        key.append(source["label"].index(source["answerKey"]))
    picks = np.asarray(vectors).reshape(len(MODELS), COUNT, ORDER_CALLS)
    correct = np.asarray(ORDERS)[np.arange(ORDER_CALLS), picks] == np.array(key)[None, :, None]
    gap = correct[0].astype(np.int64) - correct[1].astype(np.int64)
    results = measure(gap, supports())
    masks = [[sum(int(bit) << i for i, bit in enumerate(row)) for row in selected] for selected in correct]
    write(ROOT / "correctness-masks.json", {"study_sha256": pin,
          "format": "Question ID followed by Qwen and SmolLM3 hexadecimal correctness masks. Bit zero is the first lexicographic display permutation.",
          "rows": [[qid, f"{a:030x}", f"{b:030x}"] for qid, a, b in zip(scoped, *masks, strict=True)]})
    restored = release["boinc"][GPU]
    results.update(inference_receipt)
    results.update({"questions": COUNT, "deploy_calls_per_design": 5*COUNT*len(MODELS),
                    "prediction_checkpoints": COUNT//SHARD*len(MODELS), "device": "RTX 4060 Ti",
                    "source_sha256": SOURCE_SHA, "roster_sha256": digest(ROSTER),
                    "reference_gap": float(gap.mean()),
                    "accuracy_by_model": dict(zip(MODELS, map(float, correct.mean(axis=(1,2))), strict=True)),
                    "started_utc": started["actual_start_utc"], "released_utc": release["released_utc"],
                    "boinc_mode_restored": restored["before"] == restored["restored"]})
    write(ROOT / "results.json", results)
    print("Scored", results["relative_gain"], results["paired_bootstrap_95"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("job", "infer", "score"))
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    if args.action == "job":
        prepare_job(args.value)
    elif args.action == "infer":
        inference(args.value)
    else:
        score()
