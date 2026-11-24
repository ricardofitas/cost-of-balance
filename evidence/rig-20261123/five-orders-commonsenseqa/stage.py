"""Prepare the fixed CommonsenseQA panel and token fixtures without answer keys."""
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from itertools import permutations
import os
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OLD = Path(os.environ.get("RIG_PREVIOUS_EVALUATION_WORK", ROOT.parents[2] / "col/work")).resolve()
sys.path.insert(0, str(OLD / "github-followup"))
from qualify import PIN, ASSETS, CODE, digest, module, read, write
from common import GPU, QUEUE

SOURCE = ROOT / "commonsenseqa.parquet"
SOURCE_SHA = "bdbd9bf9cc4d2349b24901038b2ab2f58e10e4e507ad2fd425dca55cd3cb6660"
SOURCE_REV = "94630fe30dad47192a8546eb75f094926d47e155"
MODELS = {k: (ROOT.parent / "hellaswag-3b/weights" / v[0], v[1], v[2]) for k, v in PIN.items()}
BINARY = ASSETS / "llama-cpp/llama-b12631/llama-server"
NATIVE = ROOT / "static_decode_five"
COUNT, SHARD, SLOTS = 192, 8, 8
ORDERS = tuple(permutations(range(5)))
DIGITS = "12345"
ROSTER = ROOT / "frozen.json"
COMMITMENT = ROOT / "commitment.json"


def signature(v):
    return sha256(" ".join(v.casefold().split()).encode()).hexdigest()


def prompt(row, order):
    instruction = "Choose the best answer. Reply with exactly one digit: 1, 2, 3, 4, or 5."
    choices = "\n".join(f"{j}. {row['choices']['text'][i]}" for j, i in enumerate(order, 1))
    return f"{instruction}\n\nQuestion: {row['question']}\n\nOptions:\n{choices}"


def tokenize(cpu, endpoint, row):
    vectors = [cpu.render_tokens(*endpoint, prompt(row, order)) for order in ORDERS]
    if len(set(map(tuple, vectors))) != len(ORDERS):
        return None
    return vectors if all(0 < len(tokens) < 512 for tokens in vectors) else None


def fixture(path, digit_ids, vectors):
    assert len(vectors) == SHARD * len(ORDERS)
    path.write_text(" ".join(map(str, digit_ids)) + "\n" +
                    "".join(f"{len(seq)} {' '.join(map(str, seq))}\n" for seq in vectors))
    return {"file": path.name, "sha256": digest(path)}


def load_candidates():
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError("CommonsenseQA source hash differs")
    rows = pq.read_table(SOURCE, columns=["id", "question", "question_concept", "choices"]).to_pylist()
    ids = Counter(row["id"] for row in rows)
    stems = Counter(signature(row["question"]) for row in rows)
    eligible = [r for r in rows if ids[r["id"]] == stems[signature(r["question"])] == 1
                and r["question"].strip() and r["question_concept"].strip()
                and r["choices"]["label"] == list("ABCDE") and len(r["choices"]["text"]) == 5
                and len({signature(text) for text in r["choices"]["text"] if text.strip()}) == 5]
    return sorted(eligible, key=lambda r: sha256(f"commonsenseqa-five-v1:{r['id']}".encode()).digest())


def prepare():
    if ROSTER.exists():
        print("Existing label-free roster", digest(ROSTER))
        return
    cpu = module("prompt_admission", CODE / "race-prompt-admission/run.py")
    if digest(BINARY) != cpu.BINARY_SHA:
        raise ValueError("Tokenizer binary differs")
    for path, _, pin in MODELS.values():
        if digest(path) != pin:
            raise ValueError("3B model bytes differ")
    candidates = load_candidates()
    rows, concepts = [], set()
    manifest = {"source_sha256": SOURCE_SHA, "source_revision": SOURCE_REV, "fixtures": {k: [] for k in MODELS},
                "qualification": {}, "rows": rows}
    with ExitStack() as stack:
        endpoints, ids, pending = {}, {}, {k: [] for k in MODELS}
        for name, (path, _, _) in MODELS.items():
            folder = ROOT / name
            folder.mkdir(exist_ok=True)
            endpoints[name] = stack.enter_context(cpu.cpu_server(BINARY, path, folder))
            label_ids = []
            for digit in DIGITS:
                data = cpu.runtime.request(*endpoints[name], "/tokenize",
                                           {"content": digit, "add_special": False,
                                            "parse_special": False, "with_pieces": True})
                token, = data["tokens"]
                if token["piece"] != digit or type(token["id"]) is not int:
                    raise ValueError("Digit is not a standalone token")
                label_ids.append(token["id"])
            assert len(set(label_ids)) == 5
            ids[name] = label_ids
            synthetic = {"question": "Which tool is used to make a written mark on paper?",
                         "choices": {"text": ["a pen", "a shoe", "a cup", "a pillow", "a spoon"]}}
            tokens = tokenize(cpu, endpoints[name], synthetic)
            if tokens is None:
                raise ValueError("Synthetic admission question does not fit")
            manifest["qualification"][name] = fixture(folder / "synthetic.tokens", label_ids, tokens * SHARD)
        for row in candidates:
            concept = signature(row["question_concept"])
            if concept in concepts:
                continue
            tokenized = {}
            for name in MODELS:
                vectors = tokenize(cpu, endpoints[name], row)
                if vectors is None:
                    break
                tokenized[name] = vectors
            if len(tokenized) != len(MODELS):
                continue
            concepts.add(concept)
            rows.append({"id": row["id"], "concept_sha256": concept, "stem_sha256": signature(row["question"])})
            for name, vectors in tokenized.items():
                pending[name].extend(vectors)
            if len(rows) % SHARD == 0:
                for name in MODELS:
                    path = ROOT / name / f"batch-{len(rows)//SHARD-1:03d}.tokens"
                    manifest["fixtures"][name].append(fixture(path, ids[name], pending[name]))
                    pending[name].clear()
                if len(rows) % 32 == 0:
                    print("Token-qualified", len(rows), "/", COUNT, flush=True)
            if len(rows) == COUNT:
                break
    if len(rows) != COUNT:
        raise RuntimeError(f"Only {len(rows)} label-free items passed context qualification")
    write(ROSTER, manifest)
    print("Frozen label-free roster", digest(ROSTER), flush=True)


if __name__ == "__main__":
    prepare()
