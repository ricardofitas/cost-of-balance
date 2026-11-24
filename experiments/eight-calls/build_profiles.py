"""Recreate anonymous stratum-by-profile counts from the released four-option studies."""

import argparse
from collections import Counter, defaultdict
from gzip import GzipFile
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path


ORDERS = tuple(permutations(range(4)))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def correctness_mask(picks, key):
    assert len(picks) == 24 and all(type(x) is int and 0 <= x < 4 for x in picks)
    assert 0 <= key < 4
    return sum(1 << i for i,(row,choice) in enumerate(zip(ORDERS,picks)) if row[choice] == key)


def histogram(rows):
    groups = defaultdict(Counter)
    for group, left, right in rows:
        assert 0 <= left < 2**24 and 0 <= right < 2**24
        groups[group][left & ~right, right & ~left] += 1
    return {s:[[f"{a:06x}", f"{b:06x}", count] for (a,b),count in sorted(c.items())]
            for s,c in sorted(groups.items())}


def matched_confirmation(folder, panel):
    import pyarrow.parquet as pq
    study = folder/panel
    plan = read(study/"frozen.json")
    pin = digest(study/"frozen.json")
    assert read(folder/"study-commitment.json")["roster_sha256"][panel] == pin
    assert read(study/"inference.json")["manifest_sha256"] == pin
    src = folder/("mmlu-validation.parquet" if panel == "mmlu-validation" else "logiqa-test.jsonl")
    assert digest(src) == plan["source_sha256"]
    if panel == "mmlu-validation":
        keys = pq.read_table(src, columns=["answer"]).column(0).to_pylist()
    else:
        keys = [json.loads(line)["answer"] for line in src.read_bytes().splitlines() if line.strip()]
    traces = []
    checkpoints = sha256()
    for model in ("qwen","smol"):
        values = []
        for j, fixture in enumerate(plan["fixtures"][model]):
            path = study/model/f"pred-{j:04d}.json"
            record = read(path)
            assert record["roster"] == pin and record["fixture"] == fixture["sha256"]
            assert len(record["choices"]) == 8*24
            checkpoints.update(bytes.fromhex(digest(path)))
            values.extend(record["choices"])
        assert len(values) == len(plan["rows"])*24
        traces.append(values)
    rows = []
    assert len(plan["rows"]) == len({r["index"] for r in plan["rows"]}) == 992
    for i, r in enumerate(plan["rows"]):
        rows.append((r["subject"],*(correctness_mask(v[24*i:24*(i+1)],keys[r["index"]]) for v in traces)))
    return histogram(rows), {"roster_sha256":pin,"publisher_sha256":digest(src),
                             "ordered_checkpoint_digests_sha256":checkpoints.hexdigest()}


def matched_hellaswag(folder):
    data = read(folder/"correctness-masks.json")
    results = read(folder/"results.json")
    assert data["roster_sha256"] == results["manifest_sha256"] and len(data["rows"]) == 992
    return histogram((s,int(a,16),int(b,16)) for s,a,b in data["rows"]), {
        "roster_sha256":data["roster_sha256"],"archive_masks_sha256":digest(folder/"correctness-masks.json")}


def historical_hellaswag(capsule, publisher):
    import pyarrow.parquet as pq
    approval = read(capsule/"label-approval.json")
    assert approval["authorize_confirmation_label_read"] and approval["independent_review"]
    assert digest(capsule/"prediction-export.json") == approval["predictions_sha256"]
    assert digest(capsule/"stage/staging-receipt.json") == approval["staging_sha256"]
    assert digest(capsule/"stage/roster.json") == read(capsule/"stage/staging-receipt.json")["cohort_file_sha256"]
    assert digest(publisher) == approval["source_sha256"]
    export = read(capsule/"prediction-export.json")
    assert tuple(map(tuple,export["orders"])) == ORDERS
    roster = read(capsule/"stage/roster.json")
    assert len(roster) == len(export["rows"]) == 6144
    def key(r):
        return r["ind"], r["source_id"]
    admitted = {key(r) for r in roster}
    assert len(admitted) == len(roster)
    labels = {}
    for r in pq.read_table(publisher,columns=["ind","source_id","label"]).to_pylist():
        pair = key(r)
        if pair in admitted:
            assert pair not in labels
            labels[pair] = int(r["label"])
    assert labels.keys() == admitted
    rows = []
    for r, record in zip(roster,export["rows"],strict=True):
        assert key(r) == key(record)
        left,right = record["displayed_digits"]
        rows.append((r["stratum"],*(correctness_mask([int(v)-1 for v in d],labels[key(r)]) for d in (left,right))))
    return histogram(rows), {"roster_file_sha256":digest(capsule/"stage/roster.json"),
                             "prediction_export_sha256":approval["predictions_sha256"],
                             "publisher_sha256":approval["source_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation-root","matched-hellaswag","historical-capsule","hellaswag-publisher","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    acquired = {"hellaswag-3b":matched_hellaswag(args.matched_hellaswag),
                "mmlu-3b":matched_confirmation(args.confirmation_root,"mmlu-validation"),
                "logiqa-3b":matched_confirmation(args.confirmation_root,"logiqa-test"),
                "hellaswag-1.5b-2b":historical_hellaswag(args.historical_capsule,args.hellaswag_publisher)}
    args.output.mkdir(exist_ok=True,parents=True)
    payload = {name:groups for name,(groups,_) in acquired.items()}
    with (args.output/"profiles.json.gz").open("wb") as dst:
        with GzipFile(filename="",mode="wb",fileobj=dst,mtime=0) as z:
            z.write((json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n").encode())
    provenance = {"format":"Each stratum lists [positive 24-bit mask, negative 24-bit mask, item count]. "
                  "Bit i denotes lexicographic display order i. Positive means only the first model is correct.",
                  "profile_sha256":digest(args.output/"profiles.json.gz"),
                  "sources":{name:meta for name,(_,meta) in acquired.items()}}
    (args.output/"source-receipts.json").write_text(json.dumps(provenance,indent=2)+"\n")
