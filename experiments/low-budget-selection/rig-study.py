"""Stage blind four-option benchmarks and run the pinned paired 3B decoder on the rig."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
OLD = Path(os.environ.get("RIG_PREVIOUS_EVALUATION_WORK", PROJECT.parent/"col/work"))
sys.path.insert(0, str(OLD/"confirmation-metadata"))
import transport
import confirmation
from common import GPU, NATIVE, QUEUE, digest, read, write
import pyarrow.parquet as pq


SIZES = {"arc-challenge":256, "openbookqa":256, "cosmosqa":256, "race":128}
PINS = read(ROOT/"source_pins.json")
FILES = {"arc-challenge":OLD/"confirmation-metadata/arc-challenge-validation.parquet",
         "openbookqa":OLD/"near-tie/openbookqa-validation.parquet",
         "race":OLD/"confirmation-metadata/race-validation.parquet","cosmosqa":ROOT/"cosmos-valid.csv"}
MODELS = {name:(str(PROJECT/"work/hellaswag-3b/weights"/file),rev,pin) for name,(file,rev,pin) in transport.PIN.items()}
for ns in (transport,confirmation):
    ns.ROOT = ROOT
    ns.PIN = MODELS
    ns.SOURCES = {k:(str(FILES[k]),p["revision"],p["sha256"]) for k,p in PINS.items()}
EXPERIMENT = ROOT.parent/"paper/experiments/low-budget-selection"
POSITIONAL = re.compile(r"\b(?:all|none|both|either|neither|one|two|three) of (?:the |these )?(?:above|below|preceding|following)\b|\b(?:both|either|neither) (?:options? |choices? |answers? )?[A-D1-4] (?:and|or|nor|&) [A-D1-4]\b|^\s*[A-D]\s*(?:and|or|&)\s*[A-D]\s*[.)]?\s*$",re.I)


def signature(s):
    return sha256(" ".join(s.casefold().split()).encode()).hexdigest()


def metadata(panel):
    file = FILES[panel]
    assert digest(file) == PINS[panel]["sha256"]
    if panel == "cosmosqa":
        with file.open(newline="", encoding="utf-8-sig") as handle:
            for i,raw in enumerate(csv.DictReader(handle)):
                # Only these fields enter the staging process.
                yield i,raw["id"],raw["context"],raw["question"],[raw["answer"+str(j)] for j in range(4)]
    elif panel == "race":
        rows = pq.read_table(file,columns=["example_id","article","question","options"]).to_pylist()
        for i,r in enumerate(rows):
            yield i,r["example_id"],r["article"],r["question"],r["options"]
    else:
        stem = "question" if panel=="arc-challenge" else "question_stem"
        for i,r in enumerate(pq.read_table(file,columns=["id",stem,"choices"]).to_pylist()):
            yield i,r["id"],"",r[stem],r["choices"]["text"]


def prepare(panel):
    count = SIZES[panel]
    transport.COUNT = count
    out = []
    for i,id_,context,question,choices in metadata(panel):
        q = (context.strip()+"\n"+question.strip()).strip()
        if not question.strip() or len(choices)!=4 or len({signature(a) for a in choices if a.strip()})!=4:
            continue
        if any(POSITIONAL.search(x) for x in choices) or len(q)+sum(map(len,choices))>1450:
            continue
        out.append(dict(index=i,id=str(id_),subject="all",question=q,choices=choices,
                        stem_sha256=signature(q),group_sha256=signature(context or question)))
    freq = Counter(x["stem_sha256"] for x in out)
    out = sorted((x for x in out if freq[x["stem_sha256"]]==1),
                 key=lambda r:sha256(("fourbench-eight-v1|"+panel+"|"+r["stem_sha256"]).encode()).digest())
    chosen, seen = [], set()
    for x in out:
        if x["group_sha256"] not in seen:
            chosen.append(x)
            seen.add(x["group_sha256"])
    print(panel,"independent candidate groups",len(chosen),flush=True)
    if len(chosen)<count:
        raise ValueError(f"{panel}: fewer than {count} eligible groups")
    # Give both tokenizers a reserve. Expand in the same frozen order if necessary.
    end = min(count+48,len(chosen))
    while True:
        write(ROOT/f"{panel}-candidates.json",chosen[:end])
        try:
            transport.prepare(panel)
        except RuntimeError as e:
            if "insufficient context-qualified items" not in str(e) or end==len(chosen):
                raise
            end = min(end+64,len(chosen))
            continue
        break
    manifest = read(ROOT/panel/"frozen.json")
    assert len(manifest["rows"]) == len({r["group_sha256"] for r in manifest["rows"]}) == count
    for model,entries in manifest["fixtures"].items():
        for entry in entries:
            lines=(ROOT/panel/model/entry["file"]).read_text().splitlines()[1:]
            assert len(lines)==192 and all(len(set(lines[o:o+24]))==24 for o in range(0,192,24))


def bindings():
    return {"protocol":EXPERIMENT/"protocol.md", "score":EXPERIMENT/"pilot.py", "stage":Path(__file__),
            "displays":ROOT.parent/"paper/four_option_budgets.py","samplers":ROOT.parent/"paper/certificates/four-budget-templates.json",
            "source_pins":ROOT/"source_pins.json","native_worker":Path(confirmation.__file__),
            "transport":Path(transport.__file__),"prompt":Path(transport.evaluation.__file__),"native_decoder":NATIVE,
            **{k:ROOT/k/"frozen.json" for k in PINS}}


def prepare_job(run_id):
    data={"gpu":GPU,"models":{n:list(v[1:]) for n,v in MODELS.items()},"sha256":{k:digest(v) for k,v in bindings().items()}}
    commit=ROOT/"commitment.json"
    if commit.exists():
        assert read(commit)==data
    else:
        write(commit,data)
    pin=digest(commit)
    job={"run_id":run_id, "campaign":"evaluation", "cwd":str(ROOT),
         "argv":[str(OLD/"extended/run"),str(Path(__file__).resolve()),"infer",pin],
         "max_seconds":7200,"protocol":f"{EXPERIMENT/'protocol.md'} sha256={data['sha256']['protocol']} study_sha256={pin}",
         "checkpoint":str(ROOT/"inference.json"),"gpu_count":1,"eligible_gpu_uuids":[GPU]}
    write(ROOT/(run_id+".json"),job)
    print(pin, flush=True)


def infer(pin):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==GPU and os.environ.get("RIG_GPU_RUN_ID")
    assert digest(ROOT/"commitment.json")==pin
    pins=read(ROOT/"commitment.json")["sha256"]
    assert {k:digest(v) for k,v in bindings().items()}==pins
    for panel in PINS:
        confirmation.COUNT=SIZES[panel]
        frozen=read(ROOT/panel/"frozen.json")
        confirmation.inference(panel,pins[panel])
        confirmation.predict_only(panel,frozen,pins[panel])
    write(ROOT/"inference.json",{"study":pin,"run_id":os.environ["RIG_GPU_RUN_ID"],"panels":list(PINS),
          "reference_calls":48*sum(SIZES.values()),"labels_used_for_inference":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["stage","job","infer"])
    parser.add_argument("value")
    args=parser.parse_args()
    if args.action=="stage":
        prepare(args.value)
    elif args.action=="job":
        prepare_job(args.value)
    else:
        infer(args.value)
