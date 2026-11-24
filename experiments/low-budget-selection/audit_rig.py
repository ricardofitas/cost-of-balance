"""Independently rebuild the released contrast masks from the publisher rows and raw decoder shards."""
import argparse
import csv
import gzip
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ORDERS=list(permutations(range(4)))


def digest(f):
    return sha256(f.read_bytes()).hexdigest()


def sig(s):
    return sha256(" ".join(s.casefold().split()).encode()).hexdigest()


def audit(root,files):
    published=json.loads(gzip.decompress((HERE/"profiles.json.gz").read_bytes()))
    pins=json.loads((HERE/"source-pins.json").read_text())
    output={}
    for panel,path in files.items():
        assert digest(path)==pins[panel]["sha256"]
        if panel=="cosmosqa":
            with path.open(newline="",encoding="utf-8-sig") as f:
                source=list(csv.DictReader(f))
        else:
            source=pq.read_table(path).to_pylist()
        folder=root/panel
        manifest=json.loads((folder/"frozen.json").read_text())
        pin=digest(folder/"frozen.json")
        targets=[]
        keys=[]
        for row in manifest["rows"]:
            s=source[row["index"]]
            if panel=="cosmosqa":
                id_=s["id"]; grp=s["context"]; question=grp.strip()+"\n"+s["question"].strip(); gold=int(s["label"])
            elif panel=="race":
                id_=s["example_id"]; grp=s["article"]; question=grp.strip()+"\n"+s["question"].strip(); gold=ord(s["answer"])-ord("A")
            else:
                id_=s["id"]
                question=s["question"] if panel=="arc-challenge" else s["question_stem"]
                grp=question
                gold=next(j for j,label in enumerate(s["choices"]["label"]) if s["answerKey"]==label)
            assert str(id_)==row["id"] and sig(grp)==row["group_sha256"] and sig(question)==row["stem_sha256"]
            targets.append([order.index(gold) for order in ORDERS])
            keys.append(sig(grp))
        decoded=[]
        for model in ("qwen","smol"):
            shards=[]
            entries=manifest["fixtures"][model]
            for i,entry in enumerate(entries):
                assert digest(folder/model/entry["file"])==entry["sha256"]
                saved=json.loads((folder/model/f"pred-{i:04d}.json").read_text())
                assert saved["fixture"]==entry["sha256"] and saved["roster"]==pin
                shards.append(np.array(saved["choices"],dtype=int).reshape(8,24))
            arr=np.concatenate(shards)
            assert arr.shape==(len(targets),24) and np.all((arr>=0)&(arr<=3))
            decoded.append((arr==np.array(targets)).astype(np.int64))
        delta=decoded[0]-decoded[1]
        bit=2**np.arange(24,dtype=np.int64)
        pos=(delta==1)@bit; neg=(delta==-1)@bit
        computed={k:(int(a),int(b)) for k,a,b in zip(keys,pos,neg,strict=True)}
        expected={k:(int(a,16),int(b,16)) for k,a,b in published[panel]}
        assert len(expected)==len(keys)==len(set(keys)) and computed==expected
        output[panel]={"questions":len(keys),"matched_model_item_orders":48*len(keys),
                       "all_signed_masks_equal":True,"full_reference_gap_numerator":int(delta.sum())}
    return output


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--rig-dir",type=Path,required=True)
    a.add_argument("--source-dir",type=Path,required=True,help="original evaluation working directory with confirmation-metadata and near-tie")
    args=a.parse_args()
    src={"arc-challenge":args.source_dir/"confirmation-metadata/arc-challenge-validation.parquet",
         "openbookqa":args.source_dir/"near-tie/openbookqa-validation.parquet","race":args.source_dir/"confirmation-metadata/race-validation.parquet",
         "cosmosqa":args.rig_dir/"cosmos-valid.csv"}
    print(json.dumps(audit(args.rig_dir,src),indent=2))
