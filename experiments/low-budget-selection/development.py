"""Reproduce the pre-transfer pilot dispersion comparison on the three published 3B panels."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pilot


def compare():
    raw=json.loads(gzip.decompress((HERE.parent/"eight-calls/profiles.json.gz").read_bytes()))
    result={}
    for panel in ("hellaswag-3b","mmlu-3b","logiqa-3b"):
        d=[]; counts=[]
        for group in raw[panel].values():
            for a,b,n in group:
                a,b=int(a,16),int(b,16)
                d.append([((a>>j)&1)-((b>>j)&1) for j in pilot.ALL])
                counts.append(n)
        profiles=np.array(d); freq=np.array(counts)/sum(counts)
        means=profiles.mean(axis=1)
        result[panel]={}
        for method in pilot.METHODS:
            law=pilot.weights(method)
            risks=sum(float(w)*(freq@(profiles[:,block].mean(axis=1)-means)**2) for block,w in law.items())
            scores=[]
            for block in law:
                if method=="precedence-pilot":
                    score=sum(float(pilot.EDGE[i,j])*(profiles[:,i]-profiles[:,j])**2
                              for i,j in pilot.combinations(block,2))
                else:
                    def sq(direction):
                        pos,neg=direction
                        return (profiles[:,pos].sum(axis=1)-profiles[:,neg].sum(axis=1))**2
                    pos=sq(pilot.G_DIR[block])/6 if block in pilot.G_DIR else 0
                    neg=sq(pilot.F_DIR[block])/15 if block in pilot.F_DIR else 0
                    score=(pos-neg)/(176*float(law[block]))
                scores.append(score)
            scores=np.array(scores).T
            weights=np.array(list(map(float,law.values())))
            mean=freq@(scores@weights)
            sd=np.sqrt(freq@((scores**2)@weights)-mean**2)
            result[panel][method]={"mean_signed_score":float(mean),"score_sd":float(sd),"mean_item_gap_mse":float(risks)}
        values=result[panel]
        assert abs(values["residual-pilot"]["mean_signed_score"]-values["precedence-pilot"]["mean_signed_score"])<1e-14
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    value=compare()
    if args.check:
        old=json.loads((HERE/"development.json").read_text())
        assert all(np.isclose(x,old[p][m][k],rtol=1e-12,atol=1e-15)
                   for p,v in value.items() for m,r in v.items() for k,x in r.items())
        print("Development-panel pilot risks and dispersions matched.")
    else:
        print(json.dumps(value,indent=2))
