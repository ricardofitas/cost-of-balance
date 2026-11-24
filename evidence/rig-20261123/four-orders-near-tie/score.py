"""Paired finite-panel decision errors with checked convolution."""
from collections import Counter
from math import ceil
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from setup import ROOT, SOURCES, N, MARGIN, native, write, read
from common import QUEUE, OUTPUTS, GPU
from rank_risk import designs, replay
from evaluation import ORDERS

DESIGNS={key:designs[key] for key in ("williams","precedence-minimax","ordinary-cyclic")}
BOOTSTRAPS=4000
MSE_ONLY=("all-balanced","position-minimax")


def gaps_for(traces,y):
 gap=[]
 for j,label in enumerate(y):
    a,b=([int(ORDERS[k][v[j*24+k]]==label) for k in range(24)] for v in traces)
    gap.append([a[k]-b[k] for k in replay.DISPLAY_INDEX])
 return np.asarray(gap)


def item_pmfs(gap,blocks):
 totals=gap[:,np.asarray(blocks)].sum(axis=2)
 pmfs=np.asarray([np.bincount(t+4,minlength=9)/len(blocks) for t in totals])
 return pmfs,float(((totals/4-gap.mean(axis=1)[:,None])**2).mean()/len(gap))


def convolve_independent(pmfs):
 out=np.array([1.],dtype=np.longdouble)
 for row in pmfs: out=np.convolve(out,row.astype(np.longdouble))
 if abs(out.sum()-1)>1e-10: raise RuntimeError("Probability mass lost")
 return out


def probabilities(pmfs,weights,cutoff,full_lead,batch=64):
 n=len(pmfs)
 width=8*n+1
 fftlen=1<<(width-1).bit_length()
 unique, row_types=np.unique(pmfs,axis=0,return_inverse=True)
 selector=np.eye(len(unique))[row_types]
 f=np.fft.rfft(unique,fftlen)
 zeros=abs(f)<1e-13
 safe=np.where(zeros,1,f)
 logs=np.log(safe)
 for i in range(0,len(weights),batch):
    counts=weights[i:i+batch]@selector
    prod=np.exp(counts@logs)
    prod[counts@zeros.astype(float)>0]=0
    pmf=np.fft.irfft(prod,fftlen,axis=1)[:,:width]
    if (np.min(pmf)<-1e-9 or np.max(abs(pmf.sum(axis=1)-1))>1e-9):
        raise RuntimeError("FFT convolution is numerically unstable")
    below=pmf[:,:cutoff].sum(axis=1)
    yield from np.clip(np.where(full_lead[i:i+batch],below,1-below),0,1)


def metrics(gap,seed):
 n=len(gap)
 cutoff=ceil(n/5)+4*n
 full_counts=gap.sum(axis=1)
 full_lead=int(full_counts.sum())*5>=6*n
 rng=np.random.default_rng(seed)
 weights=rng.multinomial(n,np.full(n,1/n),size=BOOTSTRAPS).astype(float)
 bootstrap_decision=(weights@full_counts)*5>=6*n
 rows={}
 boot={}
 for key,blocks in DESIGNS.items():
    pmfs,mse=item_pmfs(gap,blocks)
    direct=convolve_independent(pmfs)
    if full_lead: error=float(direct[:cutoff].sum())
    else: error=float(direct[cutoff:].sum())
    check=next(probabilities(pmfs,np.ones((1,n)),cutoff,np.asarray([full_lead])))
    if abs(check-error)>1e-9: raise RuntimeError("Direct and FFT convolution disagree")
    boot[key]=np.fromiter(probabilities(pmfs,weights,cutoff,bootstrap_decision),float)
    rows[key]={"decision_error":error,"paired_gap_mse":mse}
 for key in MSE_ONLY:
    _,mse=item_pmfs(gap,designs[key])
    rows[key]={"paired_gap_mse":mse}
 diff=boot["precedence-minimax"]-boot["williams"]
 return {"items":n,"all_order_gap":float(full_counts.mean()/24),"all_order_meaningful_lead":full_lead,
         "designs":rows,"absolute_reduction":rows["precedence-minimax"]["decision_error"]-rows["williams"]["decision_error"],
         "directional_bootstrap_interval":np.quantile(diff,[.025,.975]).tolist(),
         "bootstrap_reference_label_flip_fraction":float((bootstrap_decision!=full_lead).mean())}


def development():
 result={}
 source=ROOT.parent/"autonomous/state"
 for i,panel in enumerate(("sciq-test","arc-easy")):
    rows=read(source/f"{panel}-roster.json")["items"][:N]
    pred=[sum([read(source/panel/f"model-{m}-{o:04d}.json")["displayed_choices"] for o in range(0,N,8)],[]) for m in range(2)]
    result[panel]=metrics(gaps_for(pred,[r["answer"] for r in rows]),413+i)
 write(OUTPUTS/"quatro-ordens-quase-empate-desenvolvimento.json",
       {"used_opened_rows_only":True,"minimum_absolute_decision_error_reduction":MARGIN,"panels":result})
 print({k:[v["all_order_gap"],v["absolute_reduction"],v["directional_bootstrap_interval"]] for k,v in result.items()},flush=True)


def official_labels(panel,rows):
 filename,_,sha=SOURCES[panel]
 if native.digest(Path(filename))!=sha: raise ValueError("Source digest mismatch")
 table=pq.read_table(filename,columns=["id","choices","answerKey"]).to_pylist()
 labels=[]
 for r in rows:
    source=table[r["index"]]
    if source["id"]!=r["id"]: raise ValueError("Scoring identity mismatch")
    labels.append(source["choices"]["label"].index(source["answerKey"]))
 return labels


def confirm():
 inputs={}
 for panel in SOURCES:
    done=read(ROOT/panel/"inference.json")
    rel=read(QUEUE/"state"/done["run_id"]/"release.json")
    if rel["status"]!="completed" or not rel["eligible_checkpoint"] or done["label_columns_read"]!=0:
        raise RuntimeError("Both inference releases required before reading labels")
    pin=done["manifest_sha256"]
    plan=native.frozen(panel,pin)
    pred=native.predict_only(panel,plan,pin)
    inputs[panel]=(done,plan["rows"],pred,rel)
 panels={}
 for i,(panel,(done,rows,pred,rel)) in enumerate(inputs.items()):
    panel_results=metrics(gaps_for(pred,official_labels(panel,rows)),2026112310+i)
    panel_results["run_id"]=done["run_id"]
    panel_results["source_sha256"]=SOURCES[panel][2]
    panel_results["roster_sha256"]=done["manifest_sha256"]
    panels[panel]=panel_results
 intervals=[r["directional_bootstrap_interval"] for r in panels.values()]
 verdict="pass" if all(a>MARGIN for a,b in intervals) else "rule-out" if any(b<MARGIN for a,b in intervals) else "inconclusive"
 write(OUTPUTS/"quatro-ordens-quase-empate-resultados.json",{"panels":panels,"verdict":verdict,"margin":MARGIN,
       "independent_numerical_check":"direct convolution vs FFT","bootstrap_draws":BOOTSTRAPS,"prior_holdouts_opened":0})

if __name__=="__main__":
 if sys.argv[1]=="dev": development()
 elif sys.argv[1]=="infer": native.inference(*sys.argv[2:])
 elif sys.argv[1]=="score": confirm()
 else: raise ValueError("Unknown action")
