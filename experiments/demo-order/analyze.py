"""Replay the original call equivalents and the registered paired demonstration risks."""
from fractions import Fraction as F
from pathlib import Path
import argparse, gzip, importlib.util, json
from math import ceil
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'eight-calls'
NEW=HERE.parent/'low-budget-selection'
spec=importlib.util.spec_from_file_location('order_risks',OLD/'replay.py')
engine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

def read(path):
    return json.loads(path.read_text())

def minimum_calls(variance, target):
    if not variance: return 1
    k=ceil(24*variance/(variance+23*target))
    assert (24-k)*variance<=23*k*target
    assert k==1 or (25-k)*variance>23*(k-1)*target
    return k

def costs():
    rows={}
    for name,panel in read(OLD/'results.json').items():
        if not name.endswith('-3b'): continue
        rows[name]={k:F(v) for k,v in panel['mse'].items()}
    plan=read(NEW/'registered-design.json')['panels']
    for name,items in json.loads(gzip.decompress((NEW/'profiles.json.gz').read_bytes())).items():
        held=set(plan[name]['held_out'])
        risks=[engine.item_risks(int(a,16),int(b,16)) for key,a,b in items if key in held]
        assert len(risks)==len(held)
        rows[name]={k:sum(v[k] for v in risks)/len(held)**2 for k in ('8-precedence','8-rotation-reversal','8-unrestricted')}
    return {name:{'uniform_calls':minimum_calls(v['8-unrestricted']*F(23,2),v['8-precedence']),
                  'gain_vs_rotations_percent':float(100*(1-v['8-precedence']/v['8-rotation-reversal']))}
            for name,v in rows.items()}

def summarize(task,panels):
    sums=[]; totals=[0,0]; profiles=[]
    for group in panels:
        q,s,ys=group['qwen'],group['smol'],group['labels']
        assert len(q)==len(s)==24 and len(ys)==8 and all(len(r)==8 for r in q+s)
        vs=[]
        for j,y in enumerate(ys):
            a=b=0
            for h in range(24):
                x,z=q[h][j]==y,s[h][j]==y
                totals[0]+=x;totals[1]+=z
                a|=int(x and not z)<<h;b|=int(z and not x)<<h
            vs.append(engine.item_risks(a,b));profiles.append(a|b)
        sums.append([sum(v[k] for v in vs) for k in ('8-precedence','8-rotation-reversal','8-unrestricted')])
    assert len(sums)==8
    p,r,u=map(sum,zip(*sums))
    rng=np.random.default_rng(112304)
    draws=np.array(sums,dtype=float)[rng.integers(8,size=(50000,8))].sum(axis=1)
    assert r and np.all(draws[:,1]>0)
    return dict(questions=64,quartets=8,full_order_accuracy=[c/1536 for c in totals],
                precedence_mse=str(p/4096),rotation_mse=str(r/4096),
                primary_gain_percent=float(100*(1-p/r)),
                quartet_95=np.percentile(100*(1-draws[:,0]/draws[:,1]),[2.5,97.5]).tolist(),
                uniform_calls=minimum_calls(u*F(23,2),p))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--costs',action='store_true');args=parser.parse_args()
    result=costs() if args.costs else {t:summarize(t,v) for t,v in read(HERE/'responses.json').items()}
    print(json.dumps(result,indent=2))
