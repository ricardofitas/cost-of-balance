"""Reproduce the prospective benchmark comparisons, bootstrap intervals and equal-call policy risks."""
import argparse
from collections import Counter
from fractions import Fraction as F
from functools import cache
import gzip
from hashlib import sha256
import importlib.util
import json
from math import lcm, sqrt
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pilot
from verify_decisions import decisions

# The previous study's block enumeration includes separate feature-projection checks for each item.
spec=importlib.util.spec_from_file_location("full_order_replay",HERE.parent/"eight-calls/replay.py")
previous=importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
B=50_000
M=20_000


def read(f):
    return json.loads((HERE/f).read_text())


def vector(a,b):
    assert 0<=a<2**24 and 0<=b<2**24 and not a&b
    return tuple(((a>>j)&1)-((b>>j)&1) for j in pilot.ALL)


@cache
def item(a,b):
    d=vector(a,b)
    prior=previous.item_risks(a,b)
    p,w=prior["8-precedence"],prior["8-full"]
    # Independent implementation used to audit the orientation and finite target.
    assert p==pilot.exact_risk(d,pilot.P) and w==pilot.exact_risk(d,pilot.W)
    r=pilot.exact_risk(d,pilot.R)
    return p,w,r,F(sum(d),24)


def seed(panel,operation):
    return int.from_bytes(sha256(("eight-pilot-analysis-v1|"+panel+"|"+operation).encode()).digest()[:8],"big")


def gain(value,base):
    return float(100*(1-value/base)) if base else None


def held_bootstrap(panel,pw):
    data=np.array(pw,dtype=float)
    assert len(data)>1 and np.all(data>=0)
    rng=np.random.Generator(np.random.PCG64(seed(panel,"paired-held-out-intervals")))
    draws=[]
    for lo in range(0,B,256):
        n=min(256,B-lo)
        rows=rng.integers(len(data),size=(n,len(data)))
        vals=data[rows].sum(axis=1)
        assert np.all(vals[:,0]>0)
        draws.extend(100*(1-vals[:,1]/vals[:,0]))
    low,low_panel,high_panel,high=np.percentile(draws,[.625,2.5,97.5,99.375])
    return {"replicates":B,"panel_95":[float(low_panel),float(high_panel)],
            "bonferroni_95_family":[float(low),float(high)]}


def exact_primary_probability(scores,tickets):
    """Convolve integer ticket counts over the fixed calibration questions."""
    counts=np.array([1],dtype=object)
    minimum=0
    for row in scores:
        frequencies=Counter(int(row[j]) for j in tickets)
        lo,hi=min(frequencies),max(frequencies)
        after=np.zeros(len(counts)+hi-lo,dtype=object)
        for value,count in frequencies.items():
            start=value-lo
            after[start:start+len(counts)]+=counts*count
        counts=after
        minimum+=lo
    ways=len(tickets)**len(scores)
    assert sum(counts)==ways
    positive=F(int(sum(counts[max(0,1-minimum):])),ways)
    zero=F(int(counts[-minimum]),ways) if 0<=-minimum<len(counts) else F(0)
    return positive,zero


def random_pilots(panel,method,cal_rows):
    sampler=pilot.weights(method)
    blocks=tuple(sampler)
    probs=[float(sampler[b]) for b in blocks]
    score_rows=[]
    for a,b in cal_rows:
        d=vector(a,b)
        scores=[pilot.sparse_score(method,block,{j:d[j] for j in block}) for block in blocks]
        assert sum(score*sampler[block] for block,score in zip(blocks,scores,strict=True))==item(a,b)[0]-item(a,b)[1]
        score_rows.append(scores)
    denom=lcm(*(x.denominator for row in score_rows for x in row))
    scores=np.array([[int(denom*x) for x in row] for row in score_rows],dtype=np.int64)
    assert len(cal_rows)*int(np.max(np.abs(scores)))<2**62
    block_ids={b:i for i,b in enumerate(blocks)}
    tickets=np.array([block_ids[b] for b in pilot.tickets(method)],dtype=int)
    rng=np.random.Generator(np.random.PCG64(seed(panel,method)))
    counts=ties=0
    for lo in range(0,M,256):
        n=min(256,M-lo)
        draws=tickets[rng.integers(len(tickets),size=(n,len(cal_rows)))]
        totals=scores[np.arange(len(cal_rows)),draws].sum(axis=1)
        counts+=int(np.count_nonzero(totals>0))
        ties+=int(np.count_nonzero(totals==0))
    prob=counts/M
    # Conditional variance of the pilot mean given its fixed calibration questions.
    moments=np.array([[float(x) for x in row] for row in score_rows])
    variance=np.maximum(0,(moments**2)@probs-(moments@probs)**2)
    exact={}
    if method=="residual-pilot":
        positive,zero=exact_primary_probability(scores,tickets)
        exact={"exact_williams_probability":str(positive),"exact_tie_probability":str(zero)}
    return {**exact,"repetitions":M,"williams_selections":counts,"exact_ties_in_simulation":ties,
            "selection_mcse":sqrt(prob*(1-prob)/M),"pilot_mean_sd":float(sqrt(float(variance.sum()))/len(cal_rows))}


def chosen_outcome(chosen,testp,testw):
    winner="tie" if testp==testw else ("full" if testw<testp else "precedence")
    value=testw if chosen=="full" else testp
    oracle=min(testp,testw)
    return {"choice":chosen,"matches_lower_held_out_mse":True if winner=="tie" else chosen==winner,
            "held_out_mse":str(value),"gain_vs_fixed_precedence_percent":gain(value,testp),
            "excess_vs_held_out_oracle_percent":-gain(value,oracle) if oracle else None}


def panel_result(panel,plan,raw,observed,registered):
    cal=[r[0] for r in plan["calibration"]]
    held=plan["held_out"]
    all_ids=cal+held
    assert len(raw)==len(all_ids)==len(set(all_ids))
    assert [row[0] for row in raw]==all_ids and len(raw)==pilot.SIZES[panel]
    pairs={g:(int(a,16),int(b,16)) for g,a,b in raw}
    for j,method in enumerate(pilot.METHODS,1):
        # Verify the earlier public choices against the subsequently released full profiles.
        checks=[]
        for row in plan["calibration"]:
            group=row[0]; d=vector(*pairs[group])
            checks.append([group,"".join("-0+"[d[k]+1] for k in row[j])])
        assert checks==observed[method]
    it={g:item(*pairs[g]) for g in all_ids}
    c=len(cal);h=len(held);n=c+h
    cal_sum=[sum(it[g][j] for g in cal) for j in range(4)]
    held_sum=[sum(it[g][j] for g in held) for j in range(4)]
    testp,testw=held_sum[0]/h**2,held_sum[1]/h**2
    fp=(cal_sum[0]+held_sum[0])/n**2
    fw=(cal_sum[1]+held_sum[1])/n**2
    winner="tie" if testp==testw else ("full" if testw<testp else "precedence")
    result={"calibration_questions":c,"held_out_questions":h,"deployment_calls_per_model":8*n,
            "reference_calls_both_models":48*n,"reference_gap":str((cal_sum[3]+held_sum[3])/n),
            "held_out_gap":str(held_sum[3]/h),"held_out_fixed_mse":{"precedence":str(testp),"full":str(testw)},
            "held_out_lower_mse":winner,"held_out_williams_gain_percent":gain(testw,testp),
            "held_out_retained_energy_share":str(F(5,11)*testw/testp) if testp else None,
            "held_out_williams_gain_intervals":held_bootstrap(panel,[it[g][:2] for g in held]),
            "all_item_fixed_mse":{"precedence":str(fp),"full":str(fw)},
            "calibration_exact_mean_signed_difference":str((cal_sum[0]-cal_sum[1])/c),
            "full_profile_calibration_diagnostic":chosen_outcome(pilot.choose(cal_sum[0]-cal_sum[1]),testp,testw),
            "pilots":{}}
    for method in pilot.METHODS:
        actual=chosen_outcome(registered[method]["chosen"],testp,testw)
        actual["registered_mean_score"]=registered[method]["mean_score"]
        mc=random_pilots(panel,method,[pairs[g] for g in cal])
        prob=F(mc["williams_selections"],M)
        future=(1-prob)*held_sum[0]+prob*held_sum[1]
        pilot_risk=cal_sum[0] if method=="precedence-pilot" else cal_sum[2]
        end=(pilot_risk+future)/n**2
        correct=1 if winner=="tie" else (prob if winner=="full" else 1-prob)
        attainable=(pilot_risk+min(held_sum[0],held_sum[1]))/n**2
        mc.update({"prob_selects_lower_held_out_mse":float(correct),
                   "best_possible_end_to_end_gain_vs_always_precedence_percent":gain(attainable,fp),
                   "calibration_mse_change_vs_precedence_percent":-gain(pilot_risk,cal_sum[0]),
                   "held_out_expected_regret_percent":-gain(future/h**2,min(testp,testw)),
                   "end_to_end_mse_at_mc_frequency":float(end),"gain_vs_always_precedence_percent":gain(end,fp),
                   "gain_vs_always_williams_percent":gain(end,fw),
                   "end_to_end_gain_mcse_pp":100*float(abs(held_sum[0]-held_sum[1])/n**2/fp)*mc["selection_mcse"],
                   "exact_calibration_sum_risk":str(pilot_risk)})
        if method=="residual-pilot":
            exact_prob=F(mc["exact_williams_probability"])
            exact_future=(1-exact_prob)*held_sum[0]+exact_prob*held_sum[1]
            exact_end=(pilot_risk+exact_future)/n**2
            mc["exact_end_to_end_mse"]=str(exact_end)
            mc["exact_gain_vs_always_precedence_percent"]=gain(exact_end,fp)
            mc["exact_gain_vs_always_williams_percent"]=gain(exact_end,fw)
            mc["exact_prob_selects_lower_held_out_mse"]=float(
                1 if winner=="tie" else (exact_prob if winner=="full" else 1-exact_prob))
        result["pilots"][method]={"registered":actual,"randomized":mc}
    return result


def analyze():
    design=read("registered-design.json")
    obs=read("pilot-observations.json"); reg=read("decisions.json")
    assert decisions(design,obs)==reg
    raw=(HERE/"profiles.json.gz").read_bytes()
    assert sha256(raw).hexdigest()==read("run-receipt.json")["profile_sha256"]
    panels=json.loads(gzip.decompress(raw))
    assert set(panels)==set(pilot.PANELS)
    return {p:panel_result(p,design["panels"][p],panels[p],obs[p],reg[p]) for p in pilot.PANELS}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--check",action="store_true")
    group.add_argument("--output",type=Path)
    a=parser.parse_args()
    result=analyze()
    if a.check:
        assert result==read("results.json")
        print("All registered choices, reference risks, intervals and equal-call policy comparisons matched.")
    elif a.output:
        a.output.write_text(json.dumps(result,indent=2)+"\n")
    else:
        print(json.dumps(result,indent=2))
