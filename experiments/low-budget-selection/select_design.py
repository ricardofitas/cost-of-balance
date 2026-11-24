"""Draw eight orders on new labeled calibration questions, then choose a deployment sampler."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pilot
from four_option_budgets import ORDERS


def evaluate(request,action):
    name=request["benchmark"]
    method=request.get("method","residual-pilot")
    items=request["items"]
    ids=[q["id"] for q in items]
    if not isinstance(name,str) or not name or method not in pilot.METHODS:
        raise ValueError("Supply a benchmark name and a supported pilot method.")
    if not ids or any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
        raise ValueError("Supply distinct nonempty question IDs.")
    blocks=[pilot.chosen_block(method,name,i) for i in ids]
    if action=="orders":
        return {"benchmark":name,"method":method,"items":[{"id":i,"display_orders":[ORDERS[k] for k in b]}
                                                           for i,b in zip(ids,blocks,strict=True)]}
    total=Fraction(0)
    for q,b in zip(items,blocks,strict=True):
        vals=q["correctness_differences"]
        if len(vals)!=8 or any(type(x) is not int or x not in (-1,0,1) for x in vals):
            raise ValueError("Each item needs eight integer correctness differences in its assigned display order.")
        total+=pilot.sparse_score(method,b,dict(zip(b,vals,strict=True)))
    return {"benchmark":name,"method":method,"questions":len(ids),"calls_per_model":8*len(ids),
            "mean_signed_item_risk":str(total/len(ids)),"deployment_sampler":pilot.choose(total)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["orders","choose"])
    parser.add_argument("input",type=Path)
    args=parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.input.read_text()),args.action),indent=2))
