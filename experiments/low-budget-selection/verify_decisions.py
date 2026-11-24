"""Replay the registered choices using only the eight revealed orders on each calibration item."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pilot


def decisions(plan, observations):
    out={}
    seen=set()
    assert set(observations)==set(pilot.PANELS)
    for panel,design in plan["panels"].items():
        calibration=design["calibration"]
        ids=[entry[0] for entry in calibration]
        assert len(calibration)*4==pilot.SIZES[panel]
        assert len(set(ids)|set(design["held_out"]))==pilot.SIZES[panel]
        assert len(set(ids)&set(design["held_out"]))==0
        groups=set(ids)|set(design["held_out"])
        assert seen.isdisjoint(groups)
        seen.update(groups)
        assert (ids,design["held_out"])==pilot.roles(panel,[*ids,*design["held_out"]])
        assert set(observations[panel])==set(pilot.METHODS)
        out[panel]={}
        for index,method in enumerate(pilot.METHODS,1):
            vals=observations[panel][method]
            assert [x[0] for x in vals]==ids
            total=F(0)
            for entry,(group,raw) in zip(calibration,vals,strict=True):
                b=tuple(entry[index])
                assert b==pilot.chosen_block(method,panel,group)
                assert len(raw)==len(b)==8 and all(c in "-0+" for c in raw)
                d={j:"-0+".index(c)-1 for j,c in zip(b,raw,strict=True)}
                total+=pilot.sparse_score(method,b,d)
            out[panel][method]={"pilot_questions":len(ids),"calls_per_model":8*len(ids),
                                "mean_score":str(total/len(ids)), "chosen":pilot.choose(total)}
    return out


if __name__=="__main__":
    def read(name):
        return json.loads((HERE/name).read_text())
    actual=decisions(read("registered-design.json"),read("pilot-observations.json"))
    assert actual==read("decisions.json")
    print("Both sparse choices reproduced for every panel without held-out outcomes.")
