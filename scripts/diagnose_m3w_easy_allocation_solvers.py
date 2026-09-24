"""Causal-only reproduction of fixed failed solver cases; no rule changes."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import src.world_model.m3w_easy_allocation as core
from scripts.run_m3w_easy_allocation import load, causal_view, query, CONTEXT_KEYS
from scripts.run_m3w_native_forecast import immutable_json, file_digest


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    pack=load();cfg,data,context,a,tr,eq,identity=pack
    root=ROOT/cfg['output'];manifest=json.loads((root/'decisions_complete.json').read_text())
    selected=[];counts={};all_failed=0
    for r in manifest['receipts']:
        v=json.loads((ROOT/r['path']).read_text())
        for q in v['queries']:
            failed=[k for k in ('population_optimal','unary_optimal','joint_optimal') if not q[k]]
            if not failed:continue
            all_failed+=1;key=(v['action'],tuple(failed))
            counts[key]=counts.get(key,0)+1
            if counts[key]<=2:selected.append((v,q))
    result=[];current=None
    for record,old in selected:
        key,action=record['view'],record['action']
        if current!=(key,action):
            values,pred,scale,cut=causal_view(pack,key,action);current=(key,action)
            index=np.full(len(data['sites']),-1,np.int64);index[values['ids']]=np.arange(len(values['ids']))
        rec=next(r for r in context['records'] if r['recording']==record['recording'])
        with np.load(ROOT/rec['cache']['path'],allow_pickle=False) as z:c={k:z[k].copy() for k in CONTEXT_KEYS}
        captured=[];original=core.solve_control
        def observe(p,**kwargs):
            r=original(p,**kwargs)
            captured.append(dict(objective=kwargs['objective_kind'],reason=r['reason'],
                agents=len(p.supported),eligible=int(p.supported.sum()),
                selected=int(r['switch'].sum()),count_requested=kwargs.get('exact_interventions'),
                risk_budget=p.max_mean_predicted_harm,numerical=r['numerical'],
                predicted_constraints_satisfied=r['predicted_constraints_satisfied']))
            return r
        core.solve_control=observe
        try:
            rows=np.flatnonzero(c['context_frame_ids']==old['frame'])
            ids,bits,repeated=query(rows,c,index,values,pred,data,scale,cut,action,cfg)
        finally:core.solve_control=original
        assert repeated=={k:v for k,v in old.items() if k!='frame'}
        with np.load(ROOT/record['path'],allow_pickle=False) as z:
            mapping={int(v):i for i,v in enumerate(z['ids'])}
            np.testing.assert_array_equal(bits,z['choices'][[mapping[int(i)] for i in ids]])
        result.append(dict(view=key,action=action,recording=record['recording'],frame=old['frame'],calls=captured))
    out=dict(result_source='fresh_causal_failure_reproduction_not_outcome_tuning',failed_queries=all_failed,
        failures_by_pattern={action+'__'+','.join(pattern):v for (action,pattern),v in counts.items()},
        reproduced_cases=len(result),records=result,policy_changed=False,future_outcome_arrays_loaded=False,
        code_sha256=file_digest(Path(__file__)))
    immutable_json(ROOT/cfg['reports']/'solver_diagnosis.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
