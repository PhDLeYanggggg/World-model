"""Verify new label interface on every existing nested fitting/held source view."""
import json
from pathlib import Path
import platform
import sys
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_log_cost import load
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_prefix_cost_targets import causal_prefix_disagreement, supervised_prefix_costs


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,data,views,predictions,prior,refs,identity=load()
    public=ROOT/cfg['reports']; evidence=public/'horizon_cost_support.json'
    report=json.loads(evidence.read_text())
    assert report['analysis_sha256']==file_digest(public/'analysis.json')
    rows=[]
    for key,meta in views.items():
        for role,path in (('fitting',meta['inputs_path']),('held_source',predictions[key]['path'])):
            with np.load(ROOT/path,allow_pickle=False) as z:
                ids,p=z['ids'].copy(),z['prediction'].copy()
            np.testing.assert_array_equal(ids,np.flatnonzero((data['sites']!=meta['outer_site']) if role=='fitting' else (data['sites']==meta['outer_site'])))
            b=data['geometry'][ids,332:356].reshape(-1,12,2)
            y=read_arrays(data,ids,'target'); valid=read_arrays(data,ids,'valid'); scale=data['scale'][ids]
            result=supervised_prefix_costs(b,p,y,valid,scale)
            full=valid.all(1)
            assert np.array_equal(result['available'][:,-1],full)
            assert np.isnan(result['costs'][~result['available']]).all()
            assert np.all(result['costs'].sum(-1)[result['available']]<=result['disagreement'][result['available']]+1e-8)
            for k in range(12):
                known=valid[:,:k+1].all(1)
                # Separate scalar-prefix reductions, not the builder's cumulative-sum path.
                be=np.sqrt(np.square(b[known,:k+1].astype(float)-y[known,:k+1]).sum(-1)).mean(1)*scale[known]
                ne=np.sqrt(np.square(p[known,:k+1].astype(float)-y[known,:k+1]).sum(-1)).mean(1)*scale[known]
                expected=np.column_stack((np.maximum(be-ne,0),np.maximum(ne-be,0)))
                np.testing.assert_allclose(result['costs'][known,k],expected,rtol=1e-10,atol=1e-8)
            if role=='fitting':
                with np.load(ROOT/meta['targets_path'],allow_pickle=False) as z:
                    np.testing.assert_array_equal(ids,z['ids'])
                    expected=np.column_stack((z['benefit'][full],z['harm'][full]))
                np.testing.assert_allclose(result['costs'][full,-1],expected,rtol=1e-10,atol=1e-8)
            ix=np.unique(np.linspace(0,len(ids)-1,min(17,len(ids)),dtype=int))
            changed=supervised_prefix_costs(b[ix],p[ix],np.where(valid[ix,...,None],y[ix]+37,y[ix]),valid[ix],scale[ix])
            np.testing.assert_array_equal(changed['disagreement'],result['disagreement'][ix])
            np.testing.assert_array_equal(causal_prefix_disagreement(b[ix],p[ix],scale[ix]),result['disagreement'][ix])
            rows.append(dict(view=key,role=role,rows=len(ids),complete_rows=int(full.sum()),
                supported_prefix_labels=int(result['available'].sum()),counterfactual_queries=len(ix),
                arrays_sha256=array_hash(ids,result['costs'],result['available'],result['disagreement'])))
    out=dict(result_source='fresh_real_array_validation_no_training',
        diagnosis_sha256=file_digest(evidence),analysis_sha256=report['analysis_sha256'],
        code_sha256=file_digest(Path(__file__)),
        module_sha256=file_digest(ROOT/'src/world_model/m3w_prefix_cost_targets.py'),
        tests_sha256=file_digest(ROOT/'tests/test_m3w_prefix_cost_targets.py'),
        records=rows,all_checks_passed=True,scalar_terminal_targets_preserved=True,
        counterfactual_queries=sum(r['counterfactual_queries'] for r in rows),
        prefix_reductions_checked=12*len(rows),
        label_export_materialized=False,labels_never_inference_features=True,
        future_mask_never_inference_input=True,changed_decisions=False,new_training=False,
        independent_confirmation=False,deployment=False,stage5c_executed=False,smc_enabled=False)
    assert_current(identity); immutable_json(public/'prefix_target_verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))


if __name__=='__main__': main()
