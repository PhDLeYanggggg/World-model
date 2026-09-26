"""Post-freeze easy-membership decomposition before selecting a repair mechanism."""
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_support_fractional as parent
from src.evaluation.m3w_easy_membership_diagnosis import decompose,summary
import numpy as np
import torch
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_frozen_harm_readout_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_frozen_harm_readout_v1'


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=parent.registration(); parent.checked_training(identity)
    verification=json.loads((parent.PUBLIC/'verification.json').read_text())
    for f,h in verification['artifacts'].items(): assert parent.digest(parent.PUBLIC/f)==h
    for f,h in verification['source_bindings'].items(): assert parent.digest(ROOT/f)==h
    bindings={f:parent.digest(ROOT/f) for f in ('scripts/diagnose_m3w_european_easy_membership.py',
        'src/evaluation/m3w_easy_membership_diagnosis.py','tests/test_m3w_easy_membership_diagnosis.py')}
    rows=[]
    for g,data,pairs in parent.base.contexts(identity['source']):
        bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            out=PUBLIC/'diagnosis_groups'/(g['group']+'_'+pair+'.json')
            if out.exists():
                row=json.loads(out.read_text()); assert row['bindings']==bindings
                rows.append(row); continue
            bx,env,by,*_=parent.base.pair_inputs(g,data,pairs,pair); folds=[]
            old=json.loads((parent.PUBLIC/'groups'/(g['group']+'_'+pair+'.json')).read_text())
            for held in sorted(set(sites)):
                tr,te,pr,cut,y,_=parent.previous.fold_inputs(bx,env,by,cv,sites,held)
                target=parent.previous.diagnostic.event_targets(by[te],cv[te],cut)
                oldfold=next(f for f in old['folds'] if f['held']==held)
                assert oldfold['target_sha256']==parent.array_hash(target)
                predictions=[]
                for home in (parent.previous.PRIVATE,parent.PRIVATE):
                    with np.load(home/'heads'/(g['group']+'_'+pair+'_'+held)/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); predictions.append(z['scores'].copy())
                d=decompose(*predictions,target,cv[te],env[te],cut)
                previous=oldfold['metrics']
                np.testing.assert_allclose(d['excess_MSE'],previous['fractional']['envelope_positive']['harm_MSE']-
                    previous['mean']['envelope_positive']['harm_MSE'],rtol=1e-10,atol=1e-10)
                folds.append(dict(held=held,easy_cut=cut,diagnosis=d,target_sha256=parent.array_hash(target)))
            row=dict(group=g['group'],pair=pair,folds=folds,bindings=bindings,result_source='fresh_run_diagnostic_cached_verified_predictions')
            parent.immutable_json(out,row); rows.append(row)
            print(json.dumps(dict(state='membership_diagnosis',completed=len(rows),group=g['group'],pair=pair)),flush=True)
    assert len(rows)==36
    result=dict(parent_verification=parent.artifact(parent.PUBLIC/'verification.json'),bindings=bindings,
        summary=summary(rows),groups=[parent.artifact(p) for p in sorted((PUBLIC/'diagnosis_groups').glob('*.json'))],
        previous_goal_turn='progress_144_trained_fractional_primary_gate_failed',new_training=False,
        independent_confirmation=False,policy_changed=False,stage5c_executed=False,smc_enabled=False)
    parent.immutable_json(PUBLIC/'membership_diagnosis.json',result)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'diagnosis.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
