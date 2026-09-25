"""Quantify already-read zero-CV harms without altering policies or thresholds."""
import json
import os
from pathlib import Path
import platform
import sys

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(k,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_hurdle_risk as run
from scripts.run_m3w_native_forecast import immutable_json,array_hash
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_european_conditional_risk import pointwise_rule
from src.evaluation.m3w_native_metrics import native_errors


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    reg,previous,data,identity,originals=run.load();old=run.old
    verification=json.loads((run.PUBLIC/'verification.json').read_text())
    if not verification['all_passed'] or verification['identity']!=identity:raise ValueError('Complete readout required')
    primary=json.loads((run.PUBLIC/'analysis.json').read_text());rows=[]
    allzero=np.flatnonzero(np.isfinite(data['baseline_ade'][:,1])&(data['baseline_ade'][:,1]==0))
    for name,candidate,fold,seed,design in run.jobs(previous,data,identity):
        ids=np.intersect1d(design['held_ids'],allzero)
        if not len(ids):continue
        if candidate=='neural':
            final=identity['geometric_identity']['old_identity']['producer_identity']['frozen_final_producers'][f'single{fold}_seed{seed}']
            p=old.read_predictions(final['prediction'],ids).astype(float)+data['origin'][ids,None]
        else:p=baseline_numpy(data['history'][ids],3)
        ade,_=native_errors(p,data['target_eval'][ids],data['valid'][ids],np.ones(len(ids)))
        moving=np.linalg.norm(np.diff(data['history'][ids],axis=1),axis=2).sum(1)>0
        u=old.scores(originals[name+'_utility'],ids)
        train_cv=data['baseline_ade'][design['train_ids'],1]
        for event in reg['events']:
            for arm in reg['variants']:
                r=(originals[name+'_'+event] if arm=='old' else
                    run.geo.checked(name+'_'+event,identity['geometric_identity']) if arm=='envelope' else
                    run.checked(name+'_'+event+'_'+arm,identity))
                risk=old.scores(r,ids)
                switch=pointwise_rule(u[:,0]-u[:,1],risk,moving,budget=reg['risk_budget'],support_available=True)
                added=ade[switch];key=name+'_'+event+'_'+arm
                if int((added>0).sum())!=primary['views'][key]['zero_CV']['harmed_rows']:raise ValueError('Zero-reference decision mismatch')
                rows.append(dict(view=key,excluded_zero_rows=len(ids),fitting_zero_rows=int((train_cv==0).sum()),
                    moving_zero_rows=int(moving.sum()),switched_zero_rows=int(switch.sum()),
                    harmed_zero_rows=int((added>0).sum()),
                    selected_error_min=float(added.min()) if len(added) else None,
                    selected_error_max=float(added.max()) if len(added) else None,
                    selected_error_mean=float(added.mean()) if len(added) else None,
                    zero_rows_sha256=array_hash(ids)))
    run.assert_identity(identity)
    result=dict(result_source='fresh_run_posthoc_zero_reference_audit_no_refit',
        parent_analysis_sha256=old.digest(run.PUBLIC/'analysis.json'),script_sha256=old.digest(Path(__file__)),
        distinct_zero_rows=len(allzero),zero_localities=sorted(set(data['sites'][allzero])),
        inference_uses_future=False,threshold_changed=False,views=rows)
    immutable_json(run.PUBLIC/'zero_reference_audit.json',result)
    harms=[r for r in rows if r['view'].startswith('neural_') and r['view'].endswith('_hurdle') and r['harmed_zero_rows']]
    print(json.dumps(dict(distinct_zero_rows=len(allzero),neural_hurdle_affected_views=len(harms),
        minimum_added_ADE=min(r['selected_error_min'] for r in harms),maximum_added_ADE=max(r['selected_error_max'] for r in harms))))


if __name__=='__main__':main()
