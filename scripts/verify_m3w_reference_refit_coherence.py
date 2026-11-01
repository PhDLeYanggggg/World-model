"""Original frozen-action verification with a nonintervening cost-coherence diagnostic."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_reference_cost_refit_readout as entry

run=entry.run
np=run.api.np


def coherence(prediction,allowed,scales):
    p,mask,s=np.asarray(prediction),np.asarray(allowed),np.asarray(scales)
    if (p.ndim!=3 or p.shape[1:]!=(7,5) or mask.shape!=p.shape[:2] or mask.dtype!=bool
            or s.shape!=(len(p),) or not np.isfinite(p).all() or (p<0).any()
            or not np.isfinite(s).all() or (s<=0).any()):raise ValueError('Finite native costs and causal legal masks/scales required')
    normal=s[:,None];violations={}
    for name,excess in dict(benefit_exceeds_reference=p[:,:,0]-p[:,:,2],
            negative_implied_candidate_ADE=p[:,:,0]-p[:,:,1]-p[:,:,2],
            easy_reference_exceeds_reference=p[:,:,3]-p[:,:,2],easy_harm_exceeds_harm=p[:,:,4]-p[:,:,1]).items():
        x=(excess/normal)[mask]
        violations[name]=dict(exact_positive_action_views=int((x>0).sum()),
            above1e_12_normalized_action_views=int((x>1e-12).sum()),
            max_positive_excess_over_causal_scale=float(np.maximum(x,0).max()) if len(x) else None)
    spread={}
    for name,k in (('reference',2),('easy_reference',3)):
        valid=mask.sum(1)>=2;lo=np.min(np.where(mask,p[:,:,k],np.inf),1);hi=np.max(np.where(mask,p[:,:,k],-np.inf),1)
        x=((hi-lo)/s)[valid]
        spread[name]=dict(rows_with_two_legal_actions=int(valid.sum()),
            rows_with_spread_above1e_12=int((x>1e-12).sum()),
            mean_spread_over_causal_scale=float(x.mean()) if len(x) else None,
            maximum_spread_over_causal_scale=float(x.max()) if len(x) else None)
    return dict(row_views=len(p),legal_action_views=int(mask.sum()),violations=violations,reference_spread=spread,
        exact_zero_reference_action_views=int(((p[:,:,2]==0)&mask).sum()),
        exact_zero_easy_reference_action_views=int(((p[:,:,3]==0)&mask).sum()),
        inference_only_no_targets=True,diagnostic_not_a_new_gate=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    destination=run.PUBLIC/'coherence_verification.json'
    if destination.exists():raise ValueError('Completed coherence verification immutable')
    fixed,binding=entry.corrected_evaluation();erratum=json.loads((run.PUBLIC/'execution_erratum.json').read_text())
    if binding!=erratum['binding']:raise ValueError('Only the pre-outcome scope erratum may execute')
    for path in (Path(__file__),ROOT/'scripts/run_m3w_reference_cost_refit_readout.py',run.PUBLIC/'execution_erratum.json'):
        run.parent.source.frozen.BASE.engine.committed(path)
    original=run.api.predict;cells={}
    def capture(state,causal):
        p=original(state,causal);before=run.api.parent.digest(p)
        context=state['identity']['context'];seed=state['seed'];arm=state['arm'];key=f'{context}_seed{seed}_{arm}'
        if key in cells:raise ValueError('Each frozen cost endpoint inferred once in full verification')
        cells[key]=dict(context=context,seed=seed,arm=arm,
            checkpoint_sha256=run.sha(run.checkpoint(context,seed,arm)),prediction_sha256=before,
            diagnostic=coherence(p,causal['allowed'],causal['scales']))
        if before!=run.api.parent.digest(p):raise AssertionError('Diagnostic must not alter a prediction byte')
        return p
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run.api.predict=capture
        try:fixed(True,args.resume)
        finally:run.api.predict=original
        if len(cells)!=36:raise ValueError('All36 frozen cost endpoints required')
        run.once(destination,dict(cells=cells,models=36,prediction_bytes_unchanged=True,
            executable_sha256=run.sha(Path(__file__)),registration_sha256=run.sha(run.REG),
            training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
            verification_sha256=run.sha(run.PUBLIC/'verification.json'),
            diagnostic_added_after_registration_not_a_new_primary=True,
            future_targets_read_by_diagnostic=False,thresholds_or_choices_changed=False,
            independent_confirmation=False,new_trajectory_training=False))


if __name__=='__main__':main()
