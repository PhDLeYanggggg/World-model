"""Read frozen inner-locality scores without fitting or choosing a policy."""
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_harm_tail_crossfit as run
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); done=run.checked_training(identity)
    run.parent.previous.require_committed(run.PUBLIC/'prediction_freeze.json')
    assert json.loads((run.PUBLIC/'prediction_freeze.json').read_text())['manifest']==run.artifact(run.PRIVATE/'training_complete.json')
    heads={}
    for ref in done['heads']:
        row=json.loads((ROOT/ref['path']).read_text()); d=row['input']
        heads[(d['group']['group'],d['pair'],d['held'])]=(ref,row)
    output=[]
    for g,data,pairs in run.parent.contexts(identity['parent']['parent']['parent']):
        name=g['group']; bi,ci=pairs['B']['ids'],pairs['C']['ids']; sites=data['sites'][bi]
        for pair in cfg['pairs']:
            path=run.PUBLIC/'groups'/(name+'_'+pair+'.json'); receipt=run.PRIVATE/'evaluation'/path.name
            if path.exists():
                assert run.artifact(path)==json.loads(receipt.read_text()); output.append(run.artifact(path)); continue
            run.beat('held_readout',group=name,pair=pair)
            bx,env,by,*_=run.parent.pair_inputs(g,data,pairs,pair); folds=[]
            for held in sorted(set(sites)):
                ref,h=heads[(name,pair,held)]; tr,te,pr,cut,y,masks=run.fold_inputs(bx,env,by,data['baseline_ade'][bi,1],sites,held)
                assert cut==h['input']['easy_cut']
                with np.load(ROOT/h['artifacts']['scores']['path'],allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); score=z['scores'].copy()
                target=run.diagnostic.event_targets(by[te],data['baseline_ade'][bi[te],1],cut)
                training=json.loads((ROOT/h['artifacts']['diagnosis']['path']).read_text()); edges=training['edges']
                held_metrics={k:run.diagnostic.summarize(score,target,env[te],sites[te],edges,subset=mask)
                    for k,mask in (('all',None),('envelope_positive',env[te]>0))}
                folds.append(dict(held=held,head=ref,training=training,held_metrics=held_metrics,easy_cut=cut))
            directory=run.parent.PRIVATE/'heads'/(name+'_'+pair+'_mean')
            original,state=run.parent.restore(directory); pr=state['preprocess']
            original_B=run.parent.method.predict(original,bx,env,pr)
            edges=run.training_edges(original_B,env,pr,cfg)
            with np.load(directory/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci); original_C=z['scores'].copy()
            source=run.parent.previous.PRIVATE/'source'/(name+'_'+pair); sr=json.loads((source/'receipt.json').read_text())
            assert run.artifact(source/'labels.npz')==sr['arrays']['labels']
            with np.load(source/'labels.npz',allow_pickle=False) as z:
                target=run.parent.method.event_targets(z['cv'],z['reference'],z['candidate'],g['easy_cut'])
            original_metrics={}
            for role,pred,y,e,s in (('B',original_B,by,env,sites),('C',original_C,target,pairs['C'][pair][1],data['sites'][ci])):
                original_metrics[role]={site:{k:run.diagnostic.summarize(pred[ss],y[ss],e[ss],s[ss],edges,subset=mask)
                    for k,mask in (('all',None),('envelope_positive',e[ss]>0))}
                    for site in sorted(set(s)) for ss in [s==site]}
            row=dict(identity=identity,group=name,pair=pair,producer=g['producer'],controller=g['controller'],
                seed=int(name.split('_seed')[1].split('_')[0]),folds=folds,original_metrics=original_metrics,
                original_easy_cut=g['easy_cut'],original_B_edges=edges,
                original_model=run.artifact(directory/'checkpoint.pt'),original_C_scores=run.artifact(directory/'scores.npz'),
                original_C_labels=run.artifact(source/'labels.npz'),
                result_source='fresh_run_inner_held_metrics_and_B_bins_cached_verified_original_C_inputs',
                original_not_target_matched_to_inner=True,policy_changed=False)
            run.immutable_json(path,row); run.immutable_json(receipt,run.artifact(path)); output.append(run.artifact(path))
    assert len(output)==36
    run.immutable_json(run.PUBLIC/'completion_checks.json',dict(identity=identity,groups=output,
        heads=144,all_passed=True,policy_changed=False,source_binding=run.artifact(Path(__file__))))
    run.beat('readout_complete',groups=len(output))


if __name__=='__main__':
    with (run.PRIVATE/'evaluate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
