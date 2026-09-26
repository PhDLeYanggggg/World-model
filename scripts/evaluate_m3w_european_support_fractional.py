"""Paired source-development readout after prediction freeze; no model selection."""
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_support_fractional as run
import numpy as np
import torch


def measure(pred,y,env,sites,edges):
    out={}
    for key,mask in (('all',np.ones(len(y),bool)),('envelope_positive',env>0)):
        value=run.previous.diagnostic.summarize(pred,y,env,sites,edges,subset=mask)
        known=np.isfinite(y).all(1)&mask
        value['component_MSE']=np.mean((pred[known]-y[known])**2,axis=0).tolist() if known.any() else None
        out[key]=value
    return out


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); done=run.checked_training(identity)
    run.base.previous.require_committed(run.PUBLIC/'prediction_freeze.json')
    assert json.loads((run.PUBLIC/'prediction_freeze.json').read_text())['manifest']==run.artifact(run.PRIVATE/'training_complete.json')
    refs=[]
    for g,data,pairs in run.base.contexts(identity['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]
        for pair in cfg['pairs']:
            path=run.PUBLIC/'groups'/(name+'_'+pair+'.json'); receipt=run.PRIVATE/'readout'/path.name
            if path.exists():
                assert run.artifact(path)==json.loads(receipt.read_text()); refs.append(run.artifact(path)); continue
            bx,env,by,*_=run.base.pair_inputs(g,data,pairs,pair); folds=[]
            run.beat('held_readout',group=name,pair=pair)
            previous_row=json.loads((run.previous.PUBLIC/'groups'/path.name).read_text())
            for held in sorted(set(sites)):
                tr,te,pr,cut,y,masks=run.previous.fold_inputs(bx,env,by,data['baseline_ade'][bi,1],sites,held)
                tag=name+'_'+pair+'_'+held
                target=run.previous.diagnostic.event_targets(by[te],data['baseline_ade'][bi[te],1],cut)
                predictions={}; fits={}; metrics={}; heads={}
                for arm,home in (('mean',run.previous.PRIVATE),('fractional',run.PRIVATE)):
                    directory=home/'heads'/tag; heads[arm]=run.artifact(directory/'complete.json')
                    r=json.loads((directory/'complete.json').read_text()); assert r['input']['easy_cut']==cut
                    for ref in r['artifacts'].values(): assert run.artifact(ROOT/ref['path'])==ref
                    with np.load(directory/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); predictions[arm]=z['scores'].copy()
                    fits[arm]=json.loads((directory/'fit_diagnosis.json').read_text())
                    metrics[arm]=measure(predictions[arm],target,env[te],sites[te],fits[arm]['edges'])
                old=next(f for f in previous_row['folds'] if f['held']==held)
                for subset in ('all','envelope_positive'):
                    compare={k:v for k,v in metrics['mean'][subset].items() if k!='component_MSE'}
                    assert compare==old['held_metrics'][subset]
                folds.append(dict(held=held,easy_cut=cut,heads=heads,training=fits,metrics=metrics,
                    held_ids_sha256=run.array_hash(bi[te]),target_sha256=run.array_hash(target)))
            row=dict(identity=identity,group=name,producer=g['producer'],controller=g['controller'],
                seed=int(name.split('_seed')[1].split('_')[0]),pair=pair,folds=folds,
                result_source='fresh_run_fractional_head_and_paired_metrics_cached_verified_mean_control',
                independent_confirmation=False,policy_changed=False)
            run.immutable_json(path,row); run.immutable_json(receipt,run.artifact(path)); refs.append(run.artifact(path))
    assert len(refs)==36
    run.immutable_json(run.PUBLIC/'completion_checks.json',dict(identity=identity,groups=refs,all_passed=True,
        heads=144,source_binding=run.artifact(Path(__file__)),policy_changed=False))
    run.beat('readout_complete',groups=len(refs))


if __name__=='__main__':
    with (run.PRIVATE/'evaluate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
