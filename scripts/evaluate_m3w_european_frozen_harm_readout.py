"""All matched frozen-feature readouts, only after the new prediction freeze."""
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_frozen_harm_readout as run
from scripts.evaluate_m3w_european_support_fractional import measure
from src.evaluation.m3w_easy_membership_diagnosis import decompose
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); run.checked_training(identity)
    run.parent.base.previous.require_committed(run.PUBLIC/'prediction_freeze.json')
    refs=[]
    for g,data,pairs in run.parent.base.contexts(identity['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            path=run.PUBLIC/'groups'/(name+'_'+pair+'.json'); receipt=run.PRIVATE/'readout'/path.name
            if path.exists():
                assert run.artifact(path)==json.loads(receipt.read_text()); refs.append(run.artifact(path)); continue
            bx,env,by,*_=run.parent.base.pair_inputs(g,data,pairs,pair); folds=[]
            old=json.loads((run.parent.PUBLIC/'groups'/path.name).read_text())
            for held in sorted(set(sites)):
                tr,te,pr,cut,y,masks=run.parent.previous.fold_inputs(bx,env,by,cv,sites,held)
                target=run.parent.previous.diagnostic.event_targets(by[te],cv[te],cut)
                tag=name+'_'+pair+'_'+held; homes={
                    'original_mean':run.sources(tag)['mean_features'],'original_fractional':run.sources(tag)['fractional_features'],
                    **{a:run.PRIVATE/'heads'/tag/a for a in cfg['arms']}}
                metrics={}; predictions={}; fits={}; heads={}
                for arm,directory in homes.items():
                    heads[arm]=run.artifact(directory/'complete.json'); r=json.loads((directory/'complete.json').read_text())
                    for ref in r['artifacts'].values(): assert run.artifact(ROOT/ref['path'])==ref
                    assert r['input']['easy_cut']==cut
                    with np.load(directory/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); pred=z['scores'].copy()
                    predictions[arm]=pred; fits[arm]=json.loads((directory/'fit_diagnosis.json').read_text())
                    metrics[arm]=measure(pred,target,env[te],sites[te],fits[arm]['edges'])
                oldfold=next(f for f in old['folds'] if f['held']==held)
                assert oldfold['target_sha256']==run.array_hash(target)
                assert metrics['original_mean']==oldfold['metrics']['mean']
                assert metrics['original_fractional']==oldfold['metrics']['fractional']
                for a in cfg['arms']:
                    np.testing.assert_array_equal(predictions[a][:,[0,2]],predictions['original_mean'][:,[0,2]])
                membership={a:decompose(predictions['original_mean'],predictions[a],target,cv[te],env[te],cut) for a in cfg['arms']}
                folds.append(dict(held=held,easy_cut=cut,metrics=metrics,training=fits,heads=heads,membership=membership,
                    held_ids_sha256=run.array_hash(bi[te]),target_sha256=run.array_hash(target)))
            row=dict(group=name,producer=g['producer'],controller=g['controller'],seed=int(name.split('_seed')[1].split('_')[0]),
                pair=pair,folds=folds,result_source='fresh_run_readouts_cached_verified_original_controls',policy_changed=False)
            run.immutable_json(path,row); run.immutable_json(receipt,run.artifact(path)); refs.append(run.artifact(path))
            run.beat('held_readout',completed=len(refs),group=name,pair=pair)
    assert len(refs)==36
    run.immutable_json(run.PUBLIC/'completion_checks.json',dict(identity=identity,groups=refs,all_passed=True,
        source_binding=run.artifact(Path(__file__)),heads=288,policy_changed=False))
    run.beat('readout_complete',groups=len(refs))


if __name__=='__main__':
    with (run.PRIVATE/'evaluate.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); main()
