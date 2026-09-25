"""Posthoc matched-action C moment errors; never changes policy or fitting."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_easy_harm_sampling as run
from scripts.audit_m3w_easy_harm_fitting import summarize_fit
import numpy as np
import torch


def evaluation_weights(y, sites):
    known = np.isfinite(y).all(1); sites = np.asarray(sites)
    p = np.zeros(len(y),float); roster = sorted(set(sites))
    for site in roster:
        take = known & (sites==site)
        if not take.any(): raise ValueError('No evaluable source-locality support')
        p[take] = 1/len(roster)/take.sum()
    return p


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); done=run.checked_training(identity)
    run.parent.previous.inc.ensure_frozen()
    _,_,ctx,_,_,_=run.parent.previous.inc.load(); data=ctx[2]; rows=[]
    for ref in done['decisions']:
        receipt=json.loads((ROOT/ref['path']).read_text()); g=receipt['group']; pair=receipt['pair']; name=g['group']
        with np.load(ROOT/receipt['array']['path'],allow_pickle=False) as z:
            ids=z['ids'].copy(); fixed=z['raw_neural'].copy()
        source=run.parent.previous.PRIVATE/'source'/(name+'_'+pair)
        old=json.loads((source/'receipt.json').read_text())
        assert run.array_hash(ids)==old['ids_sha256']
        assert run.artifact(source/'labels.npz')==old['arrays']['labels']
        with np.load(source/'labels.npz',allow_pickle=False) as z:
            y=run.parent.method.event_targets(z['cv'],z['reference'],z['candidate'],g['easy_cut'])
        p=evaluation_weights(y,data['sites'][ids]); fits={}
        for arm,directory in (
            ('uniform',run.parent.PRIVATE/'heads'/(name+'_'+pair+'_mean')),
            ('corrected',run.PRIVATE/'heads'/(name+'_'+pair))):
            _,state=run.parent.restore(directory)
            with np.load(directory/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(ids,z['ids']); prediction=z['scores'].copy()
            fits[arm]=summarize_fit(prediction,y,p,state['preprocess']['cost_scale'],state['loss_scales'],fixed)
        rows.append(dict(group=name,pair=pair,fits=fits,action_hash=run.array_hash(fixed),
            label_source=run.artifact(source/'labels.npz')))
    run.immutable_json(run.PUBLIC/'matched_transport_audit.json',dict(groups=rows,
        result_source='fresh_run_posthoc_C_reduction_cached_verified_predictions',
        same_old_raw_actions=True, B_only_cost_and_loss_scales=True,
        C_weights_evaluation_only=True, changed_fit=False,changed_actions=False,
        source_binding=run.artifact(Path(__file__))))
    print(json.dumps(dict(groups=len(rows),changed_actions=False)))


if __name__=='__main__': main()
