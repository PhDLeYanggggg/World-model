"""Input-only coordinate-rescaling probe; no accuracy or physical-scale claim."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_agent_track_refit as run
import numpy as np
import torch
from src.world_model.m3w_agent_track_context import AgentTrackSourceForecaster
from src.world_model.m3w_partial_context import PartialContextSourceForecaster,condition_partial_inputs
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_european_source_forecast import baseline_torch


def rescale_geometry(g, factor):
    if factor<=0 or not np.isfinite(factor): raise ValueError('Positive finite factor required')
    out=np.array(g,copy=True)
    for start,end in ((0,16),(38,166),(332,356)): out[:,start:end]*=factor
    return out


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    frozen=json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    registration=json.loads((run.PUBLIC/'registration.json').read_text())
    assert frozen['registration']==run.artifact(run.PUBLIC/'registration.json')
    for p,h in registration['bindings'].items(): assert run.digest(ROOT/p)==h
    ref=registration['geometry']; assert run.artifact(ROOT/ref['path'])==ref
    geo=json.loads((ROOT/ref['path']).read_text())['geometry']; assert run.artifact(ROOT/geo['path'])==geo
    g=np.load(ROOT/geo['path'],mmap_mode='r',allow_pickle=False); rows=[]
    for pred,train,control in zip(frozen['predictions'],frozen['training'],registration['controls']):
        assert pred['key']==control['key']; assert run.artifact(ROOT/train['path'])==train
        receipt=json.loads((ROOT/train['path']).read_text())
        assert run.artifact(ROOT/receipt['checkpoint']['path'])==receipt['checkpoint']
        assert run.artifact(ROOT/control['checkpoint']['path'])==control['checkpoint']
        assert run.artifact(ROOT/pred['flat']['path'])==pred['flat']
        with np.load(ROOT/pred['flat']['path'],allow_pickle=False) as z: held=z['ids'].copy()
        # Require observed extent >=4 so the one-unit conditioning clamp is inactive even at x0.25.
        pool=held[:4096]; _,scale=condition_partial_inputs(pack_geometry(g[pool]))
        ids=pool[scale.numpy()>=4][:256]; assert len(ids)==256
        x=g[ids].copy(); index=receipt['identity']['parent_trial']['baseline_index']
        for arm,cls,cp in [('grouped',AgentTrackSourceForecaster,receipt['checkpoint']),
                           ('flat',PartialContextSourceForecaster,control['checkpoint'])]:
            model=cls(index).eval()
            model.load_state_dict(torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False)['model'])
            with torch.no_grad():
                inputs=pack_geometry(x); original=model(inputs).numpy()
                base=baseline_torch(inputs['history'],index).numpy()
                for factor in (.25,.5,2.,4.):
                    changed=pack_geometry(rescale_geometry(x,factor))
                    cb=baseline_torch(changed['history'],index).numpy()/factor
                    np.testing.assert_allclose(cb,base,rtol=1e-5,atol=1e-4)
                    y=model(changed).numpy()/factor
                    error=np.abs(y-original)
                    rows.append(dict(trial=pred['key'],arm=arm,factor=factor,queries=len(ids),
                        query_ids_sha256=run.old.array_hash(ids),baseline_scale_equivariance_checked=True,
                        mean_returned_coordinate_change=float(error.mean()),maximum_returned_coordinate_change=float(error.max()),
                        approximately_scale_equivariant=bool(np.allclose(y,original,rtol=1e-5,atol=1e-4))))
    doc=dict(result_source='fresh_observation_only_diagnostic',rows=rows,future_labels_read=False,
        accuracy_evaluated=False,conditioning_clamp_active=False,
        interpretation='Causal bound scales with coordinates, but radial squash consumes a scale-restored residual',
        predictive_failure_causally_explained=False,independent_roles_read=False,deployment_changed=False)
    run.immutable_json(run.PUBLIC/'unit_sensitivity.json',doc)
    lines=['# Coordinate-Unit Sensitivity','',
        'Uniformly rescale ONLY observed geometry and causal rollouts, then undo scaling on predictions.',
        'The observed-context clamp is inactive in every chosen case. CV/selected baseline equivariance',
        'is checked. No labels, model selection or physically verified scale are involved.',
        'The unchanged wrapper computes B(x) + R(x) * squash(S(x) * f(x/S(x))).',
        'For positive rescaling c, the squash argument becomes c*S(x)*f, so the composed',
        'forecaster need not scale linearly even when the normalized core inputs match.',
        'This is a design sensitivity, not proof that it caused held forecast error or that',
        'removing scale restoration would improve learning. Current frozen models are not changed.','',
        '| Trial | Arm | Input factor | Mean output change after undo | Maximum change | Approximately equivariant |',
        '|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['trial']} | {r['arm']} | {r['factor']} | {r['mean_returned_coordinate_change']:.6g} | "+
            f"{r['maximum_returned_coordinate_change']:.6g} | {r['approximately_scale_equivariant']} |")
    (run.PUBLIC/'unit_sensitivity.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cases=len(rows),approximately_equivariant=sum(r['approximately_scale_equivariant'] for r in rows))))


if __name__=='__main__': main()
