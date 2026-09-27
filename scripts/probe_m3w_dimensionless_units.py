"""Observed-input-only unit probe, identical rows and tolerances to the control."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_dimensionless_refit as run
from scripts.probe_m3w_motion_unit_sensitivity import rescale_geometry
import numpy as np
import torch
from src.world_model.m3w_agent_track_context import AgentTrackSourceForecaster
from src.world_model.m3w_dimensionless_correction import DimensionlessAgentTrackSourceForecaster
from src.world_model.m3w_partial_context import condition_partial_inputs
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_european_source_forecast import baseline_torch


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    frozen=json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    registration=json.loads((run.PUBLIC/'registration.json').read_text())
    assert frozen['registration']==run.artifact(run.PUBLIC/'registration.json')
    for p,h in registration['bindings'].items():
        assert run.digest(ROOT/p)==h
    ref=registration['geometry']
    assert run.artifact(ROOT/ref['path'])==ref
    geo=json.loads((ROOT/ref['path']).read_text())['geometry']
    assert run.artifact(ROOT/geo['path'])==geo
    g=np.load(ROOT/geo['path'],mmap_mode='r',allow_pickle=False)
    previous=json.loads((run.previous.PUBLIC/'unit_sensitivity.json').read_text())
    rows=[]
    for pred,train,control in zip(frozen['predictions'],frozen['training'],registration['controls']):
        assert pred['key']==control['key']
        assert run.artifact(ROOT/train['path'])==train
        receipt=json.loads((ROOT/train['path']).read_text())
        for cp in (receipt['checkpoint'],control['checkpoint']):
            assert run.artifact(ROOT/cp['path'])==cp
        assert run.artifact(ROOT/pred['grouped']['path'])==pred['grouped']
        with np.load(ROOT/pred['grouped']['path'],allow_pickle=False) as z:
            held=z['ids'].copy()
        pool=held[:4096]
        _,scale=condition_partial_inputs(pack_geometry(g[pool]))
        ids=pool[scale.numpy()>=4][:256]
        assert len(ids)==256
        ids_hash=run.previous.old.array_hash(ids)
        controls=[r for r in previous['rows'] if r['trial']==pred['key'] and r['arm']=='grouped']
        assert len(controls)==4 and all(r['query_ids_sha256']==ids_hash for r in controls)
        x=g[ids].copy()
        index=receipt['identity']['parent_trial']['baseline_index']
        for arm,cls,cp in [('dimensionless',DimensionlessAgentTrackSourceForecaster,receipt['checkpoint']),
                           ('grouped',AgentTrackSourceForecaster,control['checkpoint'])]:
            model=cls(index).eval()
            model.load_state_dict(torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False)['model'])
            with torch.no_grad():
                inputs=pack_geometry(x)
                original=model(inputs).numpy()
                base=baseline_torch(inputs['history'],index).numpy()
                for factor in (.25,.5,2.,4.):
                    changed=pack_geometry(rescale_geometry(x,factor))
                    cb=baseline_torch(changed['history'],index).numpy()/factor
                    np.testing.assert_allclose(cb,base,rtol=1e-5,atol=1e-4)
                    y=model(changed).numpy()/factor
                    error=np.abs(y-original)
                    rows.append(dict(trial=pred['key'],arm=arm,factor=factor,queries=len(ids),
                        query_ids_sha256=ids_hash,baseline_scale_equivariance_checked=True,
                        mean_returned_coordinate_change=float(error.mean()),maximum_returned_coordinate_change=float(error.max()),
                        approximately_scale_equivariant=bool(np.allclose(y,original,rtol=1e-5,atol=1e-4))))
    doc=dict(result_source='fresh_observation_only_diagnostic',rows=rows,future_labels_read=False,
        accuracy_evaluated=False,conditioning_clamp_active=False,
        interpretation='Compare scale-restored squash input with dimensionless correction fraction',
        predictive_failure_causally_explained=False,independent_roles_read=False,deployment_changed=False)
    run.immutable_json(run.PUBLIC/'unit_sensitivity.json',doc)
    lines=['# Trained Coordinate-Unit Check','',
        'Identical observed-history prefixes, factors and tolerances to the previous diagnostic.',
        'Geometry is rescaled and output units are restored. The one-unit conditioning clamp is inactive.',
        'No future labels or accuracy evaluation. Approximate equivariance is not predictive lift or metric calibration.',
        'The new wrapper uses B + R*squash(f); the control uses B + R*squash(S*f).','',
        '| Trial | Arm | Factor | Mean returned-coordinate change | Maximum | Approximately equivariant |',
        '|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['trial']} | {r['arm']} | {r['factor']} | {r['mean_returned_coordinate_change']:.6g} | "+
            f"{r['maximum_returned_coordinate_change']:.6g} | {r['approximately_scale_equivariant']} |")
    (run.PUBLIC/'unit_sensitivity.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({arm:sum(r['approximately_scale_equivariant'] for r in rows if r['arm']==arm)
        for arm in ('dimensionless','grouped')}))


if __name__=='__main__':
    main()
