"""Past-only structural response, not an outcome or physical-plausibility test."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_agent_track_refit as run
import numpy as np
import torch
from src.world_model.m3w_agent_track_context import AgentTrackSourceForecaster
from src.world_model.m3w_partial_context import PartialContextSourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry
from src.evaluation.m3w_neighbor_association_probe import scramble_associations,neighbor_path_length


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    frozen=json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    registration=json.loads((run.PUBLIC/'registration.json').read_text())
    for p,h in registration['bindings'].items(): assert run.digest(ROOT/p)==h
    assert frozen['registration']==run.artifact(run.PUBLIC/'registration.json')
    ref=registration['geometry']; assert run.artifact(ROOT/ref['path'])==ref
    geo=json.loads((ROOT/ref['path']).read_text())['geometry']; assert run.artifact(ROOT/geo['path'])==geo
    g=np.load(ROOT/geo['path'],mmap_mode='r',allow_pickle=False)
    eligible=(g[:,230:294].reshape(-1,8,8)>0).all(2).sum(1)>=2
    rows=[]
    for pred,training,control in zip(frozen['predictions'],frozen['training'],registration['controls']):
        assert pred['key']==control['key']; assert run.artifact(ROOT/training['path'])==training
        receipt=json.loads((ROOT/training['path']).read_text())
        assert run.artifact(ROOT/receipt['checkpoint']['path'])==receipt['checkpoint']
        assert run.artifact(ROOT/control['checkpoint']['path'])==control['checkpoint']
        assert run.artifact(ROOT/pred['flat']['path'])==pred['flat']
        with np.load(ROOT/pred['flat']['path'],allow_pickle=False) as z: held=z['ids'].copy()
        ids=held[eligible[held]][:256]; assert len(ids)==256
        x=g[ids].copy(); changed=scramble_associations(x)
        index=receipt['identity']['parent_trial']['baseline_index']
        stats={}
        for arm,cls,cp in [('grouped',AgentTrackSourceForecaster,receipt['checkpoint']),
                           ('flat',PartialContextSourceForecaster,control['checkpoint'])]:
            model=cls(index).eval()
            model.load_state_dict(torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False)['model'])
            with torch.no_grad(): a=model(pack_geometry(x)).numpy(); b=model(pack_geometry(changed)).numpy()
            stats[arm]=dict(maximum_output_coordinate_change=float(np.max(np.abs(a-b))),
                mean_output_coordinate_change=float(np.mean(np.abs(a-b))),
                approximately_invariant=bool(np.allclose(a,b,rtol=1e-5,atol=1e-4)))
        before,after=neighbor_path_length(x),neighbor_path_length(changed)
        rows.append(dict(trial=pred['key'],held_queries=len(ids),query_ids_sha256=run.old.array_hash(ids),
            changed_neighbor_paths=int(np.count_nonzero(before!=after)),
            mean_original_neighbor_path=float(before.mean()),mean_reassigned_neighbor_path=float(after.mean()),**stats))
    doc=dict(observation_only=True,future_labels_read=False,prediction_targets_read=False,accuracy_evaluated=False,
        geometry=geo,registration=run.artifact(run.PUBLIC/'registration.json'),rows=rows,
        physical_plausibility_of_reassigned_tracks=False,causal_explanation_of_forecast_failure_proven=False,
        independent_roles_read=False,deployment_changed=False)
    run.immutable_json(run.PUBLIC/'association_probe.json',doc)
    lines=['# Trained Association Response','',
        'First256 eligible held-source histories per fixed model, selected only by observed masks.',
        'Swap two complete neighbor identities at past slots1/3/5, holding each time point set,',
        'ego, timestamps, current positions and masks unchanged. No future labels are read.',
        'The constructed paths need not be physically plausible. Sensitivity is not accuracy.','',
        '| Trial | Held queries | Flat mean output change | Grouped mean output change | Flat invariant | Grouped invariant |',
        '|---|---:|---:|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['trial']} | {r['held_queries']} | {r['flat']['mean_output_coordinate_change']:.8g} | "+
            f"{r['grouped']['mean_output_coordinate_change']:.8g} | {r['flat']['approximately_invariant']} | {r['grouped']['approximately_invariant']} |")
    lines+=['','These queries and fitted models overlap; this is not an independent statistical test.',
        'The probe checks the intended mechanism but does not establish interaction benefit.']
    (run.PUBLIC/'association_probe.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(pairs=len(rows),flat_invariant=sum(r['flat']['approximately_invariant'] for r in rows),
        grouped_invariant=sum(r['grouped']['approximately_invariant'] for r in rows))))


if __name__=='__main__': main()
