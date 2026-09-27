"""Check trained model response to input-only track reassignment, not accuracy."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_partial_neighbor_refit as run
import numpy as np
import torch
from src.world_model.m3w_partial_context import PartialContextSourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry
from src.evaluation.m3w_neighbor_association_probe import scramble_associations,neighbor_path_length


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    registration=run.artifact(run.PUBLIC/'registration.json')
    r=json.loads((run.PRIVATE/'partial/single0_seed17/complete.json').read_text())
    assert r['identity']['registration']==registration
    assert run.artifact(ROOT/r['checkpoint']['path'])==r['checkpoint']
    geometry=json.loads((run.PRIVATE/'geometry.json').read_text())
    assert geometry['registration']==registration
    assert run.artifact(ROOT/geometry['geometry']['path'])==geometry['geometry']
    g=np.load(ROOT/geometry['geometry']['path'],mmap_mode='r',allow_pickle=False)
    ids=np.flatnonzero((g[:,230:294].reshape(-1,8,8)>0).all(2).sum(1)>=2)[:256]
    x=g[ids].copy(); y=scramble_associations(x)
    model=PartialContextSourceForecaster(r['identity']['parent_trial']['baseline_index']).eval()
    model.load_state_dict(torch.load(ROOT/r['checkpoint']['path'],map_location='cpu',weights_only=False)['model'])
    with torch.no_grad(): a=model(pack_geometry(x)).numpy(); b=model(pack_geometry(y)).numpy()
    before,after=neighbor_path_length(x),neighbor_path_length(y)
    probe=dict(observation_only=True,future_labels_read=False,accuracy_evaluated=False,
        post_registration_structural_diagnostic=True,model=r['checkpoint'],geometry=geometry['geometry'],
        query_ids_sha256=run.array_hash(ids),queries=len(ids),
        changed_individual_neighbor_paths=int(np.count_nonzero(before!=after)),
        mean_original_neighbor_path=float(before.mean()),mean_reassigned_neighbor_path=float(after.mean()),
        maximum_output_coordinate_change=float(np.max(np.abs(a-b))),
        mean_output_coordinate_change=float(np.mean(np.abs(a-b))),
        approximately_invariant_rtol1e_5_atol1e_4=bool(np.allclose(a,b,rtol=1e-5,atol=1e-4)),
        interpretation='Same per-time point sets with different neighbor-track associations; flat token model cannot identify association from ordering',
        causal_explanation_of_forecast_failure_proven=False,independent_roles_read=False,deployment_changed=False)
    run.immutable_json(run.PUBLIC/'association_probe.json',probe)
    print(json.dumps(probe))


if __name__=='__main__': main()
