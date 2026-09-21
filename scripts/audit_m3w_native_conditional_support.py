"""Frozen-head in-fit diagnostics and analytic risk identity on training rows only."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_protected_risk import load, training_data
from scripts.run_m3w_native_forecast import array_hash, immutable_json, assert_current
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_conditional_support import rollout_distance, verify_cost_geometry, summarize_group
from src.world_model.m3w_native_protected_risk import build_head, predict_head
from src.world_model.m3w_native_gain_harm import build_head as gain_model, predict_neural
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, _, data, views, gain, identity = load()
    old_path = ROOT/reg['reports']/'analysis.json'
    old = json.loads(old_path.read_text())
    replay = json.loads((ROOT/reg['reports']/'independent_verification.json').read_text())
    assert old['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(old_path)
    records, bindings = [], dict(identity['source_bindings'])
    for path in ('scripts/audit_m3w_native_conditional_support.py',
                 'src/evaluation/m3w_native_conditional_support.py', 'tests/test_m3w_native_conditional_support.py',
                 reg['reports']+'/analysis.json', reg['reports']+'/independent_verification.json'):
        bindings[path] = file_digest(ROOT/path)
    for key, meta in views.items():
        ids, x, same, targets = training_data(meta, data)
        with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); pred = z['prediction'].copy()
        with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            cv, b, h = z['baseline_ade'].copy(), z['benefit'].copy(), z['harm'].copy()
        full = data['valid'][ids].all(1)
        distance = rollout_distance(pred, data['geometry'][ids, 332:356].reshape(-1, 12, 2), data['scale'][ids])
        geometry_check = verify_cost_geometry(distance, cv, h, b, full)
        risk_record = next(r for r in old['training'] if r['view'] == key and r['arm'] == 'zero_reference_harm')
        for r in (risk_record, next(r for r in gain['training'] if r['view'] == key and r['arm'] == 'mse')):
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            bindings[r['checkpoint']] = r['checkpoint_sha256']
        cp = torch.load(ROOT/risk_record['checkpoint'], map_location='cpu', weights_only=False)
        assert cp['identity']['labels_sha256'] == array_hash(targets['zero_reference_harm'][0])
        model = build_head(357, 64, meta['seed']); model.load_state_dict(cp['model'])
        score = predict_head(model, x, same, cp['preprocess'])
        gr = next(r for r in gain['training'] if r['view'] == key and r['arm'] == 'mse')
        gc = torch.load(ROOT/gr['checkpoint'], map_location='cpu', weights_only=False)
        assert gc['identity']['features_sha256'] == array_hash(x[:, :355].copy(), same)
        gm = gain_model(64, gc['preprocess'], meta['seed']); gm.load_state_dict(gc['model'])
        gp = predict_neural(gm, x[:, :355], same, gc['preprocess'])
        positive = (gp[:, 0] > gp[:, 1]) & ~same
        stop = x[:, -2] == 1
        groups = dict(all=np.ones(len(ids), bool), changed_candidate=~same,
            gain_positive=positive, gain_strict=positive & (gp[:, 1] <= .1*gp[:, 0]),
            protected_admitted=positive & (score[:, 0] <= .01),
            changed_stop=~same & stop, changed_nonstop=~same & ~stop,
            gain_positive_nonstop=positive & ~stop)
        summaries = {name:summarize_group(mask, full, cv, b, h, score[:, 0], score[:, 1], distance,
            data['tracks'][ids], data['sites'][ids]) for name, mask in groups.items()}
        z = full & (cv == 0)
        records.append(dict(view=key, rows=len(ids), outer_site=meta['outer_site'],
            training_sites=sorted(set(data['sites'][ids])), query_ids_sha256=array_hash(ids),
            analytic_identity=geometry_check, groups=summaries,
            zero_reference_positive_rows=int(z.sum()), direct_protected_positive_rows=int((z & ~same).sum()),
            zero_reference_positive_tracks=len(np.unique(data['tracks'][ids][z])),
            zero_reference_nonstop_rows=int((z & ~stop).sum()),
            zero_reference_nonstop_tracks=len(np.unique(data['tracks'][ids][z & ~stop]))))
        print(json.dumps(dict(view=key, training_only_support='complete',
            in_fit_protected_false_negatives=summaries['protected_admitted']['positive_event_rows'])), flush=True)
    assert_current(dict(source_bindings=bindings))
    result = dict(result_source='fresh_run_in_fit_head_replay_and_training_only_geometry_support',
        source_bindings=bindings, parent_analysis_sha256=file_digest(old_path), views=records,
        fitting_outer_rows=0, newly_scored_held_rows=0, new_training=False,
        threshold_search=False, deployment=False, independent_confirmation=False,
        selection_scores_are_in_fit_not_OOF_calibration=True,
        repeated_windows_not_independent_events=True)
    path = ROOT/'outputs/publication_readiness_2026_09/native_conditional_support_v1/analysis.json'
    immutable_json(path, result)
    print(json.dumps(dict(views=len(records), fresh_outer_readout=False, output=str(path))))


if __name__ == '__main__':
    main()
