"""Compare fixed cost heads on common fit/held populations without selection changes."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_cost_budget_matched import load, load_states
from scripts.run_m3w_bounded_cost import training_data, read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_native_joint_controls import distances
from src.world_model.m3w_bounded_cost_head import build, predict
from src.evaluation.m3w_conditional_cost_audit import ARMS, strict_bits, fixed_groups, conditional_stats
from src.evaluation.m3w_cost_support_audit import fit_cuts, strata
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, data, views, predictions, _, _, refs, parent = load()
    _, states = load_states(cfg, views, refs, parent)
    public = ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text())
    verify = json.loads((public/'independent_verification.json').read_text())
    assert verify['all_checks_passed'] and verify['analysis_sha256'] == file_digest(public/'analysis.json')
    local = dict(source_bindings={p:file_digest(ROOT/p) for p in (
        'scripts/audit_m3w_matched_conditional_costs.py',
        'src/evaluation/m3w_conditional_cost_audit.py', 'tests/test_m3w_conditional_cost_audit.py',
        str((public/'analysis.json').relative_to(ROOT)), str((public/'independent_verification.json').relative_to(ROOT)))})
    archives = {r['view']:r for r in report['archives']}
    rows, provenance = [], []
    for key, meta in views.items():
        ti, x, ty, td, pr = training_data(meta, data)
        tvalid = read_arrays(data, ti, 'valid')
        with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ti, z['ids'])
            tcv = z['baseline_ade'].copy()
        score = {}
        for a in ARMS:
            cp = states[key, a]
            model = build(x.shape[1], cfg['training']['width'], meta['seed'])
            model.load_state_dict(cp['model'])
            score[a] = predict(model, x, td, pr, 'bounded_native')
        train_past = data['geometry'][ti, :16].reshape(-1, 8, 2)
        train_speed = np.linalg.norm(train_past[:, -1].astype(float)-train_past[:, -2], axis=1)*data['scale'][ti]
        cuts = fit_cuts(td, train_speed, pr['known'])
        archive = archives[key]
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            hi, hd = z['ids'].copy(), z['distance'].copy()
            held_score = {a:z[a+'_score'].copy() for a in ARMS}
            held_bits = {a:z[a+'_strict_stop'].copy() for a in ARMS}
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], hi)
            prediction = z['prediction'].copy()
        y, hv = read_arrays(data, hi, 'target'), read_arrays(data, hi, 'valid')
        baseline = data['geometry'][hi, 332:356].reshape(-1, 12, 2)
        hcv, _ = distances(baseline, y, hv, data['scale'][hi])
        he, _ = distances(prediction, y, hv, data['scale'][hi])
        delta = hcv-he
        hy = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        hy[~hv.all(1)] = np.nan
        for population, ids, scores, costs, d, valid, cv in (
                ('fitting', ti, score, ty, td, tvalid, tcv),
                ('held_source', hi, held_score, hy, hd, hv, hcv)):
            past = data['geometry'][ids, :16].reshape(-1, 8, 2)
            bits = {a:strict_bits(scores[a], past, d) for a in ARMS}
            if population == 'held_source':
                for a in ARMS:
                    np.testing.assert_array_equal(bits[a], held_bits[a])
            eligible = (d > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
            groups = fixed_groups(bits, eligible)
            all_mask = np.ones(len(ids), bool)
            diagnostic = dict(all=all_mask, positive_easy=(cv > 0) & (cv <= pr['positive_easy_cut']))
            diagnostic.update({'disagreement_'+n:m for n,m in strata(d, cuts['disagreement']).items()})
            for group, mask in groups.items():
                for subset, keep in diagnostic.items():
                    rows.append(dict(view=key, population=population, group=group, subset=subset,
                        **conditional_stats(scores, costs, valid.all(1), valid.any(1), mask & keep, d)))
            provenance.append(dict(view=key, population=population, cuts=cuts,
                causal_membership_sha256=array_hash(ids, *[bits[a] for a in ARMS]), rows=len(ids)))
        print(json.dumps(dict(view=key, state='conditional_diagnosis_complete')), flush=True)
    result = dict(identity=local, analysis_sha256=file_digest(public/'analysis.json'),
        rows=rows, provenance=provenance, result_source='fresh_diagnostics_of_cached_verified_models',
        shared_population_across_score_arms=True, fit_cuts_only=True,
        easy_future_diagnostic_only=True, unknown_as_safe=False, inference_changes=False,
        new_training=False, threshold_search=False, independent_confirmation=False,
        deployment=False, stage5c_executed=False, smc_enabled=False)
    assert len(rows) == 2016 and len(provenance) == 24
    assert_current(parent)
    assert_current(local)
    path = public/'conditional_forensics.json'
    immutable_json(path, result)
    print(json.dumps(dict(records=len(rows), sha256=file_digest(path), checks_passed=True)))


if __name__ == '__main__':
    main()
