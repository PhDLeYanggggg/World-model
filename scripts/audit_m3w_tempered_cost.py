"""Post-readout diagnosis with frozen fitting cuts and unchanged policy groups."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_tempered_cost import load
import numpy as np
import torch
from scripts.run_m3w_bounded_cost import read_arrays, training_data, features, selections
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.verify_m3w_native_joint_controls import distances
from scripts.audit_m3w_eqmotion_cost_refit import groups, summarize
from src.evaluation.m3w_cost_support_audit import fit_cuts, strata, cost_stats
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_bounded_cost_head import build, predict


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, prior, identity = load()
    public = ROOT/cfg['reports']; report = json.loads((public/'analysis.json').read_text())
    verification = json.loads((public/'independent_verification.json').read_text())
    assert verification['all_checks_passed'] and verification['analysis_sha256'] == file_digest(public/'analysis.json')
    diagnostic = json.loads((ROOT/cfg['fit_diagnostic']).read_text())
    local_identity = dict(analysis_sha256=file_digest(public/'analysis.json'), source_bindings={
        p:file_digest(ROOT/p) for p in ('scripts/audit_m3w_tempered_cost.py',
        'scripts/audit_m3w_eqmotion_cost_refit.py', 'src/evaluation/m3w_cost_support_audit.py')})
    old_archives = {r['view']:r for r in prior['archives']}
    rows, comparisons = [], []
    def motion(ids):
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        speed = np.linalg.norm(past[:, -1].astype(float)-past[:, -2], axis=1)*data['scale'][ids]
        return past, speed
    for key, meta in views.items():
        ids, x, labels, d, pr = training_data(meta, data); past, speed = motion(ids)
        cuts = fit_cuts(d, speed, pr['known']); assert cuts == diagnostic['cuts'][key]
        r = next(r for r in report['training'] if r['view'] == key)
        assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
        model = build(x.shape[1], cfg['training']['width'], meta['seed']); model.load_state_dict(cp['model'])
        train_score = predict(model, x, d, pr, 'bounded_native')
        fractions = np.where(pr['known'][:, None], labels/np.where(d > 0, d, 1.)[:, None], 0)
        constant_fraction = (pr['weights'][:, None]*fractions).sum(0)
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            held_ids, p = z['ids'].copy(), z['prediction'].copy()
        hx, hd, _ = features(data['geometry'][held_ids], p, data['scale'][held_ids])
        hpast, hspeed = motion(held_ids)
        held_score = predict(model, hx, hd, pr, 'bounded_native')
        archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], held_ids); np.testing.assert_array_equal(z['score'], held_score)
            new_bits = {k:z[k].copy() for k in ('strict_stop', 'matched_count')}
        with np.load(ROOT/old_archives[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], held_ids); old_score = z['refit_bounded_fraction'].copy()
            old_bits = {k:z['refit_bounded_fraction_'+k].copy() for k in new_bits}
        target, valid = read_arrays(data, held_ids, 'target'), read_arrays(data, held_ids, 'valid')
        b = data['geometry'][held_ids, 332:356].reshape(-1, 12, 2)
        cv, _ = distances(b, target, valid, data['scale'][held_ids])
        error, _ = distances(p, target, valid, data['scale'][held_ids])
        hy = np.column_stack((np.maximum(cv-error, 0), np.maximum(error-cv, 0))); hy[~valid.all(1)] = np.nan
        for population, qids, score, y, distance, movement, history, support in (
                ('fitting', ids, train_score, labels, d, speed, past, pr['known']),
                ('held_source', held_ids, held_score, hy, hd, hspeed, hpast, valid.all(1))):
            sets = {'all':np.ones(len(qids), bool)}
            sets.update({'disagreement_'+k:v for k,v in strata(distance, cuts['disagreement']).items()})
            sets.update({'speed_'+k:v for k,v in strata(movement, cuts['past_step_displacement']).items()})
            bits = selections(score, history, distance, np.zeros(len(qids), bool), qids)
            sets.update({k:bits[k] for k in ('net_stop', 'strict_stop')})
            for label, mask in sets.items():
                rows.append(dict(view=key, population=population, stratum=label,
                    **cost_stats(score, y, distance, mask, support, pr['constant'], constant_fraction)))
        for policy in new_bits:
            for label, mask in groups(old_bits[policy], new_bits[policy]).items():
                r = summarize(mask, valid, cv, error, old_score, held_score, hd, hspeed)
                if r['costs'] is not None:
                    r['costs']['fraction'] = r['costs'].pop('frozen')
                    r['costs']['tempered'] = r['costs'].pop('refit')
                label = label.replace('frozen', 'fraction').replace('refit', 'tempered')
                comparisons.append(dict(view=key, policy=policy, group=label, **r))
        print(json.dumps(dict(state='diagnosed', view=key)), flush=True)
    out = dict(identity=local_identity, result_source='fresh_post_readout_diagnostic_cached_verified_fits',
        rows=rows, same_population_comparisons=comparisons, cuts_source=cfg['fit_diagnostic'],
        fit_is_in_sample_optimistic=True, closed_role_readout=False, decisions_changed=False,
        threshold_search=False, new_training=False, causal_explanation_proven=False,
        independent_confirmation=False, deployment=False)
    assert_current(identity); assert_current(local_identity); immutable_json(public/'fit_and_selection_forensics.json', out)
    print(json.dumps(dict(state='completed', fit_records=len(rows), same_population_records=len(comparisons),
        sha256=file_digest(public/'fit_and_selection_forensics.json'))))


if __name__ == '__main__': main()
