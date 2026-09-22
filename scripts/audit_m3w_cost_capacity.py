"""Post-readout fixed-population diagnosis; no new model or decision rule."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_cost_capacity import load, load_states
import numpy as np
import torch
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.verify_m3w_native_joint_controls import distances
from scripts.audit_m3w_eqmotion_cost_refit import groups, summarize
from src.evaluation.m3w_experiment_contract import file_digest

PAIRS = (
    ('duration_narrow', 'narrow_short', 'narrow_long'),
    ('duration_wide', 'wide_short', 'wide_long'),
    ('capacity_long', 'narrow_long', 'wide_long'),
    ('primary_reference', 'fraction', 'wide_long'),
)


def comparison_rows(old_bits, new_bits, valid, cv, candidate, old_score,
                    new_score, distance, speed, easy_cut):
    if not np.isfinite(easy_cut) or easy_cut <= 0:
        raise ValueError('Positive training-only easy cutoff required')
    rows = []
    subsets = dict(all=np.ones(len(cv), bool), positive_easy=(cv > 0) & (cv <= easy_cut))
    for subset, keep in subsets.items():
        for label, mask in groups(old_bits, new_bits).items():
            r = summarize(mask & keep, valid, cv, candidate, old_score,
                          new_score, distance, speed)
            if r['costs'] is not None:
                r['costs']['old'] = r['costs'].pop('frozen')
                r['costs']['new'] = r['costs'].pop('refit')
            rows.append(dict(subset=subset,
                group=label.replace('frozen', 'old').replace('refit', 'new'), **r))
    return rows


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, pcfg, data, views, predictions, controls, prior, identity = load()
    _, states = load_states(cfg, pcfg, views, prior, identity)
    public = ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text())
    receipt = json.loads((public/'independent_verification.json').read_text())
    assert receipt['all_checks_passed']
    assert receipt['analysis_sha256'] == file_digest(public/'analysis.json')
    local_identity = dict(analysis_sha256=file_digest(public/'analysis.json'), source_bindings={
        p: file_digest(ROOT/p) for p in (
            'scripts/audit_m3w_cost_capacity.py', 'tests/test_m3w_cost_capacity_forensics.py',
            'scripts/audit_m3w_eqmotion_cost_refit.py',
            str((public/'independent_verification.json').relative_to(ROOT)))})
    refs = {r['view']: r for r in controls['archives']}
    rows = []
    for archive in report['archives']:
        key = archive['view']
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            ids, d = z['ids'].copy(), z['distance'].copy()
            score = {a: z[a+'_score'].copy() for a in ('narrow_short', 'narrow_long', 'wide_short', 'wide_long')}
            bits = {(a, p): z[a+'_'+p].copy() for a in score for p in ('strict_stop', 'matched_count')}
        ref = refs[key]
        assert file_digest(ROOT/ref['path']) == ref['sha256']
        with np.load(ROOT/ref['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            score['fraction'] = z['refit_bounded_fraction'].copy()
            for policy in ('strict_stop', 'matched_count'):
                bits['fraction', policy] = z['refit_bounded_fraction_'+policy].copy()
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            prediction = z['prediction'].copy()
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        speed = np.linalg.norm(past[:, -1].astype(float)-past[:, -2], axis=1)*data['scale'][ids]
        y, valid = read_arrays(data, ids, 'target'), read_arrays(data, ids, 'valid')
        baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        cv, _ = distances(baseline, y, valid, data['scale'][ids])
        error, _ = distances(prediction, y, valid, data['scale'][ids])
        easy_cut = states[key, 'narrow_short']['preprocess']['positive_easy_cut']
        for name, old, new in PAIRS:
            for policy in ('strict_stop', 'matched_count'):
                for r in comparison_rows(bits[old, policy], bits[new, policy], valid, cv,
                        error, score[old], score[new], d, speed, easy_cut):
                    rows.append(dict(view=key, contrast=name, old_arm=old, new_arm=new,
                                     policy=policy, **r))
        print(json.dumps(dict(state='diagnosed', view=key)), flush=True)
    result = dict(identity=local_identity, rows=rows,
        result_source='fresh_post_readout_diagnostic_cached_verified_frozen_choices',
        same_population_for_both_score_comparisons=True, complete_outcomes_only_for_cost_means=True,
        easy_is_outcome_diagnostic_not_inference_input=True,
        unknown_as_zero=False, decisions_changed=False, threshold_search=False,
        new_training=False, independent_confirmation=False, causal_explanation_proven=False,
        deployment=False)
    assert len(rows) == 960
    assert_current(identity)
    assert_current(local_identity)
    path = public/'same_population_forensics.json'
    immutable_json(path, result)
    print(json.dumps(dict(state='complete', rows=len(rows), sha256=file_digest(path))))


if __name__ == '__main__':
    main()
