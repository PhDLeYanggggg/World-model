"""Check sealed fixed-upstream results and their public delivery, without refitting."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_crossed_head_seed as run


def near(left, right):
    assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8), (left, right)


def main():
    pub = run.PUBLIC
    cfg, reg = run.registration()
    seal = json.loads((pub / 'verification.json').read_text())
    assert seal['source_bindings'] == reg['bindings']
    for name, h in seal['artifacts'].items():
        assert run.digest(pub / name) == h, name
    assert cfg['risk_budget'] == .02 and not cfg['threshold_search']
    s = json.loads((pub / 'summary.json').read_text())
    d = json.loads((pub / 'readout.json').read_text())
    frozen = {v['view']: v for v in json.loads((pub / 'decision_freeze.json').read_text())['rows']}
    metrics = {(v['view'], v['policy']): v['metric'] for v in d['rows']}
    assert len(frozen) == 72 and len(metrics) == 792 and len(d['components']) == 216
    for row in d['rows']:
        assert all(row[k] == v for k, v in frozen[row['view']].items())

    # Reconstruct component signs and ratios independently of the array helper.
    checks = 0
    for row in d['components']:
        a = row['accounting']
        p, y = a['predicted_known_selected_moments'], a['actual_known_selected_moments']
        expected = dict(predicted_easy_budget_excess=p[4] - .02*p[3],
            actual_easy_budget_excess=y[4] - .02*y[3],
            easy_harm_error_contribution=y[4] - p[4],
            easy_reference_error_contribution=.02*(p[3] - y[3]))
        for key, value in expected.items():
            near(a[key], value)
            checks += 1
        near(a['actual_easy_budget_excess'] - a['predicted_easy_budget_excess'],
             a['easy_harm_error_contribution'] + a['easy_reference_error_contribution'])
        risk = metrics[row['view'], 'seed' + str(row['head_seed'])]['selected_easy_positive_harm_ratio']
        if y[3] > 0:
            near(a['actual_easy_risk'], y[4]/y[3])
            near(risk, y[4]/y[3])
        else:
            assert a['actual_easy_risk'] is None and risk is None
        assert a['used_for_inference'] is False
        checks += 3

    fits = run.docs()
    fresh = [v for (g, seed), v in fits.items() if seed != 43]
    assert len(fits) == 72 and len(fresh) == 48
    assert all(v['trees'] == 128 and v['same_input_hashes_as_parent43'] for v in fresh)
    assert all(not set(v['partition']['train_recordings']) & set(v['partition']['validation_recordings'])
               for v in fits.values())
    r = json.loads((ROOT / 'research_state.json').read_text())['cvpr2027_research_track'][run.NAME]
    expected = dict(new_fits_completed=48, cached_verified_fits=24,
        cumulative_new_fit_seconds=s['new_fit_seconds'], new_checkpoint_bytes=s['new_checkpoint_bytes'],
        parent43_metric_views_exact=216, independent_metric_checks=d['independent_metric_checks'],
        query_count_checks=d['query_count_checks'], verification_sha256=run.digest(pub / 'verification.json'))
    for key, value in expected.items():
        assert r[key] == value, key
    for seed in ('17', '29', '43'):
        head = s['heads'][seed]
        assert r['source_screen_pass'][seed] == head['source_screen_pass']
        risk = head['policies']['screen' + seed]['easy_risk']
        assert r['screened_easy_risk'][seed] == risk
    assert not r['best_seed_selection'] and not r['independent_roles_read']
    assert not r['deployment_changed'] and not r['stage5c_executed'] and not r['smc_enabled']
    paths = [ROOT/'README.md', ROOT/'README_RESULTS.md', ROOT/'research_state.json', Path(__file__),
             pub/'results.md', pub/'failure_analysis.md', pub/'operation.md', pub/'verification.json']
    run.immutable(pub/'delivery_verification.json', dict(status='verified',
        numeric_seal_sha256=run.digest(pub/'verification.json'), state_value_checks=len(expected)+6,
        component_scalar_checks=checks, checkpoints_verified=72, new_full_fits=48,
        recording_disjoint_source_validation=True, frozen_directional_views=72, metric_rows=792,
        bindings={str(p.relative_to(ROOT)): run.digest(p) for p in paths},
        independent_confirmation=False, deployment_changed=False, full_legacy_suite='not_run'))
    print('Verified sealed results, 72 checkpoint hashes, 216 component decompositions and delivery')


if __name__ == '__main__':
    main()
