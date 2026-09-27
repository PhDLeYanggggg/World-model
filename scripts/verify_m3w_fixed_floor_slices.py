"""Independent additive partitions, parent decisions and fixed-roster reductions."""
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_fixed_floor_slices as run

TESTS = ['tests/test_m3w_fixed_floor_slices.py', 'tests/test_m3w_fixed_floor_slices_protocol.py',
    'tests/test_m3w_fixed_floor_excess.py', 'tests/test_m3w_fixed_floor_excess_protocol.py',
    'tests/test_m3w_fixed_floor_probe.py', 'tests/test_m3w_fixed_floor_tail.py']


def metrics(s):
    def ratio(a, b, m=1): return m*a/b if b else None
    return dict(known_percent=ratio(s['known'], s['rows'], 100),
        intervention_percent=ratio(s['selected'], s['rows'], 100),
        eligible_percent=ratio(s['eligible'], s['rows'], 100),
        unknown_selected_percent=ratio(s['unknown_selected'], s['selected'], 100),
        harm_percent=ratio(s['harm_sum'], s['selected_reference_sum'], 100),
        net_gain_percent=ratio(s['benefit_sum']-s['harm_sum'], s['floor_sum'], 100),
        oracle_gain_percent=ratio(s['oracle_benefit_sum'], s['floor_sum'], 100),
        eligible_oracle_gain_percent=ratio(s['eligible_oracle_benefit_sum'], s['floor_sum'], 100),
        captured_benefit_percent=ratio(s['benefit_sum'], s['oracle_benefit_sum'], 100),
        harmful_selected_percent=ratio(s['harmful_selected'], s['selected_known'], 100),
        score_MSE=ratio(s['sq_error_sum'], s['known']), score_bias=ratio(s['bias_sum'], s['known']),
        eligible_score_MSE=ratio(s['eligible_sq_error_sum'], s['eligible_known']),
        selected_score_MSE=ratio(s['selected_sq_error_sum'], s['selected_known']),
        selected_predicted_excess=ratio(s['selected_predicted_excess_sum'], s['selected_known']),
        selected_observed_excess=ratio(s['selected_observed_excess_sum'], s['selected_known']))


def check_ci(value, expected, sites, cfg):
    assert value['by_site'] == expected
    if any(expected[s] is None for s in sites):
        assert value['point'] is None and value['ci95'] is None
    else:
        x = np.array([expected[s] for s in sites])
        indices = np.random.default_rng(cfg['bootstrap_seed']).integers(0, len(x), (cfg['bootstrap_resamples'], len(x)))
        assert value['point'] == x.mean()
        np.testing.assert_array_equal(value['ci95'], np.quantile(x[indices].mean(1), [.025, .975]))


def main():
    cfg = json.loads((ROOT/run.CONFIG).read_text())
    registration = json.loads((run.PUBLIC/'registration.json').read_text())
    for p, h in registration['bindings'].items(): assert run.parent.digest(ROOT/p) == h
    seal = registration['parent_seal']; assert run.parent.artifact(ROOT/seal['path']) == seal
    completion = json.loads((run.PUBLIC/'completion.json').read_text())
    assert completion['identity'] == registration and len(completion['groups']) == 108
    original = json.loads((run.parent.PRIVATE/'details.json').read_text())
    parent_metrics = {(r['group'], r['site'], r['policy']): r['metric'] for r in original['rows']}
    aggregate = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    count = 0; parent_checks = 0
    for ref in completion['groups']:
        assert run.parent.artifact(ROOT/ref['path']) == ref
        d = json.loads((ROOT/ref['path']).read_text()); assert d['identity'] == registration
        assert run.parent.artifact(ROOT/d['parent_head']['path']) == d['parent_head']
        fit, held = d['metadata']['fit_sites'], d['metadata']['held_sites']
        assert len(fit) == len(held) == 2 and not set(fit) & set(held)
        buckets = defaultdict(list)
        for r in d['rows']:
            buckets[(r['role'], r['site'], r['policy'])].append(r)
            dst = aggregate[(r['role'], r['policy'], r['axis']+'/'+r['bin'])][r['site']]
            for key, v in r['sums'].items(): dst[key] += v
        for (role, site, policy), rows in buckets.items():
            all_sums = next(r['sums'] for r in rows if r['axis'] == 'all')
            assert site in (fit if role == 'fit' else held)
            for axis in set(r['axis'] for r in rows):
                rr = [r['sums'] for r in rows if r['axis'] == axis]
                for key, v in all_sums.items():
                    np.testing.assert_allclose(sum(t[key] for t in rr), v, atol=1e-8, rtol=1e-10)
                    count += 1
            if role == 'held':
                p = parent_metrics[(d['metadata']['group'], site, policy)]; scale = d['metadata']['cost_scale']
                for key, value in (('rows', p['rows']), ('known', p['known_rows']),
                    ('selected', round(p['intervention_rate']*p['rows'])),
                    ('selected_known', round(p['known_intervention_rate']*p['known_rows'])),
                    ('unknown_selected', p['unknown_interventions'])):
                    assert all_sums[key] == value; parent_checks += 1
                np.testing.assert_allclose(all_sums['floor_sum'], p['floor_error_sum']/scale, rtol=1e-12)
                np.testing.assert_allclose(all_sums['benefit_sum']-all_sums['harm_sum'],
                    (p['floor_error_sum']-p['error_sum'])/scale, rtol=1e-8, atol=1e-8)
                parent_checks += 2
    summary = json.loads((run.PUBLIC/'summary.json').read_text())['summary']
    sites = registration['sites']; reductions = 0
    for (role, policy, label), local in aggregate.items():
        expected = {s: metrics(x) for s, x in local.items()}
        reported = summary[role][policy][label]
        assert reported['sums_by_site'] == local
        for metric in next(iter(expected.values())):
            check_ci(reported[metric], {s: expected.get(s, {}).get(metric) for s in sites}, sites, cfg)
            reductions += 1
    diag = json.loads((run.PUBLIC/'diagnosis.json').read_text())
    for label, values in diag['contrasts'].items():
        for k, v in values.items():
            a = summary['held']['excess'][label][k]['by_site']; b = summary['held']['mse'][label][k]['by_site']
            check_ci(v, {s: None if a[s] is None or b[s] is None else a[s]-b[s] for s in sites}, sites, cfg)
            reductions += 1
    for label, values in diag['excess_shares'].items():
        for k, v in values.items():
            key = k.removesuffix('_share_percent')
            a = summary['held']['excess'][label]['sums_by_site']; b = summary['held']['excess']['all/all']['sums_by_site']
            check_ci(v, {s: 100*a[s][key]/b[s][key] if b[s][key] > 0 else None for s in sites}, sites, cfg)
            reductions += 1
    assert json.loads((run.PUBLIC/'replay.json').read_text())['exact']
    log = run.PRIVATE/'scoped_pytest.txt'
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS], cwd=ROOT, capture_output=True, text=True)
    log.write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    tests = int(re.search(r'(\d+) passed', proc.stdout).group(1))
    before = {p.name: run.parent.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.json', '.md', '.png') and p.name != 'verification.json'}
    proc = subprocess.run([sys.executable, 'scripts/report_m3w_fixed_floor_slices.py'], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode: raise RuntimeError(proc.stderr)
    assert before == {p: run.parent.digest(run.PUBLIC/p) for p in before}
    bindings = dict(json.loads((ROOT/seal['path']).read_text())['source_bindings'])
    bindings.update(registration['bindings'])
    extras = ['scripts/report_m3w_fixed_floor_slices.py', 'scripts/verify_m3w_fixed_floor_slices.py', *TESTS]
    bindings.update({p: run.parent.digest(ROOT/p) for p in extras})
    run.parent.immutable_json(run.PUBLIC/'verification.json', dict(source_bindings=bindings,
        artifacts=before, tests=tests, test_files=len(TESTS), test_log=run.parent.artifact(log),
        full_diagnostic_replay_exact=True, independent_additive_checks=count,
        parent_metric_checks=parent_checks, independent_reductions=reductions,
        report_figure_byte_reproducible=True, full_legacy_suite='not_run', cold_raw_rebuild=False,
        new_training=False, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    print(json.dumps(dict(verified=True, tests=tests, additive_checks=count, parent_checks=parent_checks, reductions=reductions)))


if __name__ == '__main__': main()
