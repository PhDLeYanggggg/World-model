"""Separate arithmetic and preserved-receipt checks for the fitting diagnosis."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics

from scripts import audit_m3w_query_conditioning as runner


def stats(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), median=statistics.median(values) if values else None,
                mean=statistics.mean(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def comparison(probes, key, fixed=False):
    differences, tolerances = [], []
    for p in probes:
        if fixed:
            before = p['fixed_self_score_MSE_before'][key]
            a, b = [p['virtual_steps'][mode]['fixed_self_score_MSE'][key]
                    for mode in ('as_trained', 'unclipped')]
        else:
            before = p['before'][key]
            a, b = [p['virtual_steps'][mode]['after'][key] for mode in ('as_trained', 'unclipped')]
        if before is not None and a is not None and b is not None:
            differences.append(b-a); tolerances.append(max(1e-10, 1e-6*abs(before)))
    return dict(unclipped_minus_clipped=stats(differences),
                better=sum(x < -t for x, t in zip(differences, tolerances)),
                worse=sum(x > t for x, t in zip(differences, tolerances)),
                tied=sum(abs(x) <= t for x, t in zip(differences, tolerances)))


def execute(check=False):
    runner.check()
    public = runner.PUBLIC
    complete = json.loads((public/'complete.json').read_text())
    rows = runner.refs_read(complete['groups'], public/'groups')
    groups = runner.refs_read(complete['probes'], public/'probes')
    summary = json.loads((public/'summary.json').read_text())
    old = public.parent/'european_query_conditioning_v1'
    counts = dict(view_target_arithmetic=0, summary_fields=0, old_fit_views=0,
                  old_probe_groups=0, finite_probe_batches=0)
    for r in rows:
        for arm in ('self', 'peer'):
            for c in runner.audit.COHORTS:
                cohort = r['result'][arm][c]
                other = r['result']['peer' if arm == 'self' else 'self'][c]
                assert all(cohort[k] == other[k] for k in ('rows', 'known', 'unknown'))
                for name in ('all', 'easy'):
                    d = cohort[name]
                    assert math.isclose(d['residual_sum'], d['observed_excess']-d['predicted_excess'],
                                        rel_tol=1e-10, abs_tol=1e-8)
                    assert d['false_safe_rows'] <= cohort['known']
                    assert d['positive_residual']+1e-8 >= max(0, d['residual_sum'])
                    assert d['observed_above_cap_rows'] <= cohort['known']
                    counts['view_target_arithmetic'] += 1
        p = old/'groups'/f"{r['context']}_{r['source']}_seed{r['seed']}.json"
        if p.exists():
            previous = json.loads(p.read_text())
            current = copy.deepcopy(r)
            previous.pop('registration_sha256'); current.pop('registration_sha256')
            assert current == previous
            counts['old_fit_views'] += 1
    details = {}
    for arm in ('self', 'peer'):
        for c in runner.audit.COHORTS:
            for name in ('all', 'easy'):
                for key, expected in summary['arms'][arm]['cohorts'][c][name].items():
                    observed = math.fsum(r['result'][arm][c][name][key] for r in rows)
                    assert math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-8)
                    counts['summary_fields'] += 1
        probes = [p for g in groups if g['arm'] == arm for p in g['probes']]
        assert len(probes) == 144
        for p in probes:
            assert p['persistent_model_changed'] is False
            assert all(math.isfinite(v) for v in p['gradient_norms'].values())
            assert math.isclose(sum(p['before'][k] for k in ('primary', 'regret', 'dual_all', 'dual_easy')),
                                p['before']['total'], rel_tol=1e-12)
            counts['finite_probe_batches'] += 1
        by_group = []
        for g in groups:
            if g['arm'] != arm: continue
            by_group.append(dict(context=g['context'], seed=g['seed'],
                negative_data_risk_cosine=sum(p['data_risk_cosine'] is not None and p['data_risk_cosine'] < 0
                                             for p in g['probes']),
                total=comparison(g['probes'], 'total'),
                common_self_all_MSE=comparison(g['probes'], 'all', fixed=True),
                common_self_easy_MSE=comparison(g['probes'], 'easy', fixed=True)))
        details[arm] = dict(probe_batches=len(probes),
            negative_data_risk_cosine=sum(p['data_risk_cosine'] is not None and p['data_risk_cosine'] < 0 for p in probes),
            native_max_relative_additivity_error=max(p['additivity']['native_float32']['relative_component_norm_error'] for p in probes),
            double_max_relative_additivity_error=max(p['additivity']['reference_float64']['relative_component_norm_error'] for p in probes),
            total=comparison(probes, 'total'),
            common_self_all_MSE=comparison(probes, 'all', fixed=True),
            common_self_easy_MSE=comparison(probes, 'easy', fixed=True),
            step_norms={mode: stats([p['virtual_steps'][mode]['parameter_step_norm'] for p in probes])
                        for mode in ('as_trained', 'unclipped')},
            final_objective_change={mode: stats([p['virtual_steps'][mode]['change']['total'] for p in probes])
                                    for mode in ('as_trained', 'unclipped')}, by_group=by_group)
    for g in groups:
        p = old/'probes'/f"{g['context']}_head{g['seed']}_{g['arm']}.json"
        if p.exists():
            previous = json.loads(p.read_text()); current = copy.deepcopy(g)
            previous.pop('registration_sha256'); current.pop('registration_sha256')
            for probe in current['probes']: probe.pop('additivity')
            assert current == previous
            counts['old_probe_groups'] += 1
    assert counts['old_fit_views'] == 12 and counts['old_probe_groups'] == 4
    result = dict(checks=counts, details=details,
        complete_sha256=runner.run.sha(public/'complete.json'),
        summary_sha256=runner.run.sha(public/'summary.json'),
        verifier_sha256=runner.run.sha(Path(__file__)),
        result_source='cached_verified_receipt_arithmetic_no_new_inference',
        independent_external_replication=False, persistent_training=False,
        held_inference=False, independent_roles_read=False, deployment_changed=False)
    path = public/'verification.json'
    if check:
        assert result == json.loads(path.read_text())
    else:
        runner.run.once(path, result)
    print(json.dumps(dict(checks=counts, details={a: {k: v for k, v in d.items() if k != 'by_group'}
                                                for a, d in details.items()}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    execute(parser.parse_args().check)
