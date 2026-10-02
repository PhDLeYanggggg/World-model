"""Independent scalar/cluster-bootstrap reduction of frozen component diagnosis."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_quality_components_v1'
PRIOR = PUBLIC.parent/'european_past_quality_auxiliary_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return sum(equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        assert len(a) == len(b)
        return sum(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        assert b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)
    else:
        assert a == b, (a, b)
    return 1


def ci(rows, name, field, cfg):
    sites = {}
    for r in rows:
        sites.setdefault(r['source'], []).append(r['result']['arms'][name][field])
    values = [math.fsum(sites[s])/len(sites[s]) for s in sorted(sites)]
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(0, len(values),
                (cfg['bootstrap_draws'], len(values)))
    boot = [math.fsum(values[int(i)] for i in ix)/len(values) for ix in samples]
    return dict(mean=math.fsum(values)/len(values), CI95=np.quantile(boot, [.025, .975]).tolist(),
                localities=len(values), nominal_exposed_development_only=True)


def verify_cohorts(row):
    checks = 0
    for name, arm in row['result']['arms'].items():
        for c in arm['cohorts'].values():
            checks += equal(c['selected'], c['known']+c['unknown'])
            t, p = c['known_truth_moments'], c['known_prediction_moments']
            for event, hi, ri in (('all', 1, 2), ('easy', 4, 3)):
                expected = dict(observed_excess_mass=t[hi]-.02*t[ri],
                                predicted_excess_mass=p[hi]-.02*p[ri],
                                harm_underestimate_mass=t[hi]-p[hi],
                                reference_inflation_budget_mass=.02*(p[ri]-t[ri]))
                checks += equal(expected, c[event])
        cs = [arm['cohorts'][k] for k in ('added', 'retained')]
        moments = [math.fsum(c['known_truth_moments'][j] for c in cs) for j in range(5)]
        full = arm['full']; u = math.fsum(c['unknown_envelope_mass'] for c in cs)
        for j, field in enumerate(('selected_known_benefit_mass', 'selected_known_harm_mass',
                                   'selected_known_reference_mass', 'selected_known_easy_reference_mass',
                                   'selected_known_easy_harm_mass')):
            checks += equal(moments[j], full[field])
        checks += equal(sum(c['selected'] for c in cs), full['selected_count'])
        checks += equal(sum(c['unknown'] for c in cs), full['selected_unknown'])
        checks += equal(u, full['selected_unknown_envelope_mass'])
        checks += equal(moments[0]-moments[1]-u, full['selected_net_gain_lower_mass'])
        for hi, ri, field in ((1, 2, 'all_selected_risk_upper'), (4, 3, 'easy_selected_risk_upper')):
            value = (moments[hi]+u)/moments[ri] if moments[ri] > 0 else None
            checks += equal(value, full[field])
        checks += equal(arm['original_matched']['selected_count'], arm['variant_matched']['selected_count'])
        den = row['result']['full_known_reference_mass']
        matched = 100*(arm['variant_matched']['selected_net_gain_lower_mass']-
                      arm['original_matched']['selected_net_gain_lower_mass'])/den
        checks += equal(matched, arm['matched_utility_difference_percent'])
    return checks


def main():
    complete = json.loads((PUBLIC/'complete.json').read_text())
    assert sha(PUBLIC/'summary.json') == complete['summary_sha256']
    reg = json.loads((PUBLIC/'registration.json').read_text())
    for path, digest in reg['bindings'].items():
        assert sha(ROOT/path) == digest
    assert sha(PRIOR/'verification.json') == reg['parent_verification_sha256']
    rows = []; checks = 0
    for ref in complete['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text())
        checks += verify_cohorts(r); rows.append(r)
    assert len(rows) == 72 and len({r['source'] for r in rows}) == 12
    cfg = json.loads((ROOT/'configs/m3w_european_quality_components_v1.json').read_text())
    summary = json.loads((PUBLIC/'summary.json').read_text())
    prior = json.loads((PRIOR/'summary.json').read_text())
    for name, actual in summary['arms'].items():
        values = [r['result']['arms'][name] for r in rows]
        risk = [v['full']['easy_selected_risk_upper'] for v in values if v['full']['easy_selected_risk_upper'] is not None]
        expected = dict(selected=sum(v['full']['selected_count'] for v in values),
            unknown_selected=sum(v['full']['selected_unknown'] for v in values),
            complete_support=sum(v['full']['finite_completion_supported'] for v in values),
            defined_easy_risk=len(risk), upper_violations=sum(x > .02+1e-12 for x in risk),
            worst_easy_risk_upper=max(risk) if risk else None,
            known_label_violations=sum(v['full']['selected_known_easy_harm_mass']/v['full']['selected_known_easy_reference_mass'] > .02+1e-12
                for v in values if v['full']['selected_known_easy_reference_mass'] > 0),
            added=sum(v['cohorts']['added']['selected'] for v in values),
            removed=sum(v['cohorts']['removed']['selected'] for v in values),
            full_utility_difference_percent=ci(rows, name, 'full_utility_difference_percent', cfg),
            matched_utility_difference_percent=ci(rows, name, 'matched_utility_difference_percent', cfg))
        checks += equal(expected, actual)
    for name in ('original', 'quality'):
        for a, b in [('selected', 'selected'), ('unknown_selected', 'unknown_selected'),
                     ('complete_support', 'complete_support'), ('defined_easy_risk', 'defined_easy_risk'),
                     ('upper_violations', 'violations'), ('worst_easy_risk_upper', 'worst_easy_upper')]:
            checks += equal(summary['arms'][name][a], prior[name][b])
    for mode in ('full', 'matched'):
        checks += equal(summary['arms']['quality'][mode+'_utility_difference_percent'],
                        prior['quality_minus_original_'+mode+'_utility_percent'])
    result = dict(groups=72, variants_per_group=13, independent_scalar_bootstrap_fields_checked=checks,
                  summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json'),
                  exact_model_inference_replay=complete['exact_inference_replay'],
                  training=False, independent_confirmation=False)
    payload = json.dumps(result, indent=2)+'\n'; path = PUBLIC/'verification.json'
    if path.exists(): assert path.read_text() == payload
    else:
        with path.open('x') as f: f.write(payload)
    print(payload)


if __name__ == '__main__': main()
