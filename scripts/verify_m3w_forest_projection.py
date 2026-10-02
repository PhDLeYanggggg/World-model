"""Independent aggregate arithmetic reader; no model inference or fitting."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_forest_projection_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)


def check_group(group):
    r = group['result']
    n = 0
    for key, slice_name in [('original', 'parent_selected'), ('harm_first', 'retained'), ('removed', 'removed')]:
        b = r[key]
        sl = r['slices'][slice_name]
        assert b['selected_count'] == sl['rows']
        assert b['selected_count']-b['selected_unknown'] == sl['known']
        truth = sl['known_target_mass']
        for name, value in zip(('benefit','harm','reference','easy_reference','easy_harm'), truth):
            close(b['selected_known_'+name+'_mass'], value)
        u = b['selected_unknown_envelope_mass']
        close(b['selected_net_gain_lower_mass'], truth[0]-truth[1]-u)
        close(b['all_selected_risk_upper'], (truth[1]+u)/truth[2] if truth[2] > 0 else None)
        close(b['easy_selected_risk_upper'], (truth[4]+u)/truth[3] if truth[3] > 0 else None)
        if b['selected_count'] == 0 or truth[2] == 0 or truth[3] == 0:
            assert not b['finite_completion_supported']
        n += 11
    for key in ('selected_count','selected_unknown','selected_known_benefit_mass','selected_known_harm_mass',
                'selected_known_reference_mass','selected_known_easy_reference_mass','selected_known_easy_harm_mass',
                'selected_unknown_envelope_mass','selected_net_gain_lower_mass'):
        close(r['original'][key], r['harm_first'][key]+r['removed'][key])
        n += 1
    close(r['utility_contrast_percent_full_known_reference'],
          -100*r['removed']['selected_net_gain_lower_mass']/r['full_known_reference_mass'])
    assert group['hashes']['targets'] and group['inputs_source'] == 'cached_verified'
    return n+2


def check_summary(groups, summary, cfg):
    assert len(groups) == summary['groups'] == cfg['source_heads'] == 72
    n = 0
    for arm in ('original','harm_first'):
        vals = [g['result'][arm] for g in groups]
        risk = [v['easy_selected_risk_upper'] for v in vals if v['easy_selected_risk_upper'] is not None]
        got = dict(selected=sum(v['selected_count'] for v in vals),
            selected_unknown=sum(v['selected_unknown'] for v in vals),
            complete_support=sum(v['finite_completion_supported'] for v in vals),
            defined_easy_risk=len(risk), easy_upper_violations=sum(v > .02+1e-12 for v in risk),
            worst_easy_upper=max(risk) if risk else None)
        assert got == summary[arm]
        n += len(got)
    for sl, values in summary['projection_counts'].items():
        for key, value in values.items():
            assert value == sum(g['result']['slices'][sl][key] for g in groups)
            n += 1
    sources = sorted({g['source'] for g in groups})
    assert len(sources) == summary['source_localities'] == 12
    for key in ('utility_contrast_percent_full_known_reference','query_count_matched_utility_contrast_percent'):
        vals = {s: [g['result'][key] for g in groups if g['source'] == s and g['result'][key] is not None] for s in sources}
        means = np.array([math.fsum(v)/len(v) for v in vals.values() if v])
        ids = np.random.default_rng(cfg['bootstrap_seed']).integers(len(means), size=(cfg['bootstrap_resamples'], len(means)))
        boot = np.array([math.fsum(means[row])/len(means) for row in ids])
        close(summary[key]['mean'], math.fsum(means)/len(means))
        for a, b in zip(summary[key]['CI95'], np.quantile(boot,[.025,.975])):
            close(a,b)
        n += 3
    return n


def main():
    reg = json.loads((PUBLIC/'registration.json').read_text())
    for path, digest in reg['bindings'].items():
        assert sha(ROOT/path) == digest
    done = json.loads((PUBLIC/'complete.json').read_text())
    assert done['exact_full_inference_replay'] and done['exact_aggregate_replay']
    groups = []
    for ref in done['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        groups.append(json.loads((ROOT/ref['path']).read_text()))
    assert sha(PUBLIC/'summary.json') == done['summary_sha256']
    cfg = json.loads((ROOT/'configs/m3w_european_forest_projection_v1.json').read_text())
    summary = json.loads((PUBLIC/'summary.json').read_text())
    checks = sum(check_group(g) for g in groups)+check_summary(groups, summary, cfg)
    files = [Path(__file__), ROOT/'tests/test_m3w_forest_projection_readout.py']
    result = dict(status='verified', aggregate_fields_checked=checks,
        row_scalar_fields_already_checked_during_run=done['independent_arithmetic_checks'],
        complete_sha256=sha(PUBLIC/'complete.json'), summary_sha256=sha(PUBLIC/'summary.json'),
        reader_bindings={str(p.relative_to(ROOT)):sha(p) for p in files},
        independent_model_implementation=False, independent_aggregate_arithmetic=True,
        independent_confirmation=False, deployment_changed=False)
    payload = json.dumps(result,indent=2)+'\n'
    path = PUBLIC/'verification.json'
    if path.exists():
        assert path.read_text() == payload
    else:
        with path.open('x') as f:
            f.write(payload)
    print(payload)


if __name__ == '__main__':
    main()
