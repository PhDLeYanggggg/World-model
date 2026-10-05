"""Independent report arithmetic for temporal targets; no model selection."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_m3w_label_support import sha, interval, check

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_temporal_target_audit_v1'


def verify_row(row):
    checks = 0
    for role, d in row['decomposition'].items():
        assert sum(d['step_counts']) == d['rows']
        assert d['step_counts'][0] == d['unknown_rows']
        assert d['step_counts'][12] == d['full_label_rows']
        assert d['selected_netharm_with_cancellation'] <= d['selected_netharm_rows'] <= d['selected']-d['unknown_selected']
        assert d['selected_nonharmful_but_step_harmful']+d['selected_netharm_with_cancellation'] <= d['selected_opposite_sign']
        assert math.isclose(d['selected_step_harm']-d['selected_cancellation'], d['selected_harm'], rel_tol=1e-9, abs_tol=1e-7)
        assert d['known_selected_easy_step_harm'] >= d['known_selected_easy_harm']-1e-7
        den = d['known_selected_easy_reference']
        for key, numer in (('original_known_easy_ratio', 'known_selected_easy_harm'), ('changed_target_known_easy_ratio', 'known_selected_easy_step_harm')):
            expected = d[numer]/den if den > 0 else None
            checks += check(d[key], expected)
        checks += 7
    for m in row['probes'].values():
        assert m['scored_steps']+m['unsupported_observed_steps'] == m['observed_steps']
        assert m['unknown_rows'] <= m['rows']
        for arm in ('temporal_leaf', 'rowmean_leaf', 'global_temporal'):
            assert m[arm] is None or (len(m[arm]) == 2 and all(math.isfinite(v) and v >= 0 for v in m[arm]))
        checks += 5
    assert row['exact_refit'] and row['exact_inference'] and row['original_policy_unchanged']
    return checks+3


def aggregate(rows, cfg):
    out = {}
    for cohort in ('validation', 'complete_validation', 'selected_validation'):
        for other in ('rowmean_leaf', 'global_temporal'):
            for channel, index in (('signed_error', 0), ('reference_error', 1), ('mean_channels', None)):
                def get(r, cohort=cohort, other=other, index=index):
                    m = r['probes'][cohort]
                    if m['temporal_leaf'] is None: return None
                    diffs = [a-b for a, b in zip(m['temporal_leaf'], m[other])]
                    return math.fsum(diffs)/2 if index is None else diffs[index]
                out[cohort+'_temporal_minus_'+other+'_'+channel] = interval(rows, get, cfg['bootstrap_seed'], cfg['bootstrap_resamples'])
    return out


def main():
    registration = PUBLIC/'registration_amended.json'
    reg = json.loads(registration.read_text())
    assert reg['original_registration_sha256'] == sha(PUBLIC/'registration.json')
    for path, h in reg['bindings'].items(): assert sha(ROOT/path) == h
    complete = json.loads((PUBLIC/'complete.json').read_text())
    result = json.loads((PUBLIC/'summary.json').read_text())
    assert sha(PUBLIC/'summary.json') == complete['summary_sha256']
    cfg = json.loads((ROOT/'configs/m3w_european_temporal_target_audit_v1.json').read_text())
    prior = ROOT/'outputs/publication_readiness_2026_09/european_label_support_v1'
    assert sha(prior/'verification.json') == reg['source_label_verification_sha256']
    anchors = {}
    for ref in json.loads((prior/'complete.json').read_text())['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text()); anchors[r['group'], r['head_seed']] = r
    rows = []; checks = 0
    for ref in complete['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        row = json.loads((ROOT/ref['path']).read_text()); rows.append(row)
        assert row['registration_sha256'] == sha(registration)
        anchor = anchors[row['group'], row['head_seed']]
        assert row['partition'] == anchor['partition']
        for key in ('ids', 'original_prediction', 'original_action'):
            assert row['hashes']['validation'][key] == anchor['hashes'][key.removeprefix('original_')]
        assert row['decomposition']['validation']['selected'] == sum(s['selected'] for s in anchor['cohort']['strata'].values())
        checks += check(row['decomposition']['validation']['known_selected_easy_harm'],
                        math.fsum(s['easy_harm'] for s in anchor['cohort']['strata'].values()))
        checks += verify_row(row)+5
    assert len(rows) == len({(r['group'], r['head_seed']) for r in rows}) == 72
    assert len({r['source'] for r in rows}) == 12
    contrasts = aggregate(rows, cfg)
    checks += check(result['contrasts'], contrasts)
    failures = []
    for cohort in ('validation', 'complete_validation'):
        for other in ('rowmean_leaf', 'global_temporal'):
            for channel in ('signed_error', 'mean_channels'):
                key = cohort+'_temporal_minus_'+other+'_'+channel
                if contrasts[key]['CI95'] is None or contrasts[key]['CI95'][1] >= 0: failures.append(key)
    assert result['failure_reasons'] == failures
    assert result['advance_to_auxiliary_design'] == (not failures)
    for key, value in result['pooled_validation'].items():
        checks += check(value, math.fsum(r['decomposition']['validation'][key] for r in rows))
    receipt = dict(independent_scalar_bootstrap_checks=checks, groups=72,
        original_policy_hashes_preserved=True, primary_target_identity_checked=True,
        summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json'),
        registration_sha256=sha(registration), verifier_sha256=sha(Path(__file__)),
        new_neural_training=False, independent_confirmation=False, deployment_changed=False)
    path = PUBLIC/'verification.json'; text = json.dumps(receipt, indent=2)+'\n'
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
