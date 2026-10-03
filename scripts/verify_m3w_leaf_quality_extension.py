"""Independent frozen-control, scalar-identity and locality-bootstrap verification."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_m3w_past_quality_auxiliary import sha, compare, ci

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_leaf_quality_extension_v1'
PARENT = PUBLIC.parent/'european_cost_harm_newton_v1'
CONTROLS = ('original', 'additive', 'poisson', 'cost')


def aggregate(rows, cfg):
    out = dict(groups=len(rows), localities=len({r['source'] for r in rows}), new_training=False,
        threshold_search=False, independent_confirmation=False, transfer_evaluated=False, deployment_changed=False)
    for arm in rows[0]['result']['policies']:
        values = [r['result']['policies'][arm] for r in rows]
        risk = [v['easy_selected_risk_upper'] for v in values if v['easy_selected_risk_upper'] is not None]
        known = [v['selected_known_easy_harm_mass']/v['selected_known_easy_reference_mass']
                 for v in values if v['selected_known_easy_reference_mass'] > 0]
        out[arm] = dict(selected=sum(v['selected_count'] for v in values),
            unknown_selected=sum(v['selected_unknown'] for v in values),
            complete_support=sum(v['finite_completion_supported'] for v in values), defined_easy_risk=len(risk),
            violations=sum(v > .02+1e-12 for v in risk), worst_easy_upper=max(risk) if risk else None,
            known_label_violations=sum(v > .02+1e-12 for v in known))
    for k in rows[0]['result']['contrasts']:
        out[k] = ci(rows, k, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    reasons = []
    for other in CONTROLS:
        v = out['extended_minus_'+other+'_signed_MSE']['CI95']
        if v is None or v[1] >= 0:
            reasons.append('MSE_not_supported_vs_'+other)
        for mode in ('full', 'matched'):
            v = out['extended_minus_'+other+'_'+mode+'_utility_percent']['CI95']
            if v is None or v[0] <= 0:
                reasons.append(mode+'_utility_not_supported_vs_'+other)
    e, o = out['extended'], out['original']
    if e['complete_support'] < o['complete_support']:
        reasons.append('complete_support_reduced')
    if e['known_label_violations'] > o['known_label_violations']:
        reasons.append('more_known_risk_violations')
    if e['violations'] > o['violations']:
        reasons.append('more_upper_risk_violations')
    if e['worst_easy_upper'] is None or o['worst_easy_upper'] is None or e['worst_easy_upper'] > o['worst_easy_upper']:
        reasons.append('worst_upper_risk_not_preserved')
    out['advance_to_transfer'] = not reasons
    out['failure_reasons'] = reasons
    out['all_heads_easy_supported_within_budget'] = (
        e['complete_support'] == len(rows) and e['defined_easy_risk'] == len(rows) and e['violations'] == 0)
    return out


def verify_row(r, anchor):
    checks = 0
    for k in ('checkpoint', 'parent_checkpoint', 'poisson_checkpoint', 'additive_checkpoint',
              'partition', 'training_ids_hash', 'validation_ids_hash', 'targets_hash', 'past_quality_hash'):
        checks += compare(r[k], anchor[k])
    assert r['known_training_prediction_hashes']['cost'] == r['known_training_prediction_hashes']['extended']
    assert r['new_training'] is False and r['training_known_rows'] > 0
    for arm in CONTROLS:
        for field in ('prediction_hashes', 'action_hashes'):
            checks += compare(r[field][arm], anchor[field][arm])
        checks += compare(r['result']['scores'][arm], anchor['result']['scores'][arm])
        checks += compare(r['result']['policies'][arm], anchor['result']['policies'][arm])
        checks += compare(r['result']['contrasts']['extended_minus_'+arm+'_signed_MSE'],
                          r['result']['scores']['extended']-r['result']['scores'][arm])
        for mode in ('full', 'matched'):
            a = r['result']['policies'][arm if mode == 'full' else arm+'_matched_extended']
            b = r['result']['policies']['extended' if mode == 'full' else 'extended_matched_'+arm]
            if mode == 'matched':
                assert a['selected_count'] == b['selected_count']
            den = r['result']['full_known_reference_mass']
            value = 100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/den if den > 0 else None
            checks += compare(value, r['result']['contrasts']['extended_minus_'+arm+'_'+mode+'_utility_percent'])
    slices = r['slices']
    assert slices['all']['rows'] == r['validation_rows']
    assert slices['any_quality_clipped']['rows'] == r['validation_changed_rows']
    checks += compare(slices['no_quality_clipped']['global_weighted_MSE_change'], 0.)
    checks += compare(slices['all']['global_weighted_MSE_change'], r['result']['contrasts']['extended_minus_cost_signed_MSE'])
    for parts in (('extended_selected', 'extended_unselected'), ('any_quality_clipped', 'no_quality_clipped')):
        for field in ('rows', 'unknown_rows', 'global_weighted_MSE_change', 'known_query_weight_mass'):
            checks += compare(slices['all'][field], math.fsum(slices[k][field] for k in parts))
    for s in slices.values():
        assert s['rows'] == s['known_rows']+s['unknown_rows']
        checks += compare(s['global_weighted_MSE_change'], math.fsum(s['moment_contributions']))
        checks += compare(s['global_weighted_MSE_change'], math.fsum(s['score_contributions']))
        if s['known_query_weight_mass'] == 0:
            assert s['conditional_MSE_change'] is None
        else:
            checks += compare(s['conditional_MSE_change'], s['global_weighted_MSE_change']/s['known_query_weight_mass'])
    return checks


def verify():
    reg = json.loads((PUBLIC/'registration.json').read_text())
    for path, digest in reg['bindings'].items():
        assert sha(ROOT/path) == digest
    assert sha(PUBLIC.parent/'european_cost_support_diagnostic_v1/verification.json') == reg['diagnostic_verification_sha256']
    complete = json.loads((PUBLIC/'complete.json').read_text())
    assert sha(PUBLIC/'summary.json') == complete['summary_sha256']
    receipt = json.loads((PARENT/'verification.json').read_text())
    assert sha(PARENT/'complete.json') == receipt['complete_sha256']
    anchors = {}
    for ref in json.loads((PARENT/'complete.json').read_text())['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        a = json.loads((ROOT/ref['path']).read_text())
        anchors[a['group'], a['head_seed']] = a
    rows, checks = [], 0
    for ref in complete['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text())
        assert r['registration_sha256'] == sha(PUBLIC/'registration.json')
        assert sha(ROOT/r['parent_checkpoint']['path']) == r['parent_checkpoint']['sha256']
        checks += verify_row(r, anchors[r['group'], r['head_seed']])
        rows.append(r)
    assert len(rows) == len({(r['group'], r['head_seed']) for r in rows}) == 72
    assert len({r['source'] for r in rows}) == 12
    cfg = json.loads((ROOT/'configs/m3w_european_leaf_quality_extension_v1.json').read_text())
    checks += compare(aggregate(rows, cfg), json.loads((PUBLIC/'summary.json').read_text()))
    assert complete['new_training'] is False
    assert complete['known_training_predictions_exact'] and complete['controls_exact_replay'] and complete['exact_inference_replay']
    return dict(independent_scalar_bootstrap_checks=checks, source_heads=72,
        summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json'),
        registration_sha256=sha(PUBLIC/'registration.json'), verifier_sha256=sha(Path(__file__)),
        known_training_predictions_unchanged=True, controls_preserved=True,
        new_training=False, independent_confirmation=False, deployment_changed=False)


if __name__ == '__main__':
    result = verify()
    payload = json.dumps(result, indent=2)+'\n'
    path = PUBLIC/'verification.json'
    if path.exists():
        assert path.read_text() == payload
    else:
        path.write_text(payload)
    print(payload)
