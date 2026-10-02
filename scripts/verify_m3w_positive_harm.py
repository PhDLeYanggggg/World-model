"""Independent scalar and locality-bootstrap reduction, without model fitting."""
import json
from pathlib import Path
from scripts.verify_m3w_past_quality_auxiliary import sha, compare, ci

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_positive_harm_v1'


def aggregate(rows, cfg):
    out = dict(groups=len(rows), new_conditional_harm_fits=len(rows),
               cached_additive_controls=len(rows), new_tree_splits=0, new_neural_updates=0,
               independent_confirmation=False, transfer_evaluated=False, deployment_changed=False)
    for name in rows[0]['result']['policies']:
        v = [r['result']['policies'][name] for r in rows]
        ratios = [r['easy_selected_risk_upper'] for r in v if r['easy_selected_risk_upper'] is not None]
        observed = [r['selected_known_easy_harm_mass']/r['selected_known_easy_reference_mass']
                    for r in v if r['selected_known_easy_reference_mass'] > 0]
        out[name] = dict(selected=sum(r['selected_count'] for r in v),
            unknown_selected=sum(r['selected_unknown'] for r in v),
            complete_support=sum(r['finite_completion_supported'] for r in v),
            defined_easy_risk=len(ratios), violations=sum(x > .02+1e-12 for x in ratios),
            known_label_violations=sum(x > .02+1e-12 for x in observed),
            worst_easy_upper=max(ratios) if ratios else None)
    for key in rows[0]['result']['contrasts']:
        out[key] = ci(rows, key, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    reasons = []
    for other in ('original', 'additive'):
        c = out['positive_minus_'+other+'_signed_MSE']['CI95']
        if c is None or c[1] >= 0: reasons.append('MSE_not_supported_vs_'+other)
        for mode in ('full', 'matched'):
            c = out['positive_minus_'+other+'_'+mode+'_utility_percent']['CI95']
            if c is None or c[0] <= 0: reasons.append(mode+'_utility_not_supported_vs_'+other)
    p, o = out['positive'], out['original']
    if p['complete_support'] < o['complete_support']: reasons.append('complete_support_reduced')
    if p['violations'] > o['violations']: reasons.append('more_upper_risk_violations')
    if p['known_label_violations'] > o['known_label_violations']: reasons.append('more_known_risk_violations')
    if p['worst_easy_upper'] is None or o['worst_easy_upper'] is None or p['worst_easy_upper'] > o['worst_easy_upper']:
        reasons.append('worst_risk_not_preserved')
    out['gate_failure_reasons'] = reasons; out['advance_to_transfer'] = not reasons
    return out


def verify():
    complete = json.loads((PUBLIC/'complete.json').read_text())
    summary = json.loads((PUBLIC/'summary.json').read_text())
    assert sha(PUBLIC/'summary.json') == complete['summary_sha256']
    reg = json.loads((PUBLIC/'registration.json').read_text())
    for p, h in reg['bindings'].items(): assert sha(ROOT/p) == h
    cfg = json.loads((ROOT/'configs/m3w_european_positive_harm_v1.json').read_text())
    rows = []; checks = 0
    for ref in complete['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        row = json.loads((ROOT/ref['path']).read_text()); rows.append(row)
        fit = row['training']
        assert fit['maximum_gradient'] <= cfg['fit_settings']['tolerance']
        assert fit['train_relative_mean_error'] < 1e-9
        for b, a in zip(fit['training_loss']['before'], fit['training_loss']['after']):
            assert -1e-9 <= a <= b+1e-9
        for other in ('original', 'additive'):
            for mode in ('full', 'matched'):
                a = row['result']['policies'][other if mode == 'full' else other+'_matched_positive']
                b = row['result']['policies']['positive' if mode == 'full' else 'positive_matched_'+other]
                difference = 100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/row['result']['full_known_reference_mass']
                checks += compare(difference, row['result']['contrasts']['positive_minus_'+other+'_'+mode+'_utility_percent'])
                if mode == 'matched': assert a['selected_count'] == b['selected_count']
            checks += compare(row['result']['scores']['positive']-row['result']['scores'][other],
                              row['result']['contrasts']['positive_minus_'+other+'_signed_MSE'])
    assert len(rows) == 72 and len({r['source'] for r in rows}) == 12
    checks += compare(aggregate(rows, cfg), summary)
    o = summary['original']
    assert (o['selected'], o['unknown_selected'], o['complete_support'], o['violations'], o['known_label_violations']) == (95455, 918, 33, 7, 4)
    manifest = json.loads((PUBLIC/'checkpoint_manifest.json').read_text())
    assert len(manifest['checkpoints']) == 72
    expected = sorted((r['checkpoint']['group'], r['checkpoint']['sha256'], r['checkpoint']['bytes']) for r in rows)
    actual = sorted((r['group'], r['sha256'], r['bytes']) for r in manifest['checkpoints'])
    assert expected == actual and sum(v[2] for v in actual) == complete['checkpoint_bytes']
    return dict(independent_scalar_bootstrap_checks=checks, groups=72,
        gate_failure_reasons=summary['gate_failure_reasons'], independent_confirmation=False,
        raw_parameter_refit_replayed=complete['exact_fit_replay'], inference_replayed=complete['exact_inference_replay'],
        summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json'),
        checkpoint_manifest_sha256=sha(PUBLIC/'checkpoint_manifest.json'))


if __name__ == '__main__':
    result = verify(); path = PUBLIC/'verification.json'; content = json.dumps(result, indent=2)+'\n'
    if path.exists(): assert path.read_text() == content
    else: path.write_text(content)
    print(content)
