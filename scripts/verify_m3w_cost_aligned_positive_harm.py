"""Independent scalar and locality-bootstrap reduction for the fixed cost control."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.verify_m3w_past_quality_auxiliary import sha, compare, ci

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cost_aligned_positive_harm_v1'
PRIOR = PUBLIC.parent/'european_positive_harm_v1'


def aggregate(rows, cfg):
    out = dict(groups=len(rows), new_cost_fits=len(rows), cached_control_arms=3, new_tree_splits=0,
        new_neural_updates=0, independent_confirmation=False, transfer_evaluated=False, deployment_changed=False)
    for name in rows[0]['result']['policies']:
        values = [r['result']['policies'][name] for r in rows]
        easy = [r['easy_selected_risk_upper'] for r in values if r['easy_selected_risk_upper'] is not None]
        out[name] = dict(selected=sum(r['selected_count'] for r in values), unknown_selected=sum(r['selected_unknown'] for r in values),
            complete_support=sum(r['finite_completion_supported'] for r in values), defined_easy_risk=len(easy),
            violations=sum(r > .02+1e-12 for r in easy), worst_easy_upper=max(easy) if easy else None,
            known_label_violations=sum(r['selected_known_easy_harm_mass']/r['selected_known_easy_reference_mass'] > .02+1e-12
                for r in values if r['selected_known_easy_reference_mass'] > 0))
    for k in rows[0]['result']['contrasts']: out[k] = ci(rows, k, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    reasons = []
    for other in ('original', 'additive', 'poisson'):
        if out['cost_minus_'+other+'_signed_MSE']['CI95'][1] >= 0: reasons.append('MSE_not_supported_vs_'+other)
        for mode in ('full', 'matched'):
            if out['cost_minus_'+other+'_'+mode+'_utility_percent']['CI95'][0] <= 0:
                reasons.append(mode+'_utility_not_supported_vs_'+other)
    a, b = out['cost'], out['original']
    if a['complete_support'] < b['complete_support']: reasons.append('complete_support_reduced')
    if a['violations'] > b['violations']: reasons.append('more_upper_risk_violations')
    if a['worst_easy_upper'] is None or b['worst_easy_upper'] is None or a['worst_easy_upper'] > b['worst_easy_upper']:
        reasons.append('worst_upper_risk_not_preserved')
    out['advance_to_transfer'] = not reasons; out['failure_reasons'] = reasons
    return out


def main():
    completed = json.loads((PUBLIC/'complete.json').read_text())
    assert sha(PUBLIC/'summary.json') == completed['summary_sha256']
    registration = json.loads((PUBLIC/'registration.json').read_text())
    for path, digest in registration['bindings'].items(): assert sha(ROOT/path) == digest
    cfg = json.loads((ROOT/'configs'/('m3w_'+PUBLIC.name+'.json')).read_text())
    prior_complete = json.loads((PRIOR/'complete.json').read_text())
    parent_refs = {r['path']: r['sha256'] for r in prior_complete['groups']}
    rows = []; checks = 0
    for ref in completed['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text()); rows.append(r)
        old_path = PRIOR/'groups'/(r['group']+'_head'+str(r['head_seed'])+'.json')
        assert sha(old_path) == parent_refs[str(old_path.relative_to(ROOT))]
        old = json.loads(old_path.read_text())
        for key in ('training_ids_hash', 'validation_ids_hash', 'targets_hash', 'past_quality_hash', 'partition'):
            checks += compare(r[key], old[key])
        for arm, old_arm in (('original', 'original'), ('additive', 'additive'), ('poisson', 'positive')):
            assert r['prediction_hashes'][arm] == old['prediction_hashes'][old_arm]
            assert r['action_hashes'][arm] == old['action_hashes'][old_arm]
            checks += compare(r['result']['policies'][arm], old['result']['policies'][old_arm])
            checks += compare(r['result']['scores'][arm], old['result']['scores'][old_arm])
        fit = r['training']; assert fit['maximum_gradient'] <= 1e-7 and fit['train_relative_mean_error'] < 1e-9
        for a, b in zip(fit['training_loss']['before'], fit['training_loss']['after']): assert b <= a+1e-9
        assert r['diagnostic']['exponential_guard_coordinates'] == 0
        for other in ('original', 'additive', 'poisson'):
            checks += compare(r['result']['contrasts']['cost_minus_'+other+'_signed_MSE'],
                r['result']['scores']['cost']-r['result']['scores'][other])
            for mode in ('full', 'matched'):
                a = r['result']['policies'][other if mode == 'full' else other+'_matched_cost']
                b = r['result']['policies']['cost' if mode == 'full' else 'cost_matched_'+other]
                value = 100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/r['result']['full_known_reference_mass']
                checks += compare(value, r['result']['contrasts']['cost_minus_'+other+'_'+mode+'_utility_percent'])
                if mode == 'matched': assert a['selected_count'] == b['selected_count']
    assert len(rows) == 72 and len({r['source'] for r in rows}) == 12
    summary = json.loads((PUBLIC/'summary.json').read_text()); checks += compare(aggregate(rows, cfg), summary)
    manifest = json.loads((PUBLIC/'checkpoint_manifest.json').read_text())['checkpoints']
    assert sorted(manifest, key=lambda r: r['group']) == sorted([r['checkpoint'] for r in rows], key=lambda r: r['group'])
    assert sum(r['bytes'] for r in manifest) == completed['checkpoint_bytes'] and completed['owned_remote_weights_verified'] == 72
    receipt = dict(independent_scalar_bootstrap_checks=checks, source_heads=72,
        summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json'),
        checkpoint_manifest_sha256=sha(PUBLIC/'checkpoint_manifest.json'),
        exact_fit_replay=completed['exact_fit_replay'], exact_inference_replay=completed['exact_inference_replay'],
        failure_reasons=summary['failure_reasons'], independent_confirmation=False)
    dest = PUBLIC/'verification.json'; text = json.dumps(receipt, indent=2)+'\n'
    if dest.exists(): assert dest.read_text() == text
    else: dest.write_text(text)
    print(text)


if __name__ == '__main__': main()
