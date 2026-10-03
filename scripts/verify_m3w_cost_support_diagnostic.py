"""Independent scalar identities and locality-bootstrap reduction of frozen heads."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_m3w_past_quality_auxiliary import sha, compare, ci

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cost_support_diagnostic_v1'
PARENT = PUBLIC.parent/'european_cost_harm_newton_v1'
MOMENTS = ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')
SCORES = ('utility', 'all_risk_excess', 'easy_risk_excess')
DESCRIPTORS = ('zero_train_harm_fraction', 'zero_train_easy_harm_fraction',
               'outside_train_quality_box_fraction')


def aggregate(rows, cfg):
    def interval(values):
        wrapped = [dict(source=r['source'], result=dict(contrasts=dict(value=v)))
                   for r, v in zip(rows, values)]
        return ci(wrapped, 'value', cfg['bootstrap_resamples'], cfg['bootstrap_seed'])

    def reduce(get):
        d = dict(rows=sum(get(r)['rows'] for r in rows),
                 unknown_rows=sum(get(r)['unknown_rows'] for r in rows))
        for k in ('known_query_weight_mass', 'global_weighted_MSE_change', 'conditional_MSE_change'):
            d[k] = interval([get(r)[k] for r in rows])
        for key, columns in (('moment_contributions', MOMENTS), ('score_contributions', SCORES)):
            d[key] = {c: interval([get(r)[key][j] for r in rows]) for j, c in enumerate(columns)}
        return d

    out = dict(groups=len(rows), localities=len({r['source'] for r in rows}), new_training=False,
               independent_confirmation=False, policy_selection=False, deployment_changed=False)
    for section in ('cohorts', 'strata', 'projection_effects'):
        out[section] = {name: reduce(lambda r: r['diagnosis'][section][name])
                        for name in rows[0]['diagnosis'][section]}
    out['raw_error_change'] = reduce(lambda r: r['diagnosis']['raw_error_change'])
    out['training_projected_error_change'] = reduce(lambda r: r['training_projected_error_change'])
    out['tree_scores'] = {role: {name: interval([r['tree_scores'][role][name] for r in rows])
        for name in ('mean_tree_MSE_change', 'raw_ensemble_MSE_change', 'dispersion_change', 'fixed_penalty')}
        for role in ('train', 'validation')}
    out['support_cohorts'] = {name: dict(rows=sum(r['diagnosis']['support_cohorts'][name]['rows'] for r in rows),
        descriptor_mean={key: interval([r['diagnosis']['support_cohorts'][name]['descriptor_mean'][key]
                                       for r in rows]) for key in DESCRIPTORS})
        for name in rows[0]['diagnosis']['support_cohorts']}
    out['maximum_surrogate_reconstruction_residual'] = max(abs(r['surrogate_reconstruction_residual']) for r in rows)
    return out


def verify_row(row, anchor):
    checks = 0
    for k in ('checkpoint', 'training_ids_hash', 'validation_ids_hash', 'targets_hash'):
        checks += compare(row[k], anchor[k])
    for field in ('prediction_hashes', 'action_hashes'):
        for arm, digest in row[field].items():
            checks += compare(digest, anchor[field][arm])
    d = row['diagnosis']
    for arm in ('original', 'cost'):
        key = 'global_weighted_'+('old' if arm == 'original' else 'new')+'_MSE'
        checks += compare(d['cohorts']['all'][key], anchor['result']['scores'][arm])
        for field in ('global_weighted_MSE_change', 'known_query_weight_mass', 'rows', 'unknown_rows'):
            checks += compare(d['cohorts']['all'][field],
                math.fsum(d['cohorts'][arm+'_'+part][field] for part in ('selected', 'unselected')))
        checks += compare(d['cohorts'][arm+'_selected']['rows'], anchor['result']['policies'][arm]['selected_count'])
    checks += compare(d['cohorts']['all']['global_weighted_MSE_change'],
                      anchor['result']['contrasts']['cost_minus_original_signed_MSE'])
    partitions = [('outside_majority_leaf_quality_box', 'inside_majority_leaf_quality_box')]
    for channel in ('harm', 'easy_harm'):
        partitions.append(tuple(channel+'_rate_'+name for name in
            ('le_half', 'half_to_one', 'one_to_two', 'two_to_four', 'over_four', 'undefined')))
        partitions += [(channel+'_majority_train_eff_lt5', channel+'_majority_train_eff_ge5'),
                       (channel+'_all_train_zero', channel+'_any_train_positive')]
    for parts in partitions:
        for field in ('global_weighted_MSE_change', 'rows', 'unknown_rows'):
            checks += compare(d['cohorts']['all'][field], math.fsum(d['strata'][p][field] for p in parts))
    checks += compare(d['cohorts']['all']['global_weighted_MSE_change'], math.fsum([
        d['raw_error_change']['global_weighted_MSE_change'],
        d['projection_effects']['cost']['global_weighted_MSE_change'],
        -d['projection_effects']['original']['global_weighted_MSE_change']]))
    slices = [d['raw_error_change'], row['training_projected_error_change']]
    slices += [s for section in ('cohorts', 'strata', 'projection_effects') for s in d[section].values()]
    for s in slices:
        checks += compare(s['global_weighted_MSE_change'], math.fsum(s['moment_contributions']))
        checks += compare(s['global_weighted_MSE_change'], math.fsum(s['score_contributions']))
        assert s['known_rows']+s['unknown_rows'] == s['rows']
        if s['known_query_weight_mass'] == 0:
            assert s['conditional_MSE_change'] is None
        else:
            checks += compare(s['conditional_MSE_change'], s['global_weighted_MSE_change']/s['known_query_weight_mass'])
    for role, s in row['tree_scores'].items():
        for j in range(2):
            checks += compare(s['mean_tree_MSE'][j], s['raw_ensemble_MSE'][j]+s['tree_dispersion'][j])
        checks += compare(s['mean_tree_MSE_change'], s['raw_ensemble_MSE_change']+s['dispersion_change'])
        checks += compare(s['raw_ensemble_MSE_change'], s['raw_ensemble_MSE'][1]-s['raw_ensemble_MSE'][0])
    checks += compare(row['tree_scores']['validation']['raw_ensemble_MSE_change'], d['raw_error_change']['global_weighted_MSE_change'])
    loss = anchor['training']['training_loss']
    residual = row['tree_scores']['train']['mean_tree_MSE_change']+row['tree_scores']['train']['fixed_penalty']-(sum(loss['after'])-sum(loss['before']))
    checks += compare(residual, row['surrogate_reconstruction_residual'])
    assert abs(residual) <= 1e-6
    for s in d['support_cohorts'].values():
        assert 0 <= s['unknown_rows'] <= s['rows']
        if s['rows'] == 0:
            assert all(v is None for v in s['descriptor_mean'].values())
            assert s['train_effective_harm_mean'] is None
        else:
            assert all(v is not None and 0 <= v <= 1 for v in s['descriptor_mean'].values())
    return checks


def verify():
    complete = json.loads((PUBLIC/'complete.json').read_text())
    assert sha(PUBLIC/'summary.json') == complete['summary_sha256']
    reg = json.loads((PUBLIC/'registration.json').read_text())
    for path, digest in reg['bindings'].items():
        assert sha(ROOT/path) == digest
    assert sha(PARENT/'verification.json') == reg['parent_verification_sha256']
    parent_receipt = json.loads((PARENT/'verification.json').read_text())
    for name in ('complete', 'summary', 'checkpoint_manifest'):
        assert sha(PARENT/(name+'.json')) == parent_receipt[name+'_sha256']
    anchors = {}
    for ref in json.loads((PARENT/'complete.json').read_text())['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text())
        anchors[r['group'], r['head_seed']] = r
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
    cfg = json.loads((ROOT/'configs/m3w_european_cost_support_diagnostic_v1.json').read_text())
    checks += compare(aggregate(rows, cfg), json.loads((PUBLIC/'summary.json').read_text()))
    assert complete['new_training'] is False
    assert complete['exact_raw_inference_replay'] and complete['exact_readout_replay']
    return dict(independent_scalar_bootstrap_checks=checks, source_heads=72,
        summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json'),
        registration_sha256=sha(PUBLIC/'registration.json'), verifier_sha256=sha(Path(__file__)),
        exact_inference_replayed=True, frozen_actions_preserved=True,
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
