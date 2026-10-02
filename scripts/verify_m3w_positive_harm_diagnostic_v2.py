"""Independent scalar identities and locality-bootstrap diagnostic reduction."""
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.verify_m3w_past_quality_auxiliary import sha, compare, ci

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_positive_harm_diagnostic_v2'
PRIOR = PUBLIC.parent/'european_positive_harm_v1'
MOMENTS = ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')
SCORES = ('utility', 'all_risk_excess', 'easy_risk_excess')


def aggregate(rows, cfg):
    out = dict(groups=len(rows), source_localities=len({r['source'] for r in rows}), new_training=False,
        policy_selection=False, independent_confirmation=False, deployment_changed=False)
    def reduce(section, name=None):
        get = lambda r: r['diagnosis'][section] if name is None else r['diagnosis'][section][name]
        def interval(values):
            wrapped = [dict(source=r['source'], result=dict(contrasts=dict(value=v))) for r, v in zip(rows, values)]
            return ci(wrapped, 'value', cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
        d = dict(rows=sum(get(r)['rows'] for r in rows), unknown_rows=sum(get(r)['unknown_rows'] for r in rows))
        for key in ('known_query_weight_mass', 'global_weighted_MSE_change', 'conditional_MSE_change'):
            d[key] = interval([get(r)[key] for r in rows])
        for key, columns in (('moment_contributions', MOMENTS), ('score_contributions', SCORES)):
            d[key] = {col: interval([get(r)[key][j] for r in rows]) for j, col in enumerate(columns)}
        return d
    for section in ('cohorts', 'strata', 'projection_effects'):
        out[section] = {name: reduce(section, name) for name in rows[0]['diagnosis'][section]}
    for section in ('raw_error_change', 'additive_error_change'): out[section] = reduce(section)
    return out


def verify():
    complete = json.loads((PUBLIC/'complete.json').read_text())
    assert sha(PUBLIC/'summary.json') == complete['summary_sha256']
    registration = json.loads((PUBLIC/'registration.json').read_text())
    for path, digest in registration['bindings'].items(): assert sha(ROOT/path) == digest
    for path, digest in registration['preserved_v1_partial_results'].items(): assert sha(ROOT/path) == digest
    cfg = json.loads((ROOT/'configs/m3w_european_positive_harm_diagnostic_v2.json').read_text())
    rows = []; checks = 0
    for ref in complete['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        row = json.loads((ROOT/ref['path']).read_text()); rows.append(row); d = row['diagnosis']
        old = json.loads((PRIOR/'groups'/(row['group']+'_head'+str(row['head_seed'])+'.json')).read_text())
        checks += compare(d['cohorts']['all']['global_weighted_MSE_change'], old['result']['contrasts']['positive_minus_original_signed_MSE'])
        checks += compare(d['additive_error_change']['global_weighted_MSE_change'],
                          old['result']['scores']['additive']-old['result']['scores']['original'])
        num = d['numerical_reconciliation']
        checks += compare(num['registered_readout_delta'], num['fully_float64_algebraic_delta']+num['target_transform_rounding_effect'])
        checks += compare(num['v1_diagnostic_delta'], num['fully_float64_algebraic_delta']+num['scale_product_rounding_effect'])
        for arm in ('original', 'positive'):
            for field in ('global_weighted_MSE_change', 'known_query_weight_mass', 'rows', 'unknown_rows'):
                checks += compare(d['cohorts']['all'][field], math.fsum(d['cohorts'][arm+'_'+part][field] for part in ('selected', 'unselected')))
        for index in ('harm', 'easy_harm'):
            for field in ('global_weighted_MSE_change', 'rows', 'unknown_rows'):
                value = math.fsum(d['strata'][index+'_rate_'+name][field] for name in ('le_half', 'half_to_one', 'one_to_two', 'two_to_four', 'over_four', 'undefined'))
                checks += compare(d['cohorts']['all'][field], value)
        changes = [d['raw_error_change']['global_weighted_MSE_change'], d['projection_effects']['positive']['global_weighted_MSE_change'],
                   -d['projection_effects']['original']['global_weighted_MSE_change']]
        checks += compare(d['cohorts']['all']['global_weighted_MSE_change'], math.fsum(changes))
        all_slices = [d['raw_error_change'], d['additive_error_change']]
        all_slices += [s for key in ('cohorts', 'strata', 'projection_effects') for s in d[key].values()]
        for s in all_slices:
            checks += compare(s['global_weighted_MSE_change'], math.fsum(s['moment_contributions']))
            checks += compare(s['global_weighted_MSE_change'], math.fsum(s['score_contributions']))
            assert s['known_rows']+s['unknown_rows'] == s['rows']
            assert s['moment_contributions'][2] == s['moment_contributions'][3] == 0.
            if s['known_query_weight_mass'] == 0: assert s['conditional_MSE_change'] is None
    assert len(rows) == 72 and len({r['source'] for r in rows}) == 12
    checks += compare(aggregate(rows, cfg), json.loads((PUBLIC/'summary.json').read_text()))
    return dict(independent_scalar_bootstrap_checks=checks, source_heads=72,
        summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json'),
        exact_inference_replayed=complete['exact_raw_inference_replay'],
        original_readout_replayed=complete['exact_readout_replay'],
        new_training=False, independent_confirmation=False)


if __name__ == '__main__':
    result = verify(); text = json.dumps(result, indent=2)+'\n'; path = PUBLIC/'verification.json'
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    print(text)
