"""Describe all frozen bins; no policy fitting, filtering or threshold search."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_fixed_floor_slices as run
from scripts.report_m3w_fixed_floor_tail import ci


def contrast(a, b, sites, cfg):
    rows = [dict(site=s, metric={'difference': None if a[s] is None or b[s] is None else a[s]-b[s]}) for s in sites]
    return run.parent.inter.paired_localities(rows, sites, 'difference', cfg['bootstrap_resamples'], cfg['bootstrap_seed'])


def main():
    d = json.loads((run.PUBLIC/'summary.json').read_text())
    cfg = json.loads((ROOT/run.CONFIG).read_text()); s = d['summary']; sites = d['identity']['sites']
    a = s['held']['excess']; contrasts = {}
    for label in a:
        contrasts[label] = {k: contrast(a[label][k]['by_site'], s['held']['mse'][label][k]['by_site'], sites, cfg)
            for k in ('score_MSE', 'net_gain_percent', 'harm_percent', 'selected_score_MSE')}
    shares = {}
    all_sums = a['all/all']['sums_by_site']
    for label, item in a.items():
        shares[label] = {}
        for key in ('harm_sum', 'oracle_benefit_sum', 'eligible_oracle_benefit_sum', 'sq_error_sum'):
            by_site = {site: (100*item['sums_by_site'][site][key]/all_sums[site][key] if all_sums[site][key] > 0 else None) for site in sites}
            shares[label][key+'_share_percent'] = contrast(by_site, dict.fromkeys(sites, 0.), sites, cfg)
    completion = json.loads((run.PUBLIC/'completion.json').read_text())
    edge_rows = []
    for ref in completion['groups']:
        assert run.parent.artifact(ROOT/ref['path']) == ref
        edge_rows.append(json.loads((ROOT/ref['path']).read_text())['metadata']['edges'])
    ties = {k: sum(r[k][0] == r[k][1] for r in edge_rows) for k in edge_rows[0]}
    run.parent.immutable_json(run.PUBLIC/'diagnosis.json', dict(result_source='fresh_post_registration_descriptive_reduction',
        contrasts=contrasts, excess_shares=shares, policy_changed=False, multiple_comparison_adjusted=False,
        independent_roles_read=False))
    run.parent.immutable_json(run.PUBLIC/'bin_ties.json', dict(equal_quantile_boundary_groups=ties,
        result_source='fresh_description_of_frozen_fitting_cutpoints', deployment_changed=False))
    lines = ['# Frozen-Model Source-Gap Results', '', '## Material Passport', '',
        'Fresh diagnostics of108frozen paired heads; no new training or threshold selection.',
        'Twelve exposed source-training localities, three forecaster seeds. Independent roles remain closed.',
        'Nested held development is not independent confirmation. All predictions and actions match the sealed parent.', '',
        '## Whole-Role Comparison', '',
        '| Role / policy | Signed-score MSE | Eligible MSE | Selected MSE | Net floor gain % | Selected harm % | Intervention % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for role in ('fit', 'held'):
        for policy in ('mse', 'excess'):
            item = s[role][policy]['all/all']
            lines.append('| '+role+'/'+policy+' | '+' | '.join(ci(item[k]) for k in
                ('score_MSE', 'eligible_score_MSE', 'selected_score_MSE', 'net_gain_percent', 'harm_percent', 'intervention_percent'))+' |')
    lines += ['', 'Costs use each fitting head\'s training-only reference-cost scale. Ratios first pool dependent views within locality,',
        'then bootstrap12fixed locality means3,000times. This is NOT the parent primary\'s mean of view ratios.',
        'It cannot repair the parent\'s14empty-view failure. In-sample fitting values are not validation evidence.', '',
        '## All Causal Slices', '',
        'Low/middle/high boundaries use equal-source fitting-only25th/75th percentiles. Ties may leave empty bins.',
        'Each cell is a development estimate with an unadjusted95%locality interval; not simultaneous inference.', '',
        '| Axis / bin | New score MSE | New minus MSE score MSE | New harm % | Selected harm share % | Eligible oracle gain % | New floor gain % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label, item in a.items():
        if 'eval_only' in label or label == 'all/all': continue
        lines.append('| '+label+' | '+' | '.join((ci(item['score_MSE']), ci(contrasts[label]['score_MSE']),
            ci(item['harm_percent']), ci(shares[label]['harm_sum_share_percent']),
            ci(item['eligible_oracle_gain_percent']), ci(item['net_gain_percent'])))+' |')
    lines += ['', '## Evaluation-Only Strata', '',
        'Future completeness and realized reference errors below never enter inputs or deployment decisions.', '',
        '| Evaluation stratum | Known row-views | Selected known row-views | Unknown selected row-views | Harm % | Harm share % | Score MSE | New minus MSE score MSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, item in a.items():
        if 'eval_only' not in label: continue
        counts = [int(sum(v[k] for v in item['sums_by_site'].values())) for k in ('known', 'selected_known', 'unknown_selected')]
        lines.append('| '+label+' | '+' | '.join(map(str, counts))+' | '+' | '.join((ci(item['harm_percent']),
            ci(shares[label]['harm_sum_share_percent']), ci(item['score_MSE']), ci(contrasts[label]['score_MSE'])))+' |')
    lines += ['', 'Counts are repeated role/seed/producer views, not independent trajectories. Unknown outcomes cannot certify safety.',
        'Undefined cells retain the full roster; per-locality numbers and denominators are in summary.json.', '',
        '## Tied Fitting Boundaries', '',
        'Equal25th/75th cutpoints by axis (out of108groups): '+json.dumps(ties)+'.',
        'In particular, a high clipping-fraction bin can include zero clipping when both fitted cuts are zero.',
        'That degeneracy is not evidence that clipping damaged every row. Empty bins are retained, not reassigned.', '',
        '## Per-Locality Held Results', '',
        '| Locality | MSE score error | New score error | New minus MSE | New selected harm % | New floor gain % |',
        '|---|---:|---:|---:|---:|---:|']
    def number(x): return 'undefined' if x is None else f'{x:.6f}'
    for site in sites:
        values = (s['held']['mse']['all/all']['score_MSE']['by_site'][site], a['all/all']['score_MSE']['by_site'][site],
            contrasts['all/all']['score_MSE']['by_site'][site], a['all/all']['harm_percent']['by_site'][site],
            a['all/all']['net_gain_percent']['by_site'][site])
        lines.append('| '+site+' | '+' | '.join(map(number, values))+' |')
    lines += ['', '## Limits', '',
        'This localizes associations, not causal mechanisms or irreducible uncertainty. No bin is promoted to a policy.',
        'Complete detector labels are still silver. Partial future labels can change error interpretation but are unavailable at inference.',
        'Inside a radial support gate does not mean conditional exchangeability or distributional overlap.',
        'Image-local obs8/pred12 rawstride12 only; no metric/seconds/physical-safety/true3D/foundation claim.',
        'No deployment change, independent confirmation, Stage5C execution or SMC.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
    for ax, labels, key, title in (
        (axes[0], ['feature_radius_over_limit/'+k for k in ('low', 'middle', 'high')], 'score_MSE', 'Held score error by feature radius'),
        (axes[1], ['rollout_disagreement_over_extent/'+k for k in ('low', 'middle', 'high')], 'harm_percent', 'Selected harm by rollout disagreement (%)'),
        (axes[2], ['label_completeness_eval_only/'+k for k in ('partial', 'complete')], 'harm_percent', 'Selected harm by label completeness (%)')):
        for j, policy in enumerate(('mse', 'excess')):
            for i, label in enumerate(labels):
                val = s['held'][policy][label][key]
                if val['point'] is None:
                    ax.text(i+(j-.5)*.18, .04+j*.07, 'NA', transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
                else:
                    point = val['point']; lo, hi = val['ci95']
                    ax.errorbar(i+(j-.5)*.18, point, yerr=[[point-lo], [hi-point]], fmt='o', capsize=3,
                        color=('#555555', '#167d73')[j], label=policy if i==0 else None)
        ax.set_xticks(range(len(labels)), [k.split('/')[-1] for k in labels]); ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        if key == 'harm_percent': ax.axhline(2, color='#ba4340', linestyle='--', linewidth=1)
        ax.legend(fontsize=8)
    fig.suptitle('Frozen-model source diagnosis: development only, no policy change')
    fig.savefig(run.PUBLIC/'source_gap.png', dpi=150, metadata={'Software': 'M3W frozen source slices'})
    plt.close(fig)
    print(json.dumps(dict(groups=d['groups'], records=d['records'], whole_role_difference=contrasts['all/all'])))


if __name__ == '__main__': main()
