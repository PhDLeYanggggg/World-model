"""Render the registered development contrast without promoting partial evidence."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_risk_priority_policy as run


BOUNDARIES = (
    'Only12 already-opened development localities; repeated roles and the three '
    'forecast seeds are not independent samples. Intervals are nominal3,000-draw '
    'paired-locality intervals, not simultaneous or independent-confirmation '
    'intervals. Independent selection/calibration/confirmation remain closed. '
    'The original selected-risk primary remains incomplete; a full-floor '
    'denominator cannot replace it. Image-local detector silver; obs8/pred12 '
    'with raw-frame stride12. No metric, seconds-level, human-gold, physical-safety, '
    'true3D, foundation, calibration-certificate or submission-readiness claim. '
    'Stage5C and SMC remain disabled.')


def interpret(summary):
    ci = summary['paired']['risk_priority_matched_vs_uncapped_matched']['ADE_gain_percent']['ci95']
    if ci is None:
        verdict = 'undefined_primary'
    elif ci[0] > 0:
        verdict = ('exploratory_screen_pass_only' if summary['gates']['exploratory_screen_pass']
                   else 'advantage_but_screen_failed')
    elif ci[1] < 0:
        verdict = 'negative_primary'
    else:
        verdict = 'no_resolved_primary_advantage'
    return dict(verdict=verdict, deployment_changed=False, independent_confirmation=False,
                formal_primary_replaced=False, submission_ready=False,
                stage5c_executed=False, smc_enabled=False)


def structural_support(rows):
    policies = ('raw_independent', 'supervised_independent')
    views = {p: {(r['group'], r['site']): r['metric']['intervention_rate']
                 for r in rows if r['policy'] == p} for p in policies}
    assert set(views[policies[0]]) == set(views[policies[1]])
    assert views[policies[0]], 'Missing fixed-control support'
    for p in policies:
        assert len(views[p]) == sum(r['policy'] == p for r in rows)
        assert all(v is not None and 0 <= v <= 1 for v in views[p].values())
    empty = {p: {key for key, rate in views[p].items() if rate == 0} for p in policies}
    forced = empty[policies[0]] | empty[policies[1]]
    return dict(total_dependent_views=len(views[policies[0]]),
        raw_empty_views=len(empty[policies[0]]), uncapped_empty_views=len(empty[policies[1]]),
        overlapping_empty_views=len(empty[policies[0]] & empty[policies[1]]),
        forced_undefined_selected_risk_views=len(forced),
        affected_localities=sorted({site for _, site in forced}),
        forced_view_keys=[list(key) for key in sorted(forced)],
        independent_sample_count=False, based_on_new_repair_outcomes=False,
        registered_screen_can_pass=not bool(forced))


def value(row):
    if row['point'] is None:
        return 'undefined'
    ci = row['ci95']
    return f"{row['point']:.7g}"+(f" [{ci[0]:.7g}, {ci[1]:.7g}]" if ci is not None else '')


def put(name, text):
    path = run.PUBLIC/name
    if path.exists():
        assert path.read_text() == text, 'Do not rewrite a completed readout'
    else:
        path.write_text(text)


def main():
    ident = run.identity()
    s = json.loads((run.PUBLIC/'summary.json').read_text())
    assert s['identity'] == ident
    for name in ('prediction_replay', 'evaluation_replay'):
        doc = json.loads((run.PUBLIC/(name+'.json')).read_text())
        assert doc['exact']
        if name == 'prediction_replay':
            assert doc['groups'] == 108
        else:
            for key in ('summary', 'details'):
                assert run.base.artifact(ROOT/doc[key]['path']) == doc[key]
    result = interpret(s)
    support = json.loads((run.PUBLIC/'structural_support.json').read_text())['support']
    assert s['worst_views']['risk_priority_matched']['undefined_selected_risk_views'] >= support['forced_undefined_selected_risk_views']
    run.immutable(run.PUBLIC/'interpretation.json', result)
    lines = ['# Risk-Priority Learning: Registered Development Readout', '',
        'Source: fresh_run paired Torch training, causal actions and development '
        'evaluation; upstream inputs/forecasters/floor/utility are cached_verified. '
        'All108 action groups and the complete numerical readout have exact replays. '
        'Independent confirmation: not_run.', '',
        'The only training change caps the auxiliary-gradient norm at half the '
        'direct-risk-gradient norm. Architecture, initialization, sampling, AdamW, '
        '2,000 updates/head, node budget and the2% risk budget are fixed. '
        'No threshold search or held-outcome model selection was performed. '
        'A pre-optimizer Euclidean gradient bound is not an AdamW-descent or safety guarantee.', '',
        f"**Verdict: {result['verdict']}. No deployment change.**", '',
        f"The frozen raw/uncapped anchors already force at least{support['forced_undefined_selected_risk_views']} "
        'dependent views to abstain under common-query-count matching. This was checked before the new '
        'readout; the defined-selected-risk gate cannot pass merely by changing the auxiliary gradient. '
        'These views and the registered gates are retained, not excluded after seeing results.', '',
        '## Registered Contrasts', '',
        'Primary: risk_priority_matched versus uncapped_matched. Each query has '
        'the same intervention count. Positive ADE gain and positive harm reduction '
        'are favorable. The all-floor harm column is only a diagnostic.', '',
        '| Contrast | ADE gain % [nominal95% CI] | All-floor harm reduction pp | Selected-risk reduction pp |',
        '|---|---:|---:|---:|']
    for name, metrics in s['paired'].items():
        lines.append(f"| {name} | {value(metrics['ADE_gain_percent'])} | "
            f"{value(metrics['all_reference_harm_reduction_pp'])} | {value(metrics['selected_harm_reduction_pp'])} |")
    lines += ['', '## Every Policy, Including Failures', '',
        '| Policy | ADE gain/floor % | Hard gain/floor % | Intervention fraction | Risk-violating views | Undefined-risk views | Worst easy gain/CV % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for policy in run.POLICIES:
        m, w = s['summary'][policy], s['worst_views'][policy]
        lines.append(f"| {policy} | {value(m['all_gain_floor'])} | {value(m['hard_gain_floor'])} | "
            f"{value(m['intervention_rate'])} | {w['risk_violating_views']} | "
            f"{w['undefined_selected_risk_views']} | {w['worst_easy_gain_CV']:.7g} |")
    lines += ['', 'An undefined selected-risk ratio is not zero risk and cannot pass the risk gate.', '',
        '## Risk Prediction Quality', '',
        '| Arm | Brier | Signed-risk MSE | Signed bias | Conditional reference MSE | Conditional harm MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, metrics in s['quality'].items():
        lines.append('| '+arm+' | '+' | '.join(value(metrics[key]) for key in
            ('Brier', 'signed_MSE', 'signed_bias', 'conditional_reference_MSE', 'conditional_harm_MSE'))+' |')
    lines += ['', '## Gates', '']
    lines += [f'- {key}: {value_}.' for key, value_ in s['gates'].items()]
    lines += ['', '## Scope', '', BOUNDARIES]
    put('results.md', '\n'.join(lines)+'\n')

    primary = s['paired']['risk_priority_matched_vs_uncapped_matched']
    rows = ['# Locality, Seed and Tail Results', '', BOUNDARIES, '',
        '| Locality | Primary ADE gain % | All-floor harm reduction pp | Matched intervention difference pp |',
        '|---|---:|---:|---:|']
    for site in primary['ADE_gain_percent']['by_site']:
        values = [primary[key]['by_site'][site] for key in
            ('ADE_gain_percent', 'all_reference_harm_reduction_pp', 'intervention_difference_pp')]
        rows.append('| '+site+' | '+' | '.join('undefined' if v is None else f'{v:.8g}' for v in values)+' |')
    rows += ['', '| Seed | Policy | ADE gain/floor % | Hard gain/floor % |', '|---|---|---:|---:|']
    for seed, policies in s['by_seed'].items():
        for policy in run.POLICIES:
            rows.append(f"| {seed} | {policy} | {value(policies[policy]['all_gain_floor'])} | "
                        f"{value(policies[policy]['hard_gain_floor'])} |")
    rows += ['', '| Policy | FDE gain/floor % | p95 ratio/floor | Unknown interventions/view | Abstaining views |',
             '|---|---:|---:|---:|---:|']
    for policy in run.POLICIES:
        m = s['summary'][policy]
        rows.append(f"| {policy} | {value(m['FDE_gain_floor'])} | {value(m['p95_ratio_to_floor'])} | "
            f"{value(m['unknown_interventions'])} | {s['worst_views'][policy]['abstaining_views']} |")
    put('locality_seed_quality.md', '\n'.join(rows)+'\n')

    # Deterministic aggregate figure; no sample images or row-level data are published.
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt': 'm3w-risk-priority-v1', 'svg.fonttype': 'none'})
    names = [pair[0]+'_vs_'+pair[1] for pair in run.CONTRASTS[:3]]
    labels = ['Risk-priority vs uncapped', 'Uncapped vs original', 'Risk-priority vs original']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
    for ax, key, title in zip(axes, ('ADE_gain_percent', 'all_reference_harm_reduction_pp'),
                            ('Matched-count ADE gain (%)', 'All-floor harm reduction (pp; diagnostic)')):
        for i, name in enumerate(names):
            record = s['paired'][name][key]
            if record['point'] is not None and record['ci95'] is not None:
                ax.plot(record['ci95'], [i, i], color='#197978', linewidth=2)
                ax.plot(record['point'], i, 'o', color='#197978')
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
        ax.set_title(title, fontsize=10); ax.tick_params(labelsize=9); ax.grid(axis='x', alpha=.2)
    fig.suptitle('12 development localities; nominal paired bootstrap intervals', fontsize=11)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='svg', metadata={'Date': None, 'Creator': 'M3W deterministic report'})
    dest = run.PUBLIC/'matched_contrasts.svg'
    if dest.exists():
        assert dest.read_bytes() == buffer.getvalue()
    else:
        dest.write_bytes(buffer.getvalue())
    preview = run.PRIVATE/'figures'; preview.mkdir(exist_ok=True)
    fig.savefig(preview/'matched_contrasts.png', dpi=150, metadata={'Software': 'M3W deterministic report'})
    plt.close(fig)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
