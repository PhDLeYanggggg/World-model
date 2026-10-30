"""Render only the complete, separately replayed fixed allocation experiment."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_frame_action_allocation as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups


def build():
    p = run.PUBLIC
    complete, summary, verification = [json.loads((p/n).read_text()) for n in
        ('complete.json', 'summary.json', 'verification.json')]
    if (not verification['all72groups_exact'] or not verification['all72prediction_action_full_metric_controls_exact']
            or not verification['all444record_oracles_exact']
            or verification['summary_sha256'] != run.sha(p/'summary.json')
            or verification['complete_sha256'] != run.sha(p/'complete.json')
            or complete['summary_sha256'] != run.sha(p/'summary.json')):
        raise ValueError('Complete separately verified scientific readout required')
    reg = json.loads(run.REG.read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest:
            raise ValueError('Frozen scientific code changed: '+rel)
    for ref in reg['cached_verified_models']:
        if run.sha(ROOT/ref['checkpoint']['path']) != ref['checkpoint']['sha256']:
            raise ValueError('Cached verified model changed')
    groups = aggregate_groups(run.NAME)
    rows = [groups[Path(ref['path']).name] for ref in complete['groups']]
    cfg = json.loads(run.CONFIG.read_text())
    if run.evaluation.summarize(rows, reg['contexts'], cfg) != summary:
        raise ValueError('Complete summary reconstruction differs')
    oracle = [r['result']['oracle_frame_check'] for r in rows if r['seed'] == 17]
    oracle_counts = {k: sum(v[k] for v in oracle) for k in ('queries', 'algebraic_budget_failures',
        'defined_all_support', 'defined_easy_support', 'positive_gain_queries')}
    for r in rows:
        key = (r['context'], r['source'])
        ref = next(v['result']['oracle_frame_check'] for v in rows
                   if (v['context'], v['source']) == key and v['seed'] == 17)
        if r['result']['oracle_frame_check'] != ref:
            raise ValueError('Label-aware bank opportunity must not depend on cost-head seed')
    failures, reasons = Counter(), {a: Counter() for a in run.api.ARMS[2:]}
    for row in rows:
        for record in row['result']['recordings'].values():
            v = record['joint_matched']
            failures.update(k for k, passed in v['budget_status']['conditions'].items() if not passed)
        for a in reasons:
            reasons[a].update(row['result']['allocation']['reason_counts'][a])
    return dict(summary=summary, complete=complete, rows=rows, oracle_counts=oracle_counts,
        primary_failure_conditions=dict(failures), reason_counts={a: dict(v) for a, v in reasons.items()},
        verification_runtime=json.loads((run.PRIVATE/'verification_runtime.json').read_text()))


def fmt(v):
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f},{v['CI95'][1]:+.6f}]" if v['CI95'] else 'not estimable'


def render(value):
    s, c = value['summary'], value['complete']
    lines = ['# Current-query multi-action allocation: complete fixed readout', '', '## Material Passport', '',
        'Fresh run:real causal inference, fixed allocation and label-aware diagnostic recomputation.',
        'Cached verified:18 already trained unit-admission cost heads, their unchanged forecast bank and inputs.',
        'New training updates:0. SOURCE fitting/resubstitution only; no independent or whole-pipeline OOF claim.', '',
        '## Controlled question', '',
        'Keep all predictors and costs fixed. Can pooling the all/easy predicted2% risk budget',
        'within the same current recording/frame query improve on independent decisions?',
        'The primary joint arm uses exactly the same intervention count in EVERY query.',
        'Uniform uses one action for all query targets; free joint allocation may change count.',
        'These are query targets, not guaranteed all visible agents. No pair-interaction model is trained.', '',
        f"Complete readout PID{c['pid']} takes{c['seconds']:.3f}s at peak RSS{c['peak_RSS_bytes']}B.",
        f"Separate full replay PID{value['verification_runtime']['pid']} takes{value['verification_runtime']['seconds']:.3f}s.",
        'All72 prediction/action/full-policy controls and all444 repeated record-oracle checks agree.',
        'Exact replay validates computation, not independent scientific confirmation.', '',
        '## Complete-policy utility and original risk', '',
        'Known ADE gains (%) versus the frozen floor;95% bootstrap CI over12localities,',
        '3,000 draws after context/seed averaging. Three head seeds are not independent recordings.', '',
        '|Decoder|Known ADE gain|Supported recording views|Known easy violations|Undefined easy support|Selected row views|',
        '|---|---:|---:|---:|---:|---:|']
    for a in run.evaluation.NAMES:
        r = s['safety'][a]
        lines.append(f"|{a}|{fmt(s['accuracy'][a])}|{r['supported_record_views']}/{r['required_record_views']}|"
            f"{r['known_easy_risk_violations']}|{r['undefined_easy_record_views']}|{r['selected']}|")
    lines += ['', '|Fixed contrast|Known increment pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|', '|---|---:|---:|']
    for name, r in s['comparisons'].items():
        lines.append(f"|{name}|{fmt(r['full_known_pp'])}|{fmt(r['full_lower_pp'])}|")
    lines += ['', 'The no-intervention floor has zero utility change; its selected risk denominator',
        'is undefined, not a selected-risk safety certificate. Empty matched controls are not',
        'estimable and count matching is utility-only. The original complete-policy recording',
        'risk requirements are not replaced by predicted budgets or a positive average.', '',
        '## Allocation mechanism and feasible opportunity', '',
        f"Repeated query views:{s['query_views']};multi-agent:{s['multiagent_query_views']};",
        f"changed primary queries:{s['changed_primary_query_views']};nonoptimal primary queries:{s['primary_solver_nonoptimal_query_views']}.",
        f"Solver/fast-path reasons:{value['reason_counts']}.",
        f"Seed-deduplicated label-aware frame diagnostic:{value['oracle_counts']}.",
        'The oracle is the same whole-record label-aware allocation, freshly reproduced then',
        'checked frame by frame. Algebraic budget satisfaction with no easy support is not',
        'a defined easy-risk pass. This is neither an inference policy nor a matched-rate optimum.',
        f"Primary overlapping recording failure conditions:{value['primary_failure_conditions']}.", '',
        '## Limits and fitting gates', '',
        'The same149,510unique queries,74recordings and12localities generate reused context/seed views.',
        'ADE uses available future positions. Full-roster FDE is not estimable when a required',
        'view lacks final labels. Completion sensitivity covers wholly unknown rows, not all',
        'missing interior positions in partially observed ADE labels. No outcomes are imputed.',
        'Pair-distance contraction in each aggregate group is a predicted-path proxy among',
        'query targets, normalized by a frozen fitting scale, not verified physical safety.', '']
    lines += [f"- {k}:{'pass' if v else 'fail'}" for k, v in s['fitting_gates'].items()]
    lines += ['', f"Fitting gates:{sum(s['fitting_gates'].values())}/{len(s['fitting_gates'])};advancement:{s['fitting_advancement_gate']}.",
        'Independent confirmation, deployment change and submission readiness remain false.',
        'Task:8observed/12requested steps at raw stride12, not supplemental rawt+50.',
        'Detector-silver top-down pixel/raw-frame data are not human gold,metric,seconds-level,',
        'true3D or foundation evidence. No Stage5C execution or SMC.', '']
    return '\n'.join(lines)


def main():
    value = build(); p = run.PUBLIC
    (p/'readout.md').write_text(render(value))
    run.write(p/'mechanism_diagnostics.json', {k: value[k] for k in
        ('oracle_counts', 'primary_failure_conditions', 'reason_counts')})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    names = run.api.ARMS; labels = ('Independent', 'Uniform', 'Joint matched', 'Joint free')
    s = value['summary']
    for i, a in enumerate(names):
        r = s['accuracy'][a]; low, high = r['CI95']
        axes[0].errorbar(r['point'], i, xerr=[[r['point']-low], [high-r['point']]],
            fmt='o', capsize=4, color='#007C91' if a == 'joint_matched' else '#555555')
    axes[0].set_yticks(range(4), labels); axes[0].invert_yaxis(); axes[0].axvline(0, color='#999999', linewidth=.8)
    axes[0].set_xlabel('Known ADE gain over floor (%)'); axes[0].set_title('Conditional locality bootstrap (95% CI)', fontsize=11)
    axes[0].grid(axis='x', alpha=.2)
    x = __import__('numpy').arange(4)
    for offset, key, label, color in ((-.24, 'supported_record_views', 'Supported', '#007C91'),
            (0, 'known_easy_risk_violations', 'Known easy violation', '#B44D48'),
            (.24, 'undefined_easy_record_views', 'Undefined easy', '#D4A43A')):
        axes[1].bar(x+offset, [s['safety'][a][key] for a in names], .23, label=label, color=color)
    axes[1].set_xticks(x, ('Independent', 'Uniform', 'Matched', 'Free'))
    axes[1].set_ylim(0, 480); axes[1].set_ylabel('Recording/context/seed views (444 required)')
    axes[1].legend(frameon=False, fontsize=8); axes[1].set_title('Original complete-policy risk requirement', fontsize=11)
    fig.suptitle('SOURCE fitting only; fixed predictors; no deployment promotion', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); folder = p/'figures'; folder.mkdir(exist_ok=True)
    path = folder/'allocation_summary.svg'; fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'allocation_summary.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    run.write(p/'evidence.json', dict(summary_sha256=run.sha(p/'summary.json'),
        complete_sha256=run.sha(p/'complete.json'), verification_sha256=run.sha(p/'verification.json'),
        report_code_sha256=run.sha(Path(__file__)), readout_sha256=run.sha(p/'readout.md'),
        figure_sha256=run.sha(path), scientific_rescoring=False,
        cached_models_not_new_training=True, independent_confirmation=False,
        deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(gates=value['summary']['fitting_gates'], report_generated=True)))


if __name__ == '__main__':
    main()
