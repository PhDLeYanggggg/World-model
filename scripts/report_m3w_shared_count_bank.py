"""Publish only the complete, separately replayed common-count bank control."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_shared_count_bank as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups


def build():
    public = run.PUBLIC
    complete, summary, verification = [json.loads((public/n).read_text()) for n in
        ('complete.json', 'summary.json', 'verification.json')]
    if (not verification['all72groups_exact']
            or not verification['all144legacy_full_policy_controls_exact']
            or verification['summary_sha256'] != run.sha(public/'summary.json')
            or verification['complete_sha256'] != run.sha(public/'complete.json')
            or complete['summary_sha256'] != run.sha(public/'summary.json')):
        raise ValueError('Complete separately verified readout required')
    reg = json.loads(run.REG.read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest:
            raise ValueError('Frozen common-count scientific dependency changed: '+rel)
    for name in ('complete', 'summary', 'verification'):
        if reg[f'parent_{name}_sha256'] != run.sha(run.parent.PUBLIC/(name+'.json')):
            raise ValueError('Frozen parent readout changed')
    groups = aggregate_groups(run.NAME)
    rows = [groups[Path(ref['path']).name] for ref in complete['groups']]
    cfg = json.loads(run.CONFIG.read_text())
    if run.evaluation.summarize(rows, reg['contexts'], cfg) != summary:
        raise ValueError('Canonical complete summary reconstruction differs')
    reasons = {a: Counter() for a in run.api.ARMS[2:]}
    nonoptimal, warm, failures = Counter(), Counter(), Counter()
    for row in rows:
        allocation = row['result']['allocation']
        nonoptimal.update(allocation['nonoptimal_queries'])
        warm.update(allocation['neural_warm_start_counts'])
        for a in reasons:
            reasons[a].update(allocation['reason_counts'][a])
        for record in row['result']['recordings'].values():
            failures.update(k for k, passed in record['neural_joint']['budget_status']['conditions'].items()
                if not passed)
    seed_effects = {}
    for seed in cfg['seeds']:
        subset = [r for r in rows if r['seed'] == seed]
        seed_effects[str(seed)] = {k: run.evaluation.bank.legacy.interval(subset,
            lambda r, k=k: r['comparisons']['neural_joint_minus_physical_joint']['full'][k], cfg)
            for k in ('known_pp', 'lower_pp')}
    return dict(summary=summary, complete=complete, rows=rows,
        reason_counts={a: dict(v) for a, v in reasons.items()},
        nonoptimal_queries=dict(nonoptimal), warm_start_counts=dict(warm),
        primary_failure_conditions=dict(failures), seed_effects=seed_effects,
        verification_runtime=json.loads((public/'replay_runtime.json' if (public/'replay_runtime.json').exists()
            else run.PRIVATE/'verification_runtime.json').read_text()))


def fmt(v):
    if v['CI95'] is None:
        return 'not estimable'
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"


def render(value):
    s, c = value['summary'], value['complete']
    lines = ['# Reoptimized physical and neural banks at a common query count', '',
        '## Provenance', '',
        'The protocol, code and allocation checks were committed before the real pilot and these outcomes.',
        'fresh_run: current causal inference, both bank optimizations, complete-policy scoring and bootstrap.',
        'cached_verified: 18 fixed unit-admission cost heads, frozen history forecasts and source packets.',
        'The old joint choices are hash-verified cached controls, not newly solved again.',
        'No new neural training. All 72 groups and 144 legacy full-policy controls reproduce in a separate full replay.',
        f"Readout PID {c['pid']}: {c['seconds']:.3f}s, peak RSS {c['peak_RSS_bytes']}B.",
        f"Replay PID {value['verification_runtime']['pid']}: {value['verification_runtime']['seconds']:.3f}s.",
        'SOURCE fitting/resubstitution only. These are not independent efficacy or deployment results.', '',
        '## Controlled question', '',
        'Does a frozen neural forecast add value after the physical-only controller is also optimized?',
        'Every current recording/frame query uses the physical independent policy\'s intervention count k.',
        'All four controls use exactly k. Zero-count queries keep the unchanged floor in every control.',
        'Six physical actions: constant position, causal finite-difference velocity, damping 0.90 and 0.97,',
        'history OLS4 velocity and history OLS8 velocity. The expanded bank adds the frozen history forecaster.',
        'The trimmed neural independent arm keeps its k largest predicted gains, with canonical row-ID ties.',
        'Both joint optimizations use the same all/easy signed predicted 2% budgets and fixed costs.',
        'The neural joint solver retains the better feasible warm start of optimized physical and trimmed neural.',
        'Predicted bank nesting is checked for every query; it is not an observed-risk guarantee.', '',
        '## Full-cohort utility and original recording risk', '',
        'ADE is computed over available future labels. Values are locality-averaged gain over the frozen floor.',
        '95% intervals: 3,000 locality bootstrap draws after context/head-seed averaging over 12 exposed localities.',
        'These conditional exploratory intervals do not adjust for the earlier development search.', '',
        '|Policy|Known ADE gain % [95% CI]|Supported recording views|Known all-risk violations|Known easy violations|Undefined easy|Selected row views|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in run.evaluation.NAMES:
        r = s['safety'][a]
        lines.append(f"|{a}|{fmt(s['accuracy'][a])}|{r['supported_record_views']}/{r['required_record_views']}|"
            f"{r['known_all_risk_violations']}|{r['known_easy_risk_violations']}|{r['undefined_easy_record_views']}|{r['selected']}|")
    lines += ['', 'Legacy policies retain their original seven-action counts and are contextual controls,',
        'not common-count primary comparators. The no-switch floor\'s selected-risk ratio is undefined,',
        'not a safety pass. Same-count contrasts establish utility comparisons, not deployment qualification.', '',
        '|Prespecified contrast|Known increment pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for name, v in s['comparisons'].items():
        lines.append(f"|{name}|{fmt(v['full_known_pp'])}|{fmt(v['full_lower_pp'])}|")
    v = s['joint_specific_neural_increment']
    lines += ['', 'The primary is neural_joint minus physical_joint. The four-policy interaction contrast is secondary:',
        f"known {fmt(v['known_pp'])}; completion-lower {fmt(v['lower_pp'])}.",
        'It asks whether joint allocation changes the incremental value of the neural bank compared with',
        'the independent controls. Its own unknown bound pairs positive/negative forecast terms and takes',
        'the smaller of two valid Lipschitz disagreement bounds, allowing shared terms to cancel.',
        'It is not a subtraction of two lower bounds and not a causal effect on real agent behavior.', '',
        '## Numerical and safety checks', '',
        f"Repeated query views: {s['query_views']}; multi-agent: {s['multiagent_query_views']}; zero-count: {s['zero_count_query_views']}.",
        f"Solver reasons: {value['reason_counts']}.",
        f"Nonoptimal query views: {value['nonoptimal_queries']}.",
        f"Neural feasible warm starts: {value['warm_start_counts']}.",
        f"Overlapping neural-joint recording failure conditions: {value['primary_failure_conditions']}.",
        'Any solver rejection retains its checked feasible anchor. A nonoptimal fallback is reported,',
        'not renamed an exact optimum. Replay agreement does not turn a failed risk gate into a pass.', '',
        '|Head seed (same frozen forecaster)|Primary known pp [95% CI]|Primary completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for seed, v in value['seed_effects'].items():
        lines.append(f"|{seed}|{fmt(v['known_pp'])}|{fmt(v['lower_pp'])}|")
    lines += ['', '## Fitting gates', '']
    lines += [f"- {k}: {'pass' if passed else 'fail'}" for k, passed in s['fitting_gates'].items()]
    lines += ['', f"{sum(s['fitting_gates'].values())}/{len(s['fitting_gates'])} fitting checks; advancement {s['fitting_advancement_gate']}.",
        'The complete-policy recording criterion remains the original 2% requirement. No safety threshold,',
        'count anchor, label cohort or model is chosen from these readout results.', '',
        '## Limits', '',
        'The same 149,510 unique source queries, 74 recordings and 12 localities supply repeated views.',
        'Three cost-head seeds are not three new forecasters or independent datasets. Query targets are',
        'not guaranteed to cover all visible agents. No pairwise interaction term is optimized.',
        'Wholly-unknown completion bounds do not also bound missing interior labels of partial ADE rows.',
        'Full-roster FDE is not estimable when a required endpoint is unavailable. Per-group FDE values',
        'remain in the aggregate bundle rather than inventing a whole-roster value.',
        'Future labels are scoring-only; inference takes past-only features, fixed forecasts and current query IDs.',
        'Current task is obs8/pred12 at raw stride12, not supplementary raw-frame t+50.',
        'No independent confirmation, deployment change or submission-readiness claim.',
        'Detector-silver, image-local coordinates are not metric, seconds-level, human gold, true3D or foundation evidence.',
        'Stage5C is not executed and SMC is disabled.', '']
    return '\n'.join(lines)


def main():
    value = build(); s = value['summary']; public = run.PUBLIC
    run.once(public/'replay_runtime.json', value['verification_runtime'])
    (public/'readout.md').write_text(render(value))
    run.write(public/'mechanism_diagnostics.json', {k: value[k] for k in
        ('reason_counts', 'nonoptimal_queries', 'warm_start_counts', 'primary_failure_conditions', 'seed_effects')})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = run.api.ARMS
    labels = ('Physical independent', 'Neural trimmed', 'Physical joint', 'Neural joint')
    for i, a in enumerate(names):
        v = s['accuracy'][a]; low, high = v['CI95']
        axes[0].errorbar(v['point'], i, xerr=[[v['point']-low], [high-v['point']]],
            fmt='o', capsize=4, color='#007C91' if 'neural' in a else '#666666')
    axes[0].set_yticks(range(4), labels); axes[0].invert_yaxis()
    axes[0].set_xlabel('Known ADE gain over floor (%)'); axes[0].grid(axis='x', alpha=.2)
    axes[0].axvline(0, color='#999999', linewidth=.8)
    contrasts = ('neural_joint_minus_physical_joint', 'neural_trimmed_minus_physical_independent',
        'physical_joint_minus_physical_independent', 'neural_joint_minus_neural_trimmed')
    text = ('Neural vs physical joint (primary)', 'Neural vs physical independent',
        'Joint vs independent, physical', 'Joint vs independent, neural')
    for i, name in enumerate(contrasts):
        v = s['comparisons'][name]['full_lower_pp']; low, high = v['CI95']
        axes[1].errorbar(v['point'], i, xerr=[[v['point']-low], [high-v['point']]],
            fmt='o', capsize=4, color='#007C91' if i == 0 else '#666666')
    axes[1].set_yticks(range(4), text); axes[1].invert_yaxis()
    axes[1].set_xlabel('Completion-lower increment (pp)'); axes[1].grid(axis='x', alpha=.2)
    axes[1].axvline(0, color='#999999', linewidth=.8)
    fig.suptitle('Common query counts; fitting evidence only, not a safety certificate', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    folder = public/'figures'; folder.mkdir(exist_ok=True)
    path = folder/'shared_count_summary.svg'
    fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'shared_count_summary.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    run.write(public/'evidence.json', dict(summary_sha256=run.sha(public/'summary.json'),
        complete_sha256=run.sha(public/'complete.json'), verification_sha256=run.sha(public/'verification.json'),
        replay_runtime_sha256=run.sha(public/'replay_runtime.json'),
        report_code_sha256=run.sha(Path(__file__)), report_sha256=run.sha(public/'readout.md'),
        figure_sha256=run.sha(path), scientific_rescoring=False, new_training_updates=0,
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(report_generated=True, gates=s['fitting_gates'], deployment_changed=False)))


if __name__ == '__main__':
    main()
