"""Report all four peer-context cells without selecting a favorable cell."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_peer_participation as run
from scripts.report_m3w_moving_rollout_history import reduction

source, api, PUBLIC = run.source, run.api, run.PUBLIC
CONTRASTS = {
    'primary_identity_all_peers': ('same_action', 'proposed_all'),
    'identity_active_peers': ('same_active', 'proposed_action'),
    'retain_peers_same_identity': ('same_active', 'same_action'),
    'retain_peers_proposed_identity': ('proposed_action', 'proposed_all')}


def comparisons(rows, jobs):
    expected = {(j['key'], arm) for j in jobs for arm in api.ARMS}
    actual = [(r['job'], r['arm']) for r in rows]
    if (len(jobs) != 3 or len(rows) != 12 or len(set(actual)) != 12 or set(actual) != expected
            or any(r['step'] != 1000 or r['parameters'] != 76548 for r in rows)):
        raise ValueError('Twelve unique matched factorial cells required')
    if any(r['result_source'] != ('fresh_run' if r['arm'] in api.NEW_ARMS else 'cached_verified') for r in rows):
        raise ValueError('Do not relabel cached controls as fresh fits')
    result = []
    for job in jobs:
        for name, arms in CONTRASTS.items():
            a, b = (next(r for r in rows if r['job'] == job['key'] and r['arm'] == arm) for arm in arms)
            result.append(dict(job=job['key'], contrast=name, control=arms[0], treatment=arms[1],
                mean_MSE_reduction_percent=reduction(a['final_fitting']['mean_MSE'], b['final_fitting']['mean_MSE']),
                **{k+'_reduction_percent':reduction(a['joint_attribution'][k], b['joint_attribution'][k])
                    for k in ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')}))
    return result


def evidence_status(analysis, screen):
    held = analysis['status'] == 'fresh_run_exposed_development'
    if held != screen['followup_readout_allowed'] or (not held and ('held_policy' in analysis or 'contrasts' in analysis)):
        raise ValueError('Readout must agree with fixed primary screen')
    return dict(held_readout='fresh_run_exposed_development' if held else 'not_run',
        original_risk_pass=analysis['held_policy']['proposed_all_veto']['safety']['full_original_risk_pass'] if held else 'not_run',
        independent_confirmation=False, deployment_changed=False, submission_ready=False)


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt':'moving_peer_participation_v1', 'svg.fonttype':'none', 'font.size':9})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(left=.08, right=.985, bottom=.09, top=.82, hspace=.5, wspace=.42)
    jobs = sorted({r['job'] for r in rows}, key=lambda x:int(x.rsplit('critic', 1)[1]))
    colors = dict(same_action='#606060', same_active='#C04C35', proposed_action='#5878A0', proposed_all='#148479')
    for col, job in enumerate(jobs):
        by = {r['arm']:r for r in rows if r['job'] == job}
        keys = ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')
        for i, arm in enumerate(api.ARMS):
            t = by[arm]['trace']
            axes[0, col].plot([r['step'] for r in t], [r['total'] for r in t], color=colors[arm], label=arm)
            v = [by[arm]['joint_attribution'][k]/by['same_action']['joint_attribution'][k]
                if by['same_action']['joint_attribution'][k] > 0 else float('nan') for k in keys]
            axes[1, col].bar(source.np.arange(3)+(i-1.5)*.19, v, width=.19, color=colors[arm])
        axes[0, col].set(title=job.rsplit('_', 1)[1], xlabel='Continuation update', ylabel='Original joint batch loss')
        axes[1, col].axhline(1, linestyle='--', color='#333333', linewidth=.7)
        axes[1, col].set(xticks=[0, 1, 2], xticklabels=['False safe', 'False reject', 'Joint cost'], ylabel='Cost / same_action')
        for row in (0, 1):
            axes[row, col].spines[['top', 'right']].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(.5, .93), frameon=False)
    fig.suptitle('Candidate identity and peer participation: fitting evidence only', y=.98, fontsize=11)
    out = PUBLIC/'figures'; out.mkdir(exist_ok=True)
    path = out/'factorial_fitting.svg'; fig.savefig(path, metadata={'Date':None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'factorial_fitting.png', dpi=150); plt.close(fig)


def render():
    analysis = source.read(PUBLIC/'analysis.json'); verify = source.read(PUBLIC/'verification.json')
    api.base.exact(verify['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    freeze = source.read(PUBLIC/'training_freeze.json')
    if (freeze['fresh_heads'] != 6 or freeze['cached_verified_heads'] != 6 or not freeze['matched_factorial_verified']
            or not source.read(PUBLIC/'full_refit.json')['exact'] or not verify['all12_fitting_metrics_and_attributions_exact']):
        raise ValueError('Complete factorial training, full refit and replay required')
    rows = [source.read(source.checked(r)) for r in freeze['fits']]; jobs = source.read(run.REG)['jobs']
    matched = comparisons(rows, jobs); screen = api.screen(rows, jobs)
    api.base.exact(screen, analysis['fitting_screen']); status = evidence_status(analysis, screen)
    phases = {k:source.read(PUBLIC/(k+'_execution.json')) for k in ('pilot', 'train', 'evaluate', 'verify')}
    summary = dict(status=analysis['status'], comparisons=matched, fitting_screen=screen, phases=phases, **status,
        fresh_models=6, cached_verified_models=6, parameters_per_model=76548, primary_fresh_updates=6000,
        full_refit_updates=1000, pilot_updates=500, metric_replay_exact=True, exact_resume=True,
        exact_legacy_engine_replay=True, floor_context_audit=source.read(PUBLIC/'floor_context_audit.json'),
        held_policy=analysis.get('held_policy'), held_contrasts=analysis.get('contrasts'),
        original_rules_unchanged=True, identity_certification=False, threshold_search=False,
        original24group_primary='not_estimable_not_passed', no_new_H=True, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'result_summary.json', source.stable(summary)); figure(rows)
    f = lambda x:'NA' if x is None else f'{x:.4f}'
    text = ['# Candidate Identity And Peer Participation', '',
        'I separated two choices in neural risk estimation: which forecast represents',
        'each peer, and whether peers that fall back to the baseline remain visible.',
        'Six new neural fits complete the two-by-two comparison; six prior fits are',
        'hash-verified controls, not new training. All have the same initial function,',
        '76,548 parameters, 1,000 updates, query draws, scalers and original loss.', '',
        '## Primary Contrast', '',
        'Proposed forecast versus same-index forecast, with all peers retained.',
        'Positive reductions are better; these are fitting costs, not held-out ADE.', '',
        '| Context | MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |',
        '|---|---:|---:|---:|---:|']
    for r in matched:
        if r['contrast'] == 'primary_identity_all_peers':
            text.append('| '+r['job'].rsplit('_', 1)[1]+' | '+' | '.join(f(r[k+'_reduction_percent']) for k in
                ('mean_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost'))+' |')
    passed = sum(r['passed'] for r in screen['contexts'])
    text += ['', f'The fixed internal screen passes **{passed}/3** contexts. This is not a safety gate.', '',
        '## All Factor Contrasts', '',
        '| Context | Contrast (treatment vs control) | MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % |',
        '|---|---|---:|---:|---:|']
    for r in matched:
        text.append('| '+r['job'].rsplit('_', 1)[1]+' | '+r['treatment']+' vs '+r['control']+' | '+
            ' | '.join(f(r[k+'_reduction_percent']) for k in ('mean_MSE', 'total_joint_cost', 'false_safe_cost'))+' |')
    text += ['', '![Fitting losses and costs](figures/factorial_fitting.svg)', '',
        '## Evidence Boundary', '',
        'The non-proposer representation is action0, verified row-by-row to equal',
        'the causal physical-floor path. It is not a future label or a permissible',
        'intervention. Pooling excludes ego and other queries. The proposal is frozen',
        'before the final veto, not the final admitted set.', '']
    if status['held_readout'] == 'not_run':
        text += ['**not_run:** '+analysis['reason']+'.',
            'No extra nested-C or reserved-H result is inferred. The previous17-view',
            'risk/support failure and original24-group NA remain unchanged.']
    else:
        text += ['The fixed pair was read on already exposed nested-C development.',
            'This is not independent confirmation; complete policies and contrasts are in analysis.json.',
            'Original risk/support pass: '+str(status['original_risk_pass'])+'.']
    text += ['', '## Execution', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for key, r in phases.items():
        text.append(f"| {key} | {r['pid']} | {r['seconds']:.2f} | {r['peak_RSS_bytes']} |")
    text += ['', '6,000 fresh primary +1,000 full-refit +500 pilot updates. The pilot also',
        'reproduces the saved legacy100-update model exactly. All12cells replay exactly.',
        'Native arm64 CPU4/inter-op1/workers0, checkpoint/resume/heartbeat. One producer',
        'seed and three dependent fitting contexts are not three independent seeds.', '',
        '- fresh_run: six new heads, causal-floor audit, recomputed diagnostics.',
        '- cached_verified: six existing controls, parents, source splits and scalers.',
        '- not_run: independent confirmation, new trajectory/JEPA training, deployment.',
        '- Original2% all/easy harm, signed easy-net and support rules unchanged.',
        '- Empty support remains NA; no identity certification or threshold search.',
        '- Moving20, DUT, DroneCrowd and H remain closed; original24-group primary NA.',
        '- Native obs8/pred12 silver image coordinates, not metric/seconds/true3D/foundation.',
        '- Stage5C/SMC disabled. Not yet submission-ready.', '',
        '[Protocol](protocol.md) | [Metrics](result_summary.json) | [Failure analysis](failure_analysis.md)',
        '| [Reproduction](operation_zh.md)', '']
    (PUBLIC/'README.md').write_text('\n'.join(text))
    (PUBLIC/'gates.md').write_text('\n'.join(['# Evidence Gates', '', '| Gate | Status |', '|---|---|',
        '| Six new fits / six cached controls / matched four cells | pass |',
        '| Causal floor / initial identity / exact resume and refit | pass |',
        '| All12 metrics and attributions replay | pass |', f'| Fixed primary fitting screen | {passed}/3 |',
        f"| Original held risk/support | {status['original_risk_pass']} |", '| Original24-group primary | NA, not pass |',
        '| Independent confirmation | not_run |', '| Deployment / submission readiness | false / false |',
        '| Stage5C / SMC | disabled / disabled |', '', 'Engineering passes do not establish scientific safety.', '']))
    print(source.json.dumps(dict(fresh=6, cached_verified=6, screen=passed, **status)))


def main():
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        render()


if __name__ == '__main__':
    main()
