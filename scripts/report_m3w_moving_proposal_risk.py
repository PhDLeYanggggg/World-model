"""Report the completed proposal-risk study without changing frozen decisions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_proposal_risk as run
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval

source, PUBLIC = run.source, run.PUBLIC


def audit_fits():
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 108 or freeze['primary_updates'] != 216000 or not freeze['all_before_validation']:
        raise ValueError('Complete registered training before reporting')
    reg = source.read(run.REG)
    rows = []
    for ref in freeze['fits']:
        row = source.read(source.checked(ref))
        state = run.api.base.read_checkpoint(source.checked(row['checkpoint']))
        control = run.api.base.read_checkpoint(source.checked(row['cached_control']))
        if (state['identity']['registration_sha256'] != source.sha(run.REG)
                or state['step'] != 2000 or state['budget'] != 2000 or state['parameters'] != 28229):
            raise ValueError('Actual full-budget registered checkpoint required')
        for key in ('initial_model', 'preprocess', 'sampler', 'draw_hash', 'fit_ids', 'source_sites'):
            run.api.base.exact(state[key], control[key])
        names = reg['partitions'][row['job']][row['critic']]['names']
        if set(state['source_sites']) != set(names) or set(state['preprocess']['source_names']) != set(names):
            raise ValueError('Own-part preprocessing only')
        router = run.api.base.read_checkpoint(source.checked(state['identity']['router_checkpoint']))
        if set(router['source_sites']) & set(names) or set(router['fit_ids']) & set(state['fit_ids']):
            raise ValueError('Training proposer must exclude critic fitting data')
        if any(int(x['step']) != 2000 for x in state['optimizer']['state'].values()):
            raise ValueError('Optimizer must confirm the full training budget')
        if not all(run.api.router.torch.isfinite(x).all() for x in state['model'].values()):
            raise ValueError('Finite learned model required')
        if not any(not run.api.router.torch.equal(x, state['initial_model'][k]) for k, x in state['model'].items()):
            raise ValueError('Initialization is not a learned model')
        rows.append(dict(job=row['job'], proposer=row['proposer'], critic=row['critic'],
            source_excluded_proposer=True, own_control_init_preprocess_sampler_exact=True,
            optimizer_updates=2000, checkpoint=row['checkpoint']))
    first = source.read(source.checked(freeze['fits'][0]))
    repeat = source.read(PUBLIC/'first_full_refit.json')
    run.api.router.exact_state(run.api.base.read_checkpoint(source.checked(first['checkpoint'])),
        run.api.base.read_checkpoint(source.checked(repeat['checkpoint'])))
    result = dict(complete108=True, primary_updates=216000, first_full_refit_exact=True, rows=rows,
        training_freeze=source.artifact(PUBLIC/'training_freeze.json'))
    source.once(PUBLIC/'fit_audit.json', result)
    return [source.read(source.checked(ref)) for ref in freeze['fits']]


def aggregate(groups):
    rows = [r for group in groups for r in group['rows']]
    sites = sorted({r['site'] for r in rows}); cfg = source.read(source.CONFIG)
    policies, risk, reliability = {}, {}, {}
    for name in run.POLICIES:
        fields = ('known_ADE_gain_floor_percent', 'completion_lower_gain_floor_percent',
            'known_ADE_pixel', 'available_endpoint_FDE_diagnostic_pixel', 'p95_known_ADE_pixel')
        policies[name] = {k: source.group_interval(rows, lambda r, k=k, a=name: r['policies'][a][k], sites, cfg) for k in fields}
        for subset in ('easy', 'hard'):
            policies[name][subset+'_gain_floor_percent'] = source.group_interval(rows,
                lambda r, a=name, s=subset: r['policies'][a]['subsets'][s]['gain_percent'], sites, cfg)
        views = [v for r in rows for v in r['policies'][name]['recordings'].values()]
        defined = [100*v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        risk[name] = dict(required_record_views=len(views),
            worst_defined_signed_easy_net_percent=max(defined) if defined else None,
            undefined_full_known_easy_net_views=sum(v['known_easy_net_degradation'] is None for v in views),
            no_intervention_views=sum(v['selected'] == 0 for v in views))
    for arm in ('all_action', 'conditioned'):
        reliability[arm] = {}
        for subset in ('same_proposals', 'admitted'):
            values = [r['cost_reliability'][arm][subset] for r in rows]
            p = source.np.sum([v['predicted_on_selected_known_moments'] for v in values], axis=0)
            y = source.np.sum([v['realized_selected_known_moments'] for v in values], axis=0)
            reliability[arm][subset] = dict(selected_known=sum(v['selected_known'] for v in values),
                selected_unknown=sum(v['selected_unknown'] for v in values),
                predicted_known_moment_mass=p.tolist(), realized_known_moment_mass=y.tolist(),
                harm_actual_over_predicted=float(y[1]/p[1]) if p[1] > 0 else None,
                easy_harm_actual_over_predicted=float(y[4]/p[4]) if p[4] > 0 else None,
                diagnostic_only_repeated_views_not_independent_rows=True)
    source.once(PUBLIC/'policy_metrics.json', source.stable(policies))
    source.once(PUBLIC/'risk_diagnostics.json', source.stable(risk))
    source.once(PUBLIC/'cost_reliability.json', source.stable(reliability))
    return policies, risk, reliability


def audit_controls(groups):
    count = 0
    for group in groups:
        first = group['rows'][0]
        key = first['job']+f"_val{first['validation_part']}_propose{first['proposer_part']}.json"
        previous = source.read(run.parent.PUBLIC/'groups'/key)
        parent_rows = {r['site']: r for r in previous['rows']}
        for row in group['rows']:
            for name, old_name in (('physical_floor', 'physical_floor'), ('public_router', 'public_router'),
                    ('all_action_veto', 'public_veto'), ('all_action_matched', 'public_matched')):
                if row['policies'][name] != parent_rows[row['site']]['policies'][old_name]:
                    raise ValueError('Cached control readout changed')
                count += 1
    if len(groups) != 108 or count != 1104:
        raise ValueError('Every cached control must replay')
    source.once(PUBLIC/'parent_control_replay.json', dict(all108_blocks_exact=True,
        source_policy_readouts_exact=count, parent_analysis=source.artifact(run.parent.PUBLIC/'analysis.json')))


def verdict(analysis, decision):
    primary = analysis['nonempty_matched']['conditioned']['completion_lower_pp']
    positive = primary['CI95'] is not None and primary['CI95'][0] > 0
    safe = analysis['safety']['conditioned_veto']['full_original_risk_pass']
    if decision['development_criteria_pass'] != bool(positive and safe):
        raise ValueError('Reporting may not override the frozen scientific gate')
    return ('Primary nonempty matched utility: not estimable, not passed.' if primary['CI95'] is None
        else 'Primary nonempty matched utility: '+('positive' if positive else 'not positive')+' lower CI.') + (
        ' Original risk/support contract passes.' if safe else ' Original risk/support contract fails.')


def figures(fits, analysis, reliability):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'moving_proposal_risk_v1'
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    steps = [x['step'] for x in fits[0]['trace']]
    losses = source.np.asarray([[x['total'] for x in fit['trace']] for fit in fits])
    ax[0].plot(steps, source.np.median(losses, axis=0), color='#167D8D')
    ax[0].fill_between(steps, *source.np.quantile(losses, [.25, .75], axis=0), color='#167D8D', alpha=.2)
    ax[0].set(title='Training objective: median and IQR', xlabel='Optimizer update', ylabel='Cost loss (not safety)')
    names = ['all_action_veto', 'conditioned_veto']
    x = source.np.arange(2)
    for i, (key, label, color) in enumerate((('supported', 'Full risk/support pass', '#167D8D'),
            ('known_easy_violations', 'Known easy-risk violations', '#B54848'),
            ('undefined_selected_easy', 'Undefined selected easy risk', '#8B7A3C'))):
        ax[1].bar(x+(i-1)*.23, [analysis['safety'][a][key] for a in names], width=.23, label=label, color=color)
    ax[1].set(xticks=x, xticklabels=['All-action critic', 'Proposal critic'],
        ylabel='Required recording views (of 696)', title='Original rules, no certification')
    ax[1].legend(fontsize=7)
    for i, arm in enumerate(('all_action', 'conditioned')):
        vals = [reliability[arm][s]['easy_harm_actual_over_predicted'] for s in ('same_proposals', 'admitted')]
        ax[2].plot([0, 1], vals, marker='o', label=arm.replace('_', ' '))
    ax[2].axhline(1, color='#555555', linestyle='--', linewidth=.8)
    ax[2].set(xticks=[0, 1], xticklabels=['Same proposal pool', 'Own admitted actions'], yscale='log',
        ylabel='Realized / predicted easy-harm mass', title='Selected-cost diagnostic, not a gate')
    ax[2].legend(fontsize=8)
    folder = PUBLIC/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'training_and_risk.svg', metadata={'Date': None})
    png = run.PRIVATE/'training_and_risk.png'; fig.savefig(png, dpi=150); plt.close(fig)
    return png


def main():
    verification = source.read(PUBLIC/'verification.json')
    if verification['analysis_sha256'] != source.sha(PUBLIC/'analysis.json'):
        raise ValueError('Replay exact completed analysis before reporting')
    fits = audit_fits()
    analysis, decision = source.read(PUBLIC/'analysis.json'), source.read(PUBLIC/'decision.json')
    groups = [source.read(source.checked(ref)) for ref in analysis['group_receipts']]
    audit_controls(groups)
    metrics, risk, reliability = aggregate(groups)
    runtime = {phase: source.read(PUBLIC/(phase+'_execution.json')) for phase in ('pilot', 'train', 'evaluate', 'verify')}
    source.once(PUBLIC/'runtime_summary.json', runtime)
    lines = ['# Proposal-Conditioned Risk: Completed Development Experiment', '', verdict(analysis, decision), '',
        '## Material Passport', '',
        'fresh_run:108 new real Torch cost critics,216000 primary updates, evaluation and replay.',
        'cached_verified: frozen forecasts, public routers and all-action point critics. No retraining claim for controls.',
        'not_run: new H or independent final test, new trajectory dynamics, deployment upgrade, Stage5C and SMC.',
        'MovingDroneCrowd++ exposed SOURCE, native8/12 annotation order, image pixels. Not SDDt50,',
        'verified seconds, metric geometry, true3D or a foundation result.', '',
        '## Main Contrasts', '', '|Contrast|Known effect pp [95% CI]|Direct completion lower pp [95% CI]|',
        '|---|---:|---:|']
    for label, values in [('Conditioned veto minus all-action veto', analysis['conditioned_minus_all_action']),
            ('Conditioned full strategy minus own matched control (secondary)', analysis['secondary_full_strategy']['conditioned'])]:
        lines.append('|'+label+'|'+interval(values['known_pp'])+'|'+interval(values['completion_lower_pp'])+'|')
    lines += ['', 'Original required nonempty matched comparison: '+str(analysis['unavailable_primary_views']['conditioned'])+
        '/276 source views are unavailable. No unavailable view is replaced by zero or dropped.',
        'Original24group primary remains not_estimable_not_passed. All23 source units are retained.',
        'Intervals use3000source-cluster resamples after seed/context/orientation averaging;',
        'these repeated development views are not verified independent test sites.', '',
        '## Policies', '', '|Policy|Known ADE gain vs floor %|Hard gain %|Switch %|Risk/support pass|Easy-risk violations|Undefined selected easy|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in run.POLICIES:
        s = analysis['safety'][name]
        lines.append('|'+name+'|'+interval(metrics[name]['known_ADE_gain_floor_percent'])+'|'+interval(metrics[name]['hard_gain_floor_percent'])+
            '|'+interval(analysis['switch_percent'][name])+f"|{s['supported']}/696|{s['known_easy_violations']}|{s['undefined_selected_easy']}|")
    lines += ['', 'Source-macro gains are means of within-source ratios, not ratios of separately averaged pixel errors.',
        'Available-label ADE is not full12 ADE; available endpoint FDE is diagnostic. See policy_metrics.json.',
        'Worst defined signed recording easy-net degradation is '+str(risk['conditioned_veto']['worst_defined_signed_easy_net_percent'])+
        '%, not a dataset mean; undefined easy-net views remain undefined.', '', '## Selected-Cost Reliability', '',
        '|Critic|Selection|Actual/predicted harm mass|Actual/predicted easy-harm mass|', '|---|---|---:|---:|']
    for arm, subsets in reliability.items():
        for subset, result in subsets.items():
            lines.append(f"|{arm}|{subset}|{result['harm_actual_over_predicted']}|{result['easy_harm_actual_over_predicted']}|")
    lines += ['', 'Ratios pool repeated view masses for diagnosis only, not an independent calibration guarantee.',
        'Same-proposal comparisons hold actions fixed. Admitted comparisons change the selected population.', '',
        '## Engineering Evidence', '',
        'All108 checkpoints match control initialization, own-source preprocessing and query draw hashes.',
        'The real600-update pilot resumes exactly in all3cells; first full refit is exact.',
        'All108 policy/prediction blocks replay and1104cached source-policy readouts match.',
        'These checks establish implementation reproducibility, not independent scientific confirmation.', '',
        '|Phase|PID|Measured seconds|Peak RSS bytes|', '|---|---:|---:|---:|']
    for phase, v in runtime.items():
        lines.append(f"|{phase}|{v['pid']}|{v['seconds']:.3f}|{v['peak_RSS_bytes']}|")
    lines += ['', 'Native arm64 CPU4/interop1/workers0. Full checkpoints stay local, not in Git.',
        'The change is the supervised-action support, including its effective label exposure.',
        'Architecture, future-free features, forecast bank, thresholds and original risk/support rules are unchanged.',
        'No new publication-ready method, safe deployment, Stage5C or SMC is declared.', '',
        '![Training and original risk checks](figures/training_and_risk.svg)', '']
    text_once(PUBLIC/'README.md', '\n'.join(lines))
    text_once(PUBLIC/'reproducibility.md', '\n'.join(['# Reproduction', '',
        'Run from the World checkout using native arm64 `.venv-pytorch/bin/python`.',
        'Fresh execution phases: register, pilot, train, evaluate, verify. Each completed phase is immutable.',
        'Resume an interrupted fitting phase with `--resume`; do not restart completed training.',
        'Checkpoint every100updates; heartbeat file under the local private study directory.',
        'The registration binds code/config/protocol. The training freeze binds every model receipt.',
        'Reports require a verified analysis hash. The report command does not fit or choose a policy.',
        'Run `scripts/report_m3w_moving_proposal_risk.py` after successful verification.',
        'Public aggregates and code are sufficient to inspect claims, not to train without separately licensed data.',
        'Reserved H/DUT/DroneCrowd remain closed. Original failures and undefined controls remain.', '']))
    print(figures(fits, analysis, reliability))


if __name__ == '__main__':
    main()
