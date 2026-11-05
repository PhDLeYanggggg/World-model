"""Report a completed scale-input ablation without changing the primary arm."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_scale_ablation as run
from src.world_model.m3w_frozen_readout_refit import query_weights

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def summary(result):
    contrasts = {a: {k: result['contrasts'][a+'_minus_control'][k]['mean']
        for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for a in api.ARMS}
    return dict(contrasts=contrasts, main_arm='continued_mse',
        expansion_supported=result['expansion_supported'],
        objective_expansion_screens=result['objective_expansion_screens'],
        completion_bounds_are_sampling_CI=False, independent_confirmation=False,
        full_policy_coordinate_invariance=False, deployment_changed=False, submission_ready=False,
        original24group_primary='not_estimable_not_passed')


def fitting_audit(fits):
    _, values = run.register(lambda **_: None)
    states = {(r['job'], r['arm']): api.base.read_checkpoint(source.checked(r['checkpoint'])) for r in fits}
    rows = []
    for job in values[3]:
        d, warm = run.parent.prior.student_data(values, job)
        c = d['causal']; ix = source.np.arange(len(c['ids'])); choice = c['allowed'].argmax(1)
        weights = query_weights(c, warm['sampler']['counts'])
        norm = api.parent.parent.parent.normalizer(d['preprocess'])
        y = (api.parent.parent.parent.signed(d['truth_by_arm']['bound_aware']).numpy()/norm)[ix, choice]
        observed = d['observed_future']; known = (weights > 0) & observed
        api.parent.assert_matched(*(states[job['key'], a] for a in api.ARMS))
        for arm in api.ARMS:
            control_ref = source.read(run.parent.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
            control = api.base.read_checkpoint(source.checked(control_ref['checkpoint']))
            api.assert_control(states[job['key'], arm], control, d)
            for label, state, predict in (('unmasked', control, api.parent.predict),
                    ('scale_ablated', states[job['key'], arm], api.predict)):
                score = predict(state, c)[ix, choice]/c['scales'][:, None]/norm
                square = (score-y)**2
                logistic = 4*abs(y)*source.np.logaddexp(0, -source.np.sign(y)*score)
                mixed = source.np.where(observed[:, None], logistic, square)
                false_safe = known[:, None] & (score <= 0) & (y > 0)
                false_reject = known[:, None] & (score > 0) & (y < 0)
                rows.append(dict(job=job['key'], objective=arm, feature_variant=label,
                    common_MSE=float((weights*square.mean(1)).sum()),
                    signed_cost_training_objective=float((weights*mixed.mean(1)).sum()),
                    unknown_MSE_contribution=float((weights*square.mean(1)*~observed).sum()),
                    false_safe_channels=false_safe.sum(0).tolist(),
                    false_reject_channels=false_reject.sum(0).tolist(),
                    cost_weighted_sign_regret=(weights[:, None]*abs(y)*(false_safe | false_reject)).sum(0).tolist(),
                    false_safe_excess_mass=(weights[:, None]*y*false_safe).sum(0).tolist(),
                    false_reject_margin_mass=(weights[:, None]*-y*false_reject).sum(0).tolist(),
                    fitting_not_generalization=True))
        del d, warm, control; source.gc.collect()
    source.once(PUBLIC/'full_fitting_objectives.json', source.stable(dict(rows=rows,
        source='fresh_run', weights='empirical_query_draws', repeated_channels_not_independent=True)))
    return rows


def main():
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    result = source.read(PUBLIC/'analysis.json'); verification = source.read(PUBLIC/'verification.json')
    api.base.exact(verification['analysis_sha256'], source.sha(PUBLIC/'analysis.json'))
    if not verification['all3_predictions_actions_metrics_exact']:
        raise ValueError('Completed inference, action and metric replay required')
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 6 or not source.read(PUBLIC/'full_refit.json')['exact']:
        raise ValueError('Six completed real fits and exact treatment refit required')
    fits = [source.read(source.checked(ref)) for ref in freeze['fits']]
    for r in fits:
        state = api.base.read_checkpoint(source.checked(r['checkpoint']))
        if state['step'] != 2000 or state['parameters'] != 32130:
            raise ValueError('Complete actual model required')
        if any(int(v['step']) != 2000 for v in state['optimizer']['state'].values()):
            raise ValueError('Full actual optimizer budget required')
        if api.torch.equal(state['model']['encoder.weight'], state['initial_model']['encoder.weight']):
            raise ValueError('Neural encoder must really change')
    fit = fitting_audit(fits); s = summary(result)
    phases = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    source.once(PUBLIC/'result_summary.json', source.stable(dict(s, phases=phases,
        status='fresh_report_from_verified_results', analysis=source.artifact(PUBLIC/'analysis.json'),
        actual_optimizer_and_encoder_changes_verified=True, primary_updates=12000, full_refit_updates=2000,
        real_resume_updates_including_repeats=400, full_historical_test_suite_rerun=False)))
    n = lambda v: 'NA' if v is None else f'{v:.4f}'
    lines = ['# Explicit Scale Removal: Completed Neural Ablation', '',
        'I remove the risk head\'s explicit absolute-scale input while retaining',
        'normalized geometry and the original losses, source roles and risk rules.',
        'Six fresh continuations are compared with six cached_verified unmasked',
        'controls. Main contrast is MSE; signed-cost learning remains secondary.', '',
        '## Held Development Results', '',
        '| Model | Known ADE gain over floor % | Hard gain % | Intervention % | Full risk/support |',
        '|---|---:|---:|---:|---:|']
    names = ['readout_veto', 'kinematic_veto']+[prefix+a+'_veto' for a in api.ARMS for prefix in ('', 'ablated_')]
    for name in names:
        p = result['held_policy'][name]; risk = p['safety']
        lines.append(f"| {name} | {n(p['known']['mean'])} | {n(p['hard']['mean'])} | {n(p['switch_percent']['mean'])} | {risk['supported']}/{risk['required']} |")
    lines += ['', '| Objective | Ablated minus unmasked known pp | Completion lower pp | Completion upper pp | Expansion screen |',
        '|---|---:|---:|---:|---|']
    for a in api.ARMS:
        c = s['contrasts'][a]
        lines.append(f"| {a} | {n(c['known_pp'])} | {n(c['completion_lower_pp'])} | {n(c['completion_upper_pp'])} | {s['objective_expansion_screens'][a]} |")
    lines += ['', 'Missing-label completion bounds are NOT sampling confidence intervals.',
        'An expansion screen is not a safety certificate.', '', '## Original Safety Accounting', '',
        '| Variant | Known all violations | Known easy violations | Undefined selected easy | Worst defined easy-net % |',
        '|---|---:|---:|---:|---:|']
    for a in api.ARMS:
        for prefix in ('', 'ablated_'):
            name = prefix+a+'_veto'; risk = result['held_policy'][name]['safety']
            lines.append(f"| {name} | {risk['known_all_violations']} | {risk['known_easy_violations']} | {risk['undefined_selected_easy']} | {n(risk['worst_defined_signed_easy_net_percent'])} |")
    lines += ['', 'Original24group primary remainsNA, notpassed. Empty utility controls and',
        'missing supports stayNA. No identity certificate, new test, deployment',
        'or submission-readiness upgrade. Prior failures remain unchanged.', '',
        '## Runtime', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for phase, p in phases.items():
        lines.append(f"| {phase} | {p['pid']} | {p['seconds']:.2f} | {p['peak_RSS_bytes']} |")
    lines += ['', 'Nativearm64Torch CPU4/interop1/workers0. Six fits12,000primaryupdates,',
        '2,000fullrefitupdates,400resume-checkupdates. All six freeze before V.',
        'Both objective arms resume exactly. First masked logistic full refit',
        'and all new predictions/actions/metrics replay exactly. Old actions are',
        'hash-verified; old metrics recomputed fresh and match exactly.', '',
        '## Limits', '',
        'Seven exposed SOURCE units,17recording views,one producerseed. Repeated',
        'development evidence, not independent confirmation. Overlapping windows',
        'are not additional independent scenes. B has only2/3/3sourcegroups.',
        'The frozen proposer and support models retain original scale information;',
        'warm weights previously saw scale. This is not full-policy invariance.',
        'No future inference inputs, new trajectory dynamics, JEPA or reservedH.',
        'Nativeobs8/pred12 imagepixels: no verifiedseconds/metric/true3D/foundation',
        'or human-gold claim. Stage5C/SMCoff. Original2%and support rules unchanged.', '',
        '[Protocol](protocol.md) | [Fitting audit](full_fitting_objectives.json) | [Replay](verification.json)', '']
    (PUBLIC/'README.md').write_text('\n'.join(lines))
    lines = ['# Complete Fitting Diagnostics', '',
        'Empirical whole-query draw weights. Scores are not calibrated bounds.',
        'Logistic-score MSE is a cross-objective diagnostic, not a risk calibration metric.', '',
        '| Context | Objective | Features | MSE | Signed-cost objective | All/easy sign regret | False-safe all/easy |',
        '|---|---|---|---:|---:|---|---|']
    for r in fit:
        lines.append(f"| {r['job']} | {r['objective']} | {r['feature_variant']} | {r['common_MSE']:.8f} | {r['signed_cost_training_objective']:.8f} | {r['cost_weighted_sign_regret']} | {r['false_safe_channels']} |")
    lines += ['', '## Prospective Context Screens', '', '| Objective | Context | Completion lower pp | Expansion |', '|---|---|---:|---|']
    for c in result['per_context']:
        lines.append(f"| {c['arm']} | {c['job']} | {n(c['completion_lower_pp'])} | {c['expansion_condition']} |")
    (PUBLIC/'fitting_analysis.md').write_text('\n'.join(lines)+'\n')
    gate = ['# Original-Rule Gates', '', '- Real training/resume/refit/replay: pass.',
        '- Own-fitting preprocessing and held-source exclusion checks: pass.',
        f"- Main expansion screen: {s['expansion_supported']}."]
    for a in api.ARMS:
        risk = result['held_policy']['ablated_'+a+'_veto']['safety']
        gate.append(f"- {a} full risk/support: {risk['supported']}/{risk['required']} recording views.")
    gate += ['- Original24group primary: NA, notpassed.', '- Reserved-domain independent confirmation: not_run, reserved.',
        '- Deployment changed: false. Submission ready: false.', '- Stage5C execution: false. SMC: false.', '']
    (PUBLIC/'gates.md').write_text('\n'.join(gate))
    commands = '\n'.join(f'.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py {p}'
        for p in ('register', 'pilot', 'train', 'evaluate', 'verify'))
    (PUBLIC/'reproducibility.md').write_text('# Reproducibility\n\n```sh\n'+commands+'\n```\n\n'
        'Resume only unfinished phases with --resume and unchanged registration_v2.json.\n'
        'Completed phases refuse overwrite. Checkpoints/caches stay local and must match hashes.\n'
        'report_m3w_moving_scale_ablation.py reproduces fitting diagnostics and reports.\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
