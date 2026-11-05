"""Reproducible evidence report; scores are never promoted to risk guarantees."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_signed_decision as run
from src.world_model.m3w_frozen_readout_refit import query_weights

source, api, PUBLIC = run.source, run.api, run.PUBLIC


def summary(result):
    contrast = result['contrasts']['logistic_minus_mse']
    return dict(known_contrast_pp=contrast['known_pp']['mean'],
        completion_lower_pp=contrast['completion_lower_pp']['mean'],
        completion_upper_pp=contrast['completion_upper_pp']['mean'],
        expansion_supported=result['expansion_supported'],
        completion_bounds_are_sampling_CI=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False,
        original24group_primary='not_estimable_not_passed')


def fitting_audit(fit_refs):
    _, values = run.register(lambda **_: None)
    states = {(r['job'], r['arm']): api.base.read_checkpoint(source.checked(r['checkpoint'])) for r in fit_refs}
    rows = []
    for job in values[3]:
        d, warm = run.prior.student_data(values, job)
        c = d['causal']; ix = source.np.arange(len(c['ids'])); choice = c['allowed'].argmax(1)
        weights = query_weights(c, warm['sampler']['counts'])
        norm = api.parent.parent.normalizer(d['preprocess'])
        y = (api.parent.parent.signed(d['truth_by_arm']['bound_aware']).numpy()/norm)[ix, choice]
        observed = d['observed_future']; known = (weights > 0) & observed
        api.assert_matched(*(states[job['key'], a] for a in api.ARMS))
        for name, state in [('warm', warm)]+[(a, states[job['key'], a]) for a in api.ARMS]:
            score = api.predict(state, c)[ix, choice]/c['scales'][:, None]/norm
            square = (score-y)**2
            logistic = 4*abs(y)*source.np.logaddexp(0, -source.np.sign(y)*score)
            mixed = source.np.where(observed[:, None], logistic, square)
            false_safe = known[:, None] & (score <= 0) & (y > 0)
            false_reject = known[:, None] & (score > 0) & (y < 0)
            rows.append(dict(job=job['key'], arm=name,
                common_MSE=float((weights*square.mean(1)).sum()),
                signed_cost_training_objective=float((weights*mixed.mean(1)).sum()),
                unknown_MSE_contribution=float((weights*square.mean(1)*~observed).sum()),
                false_safe_channels=false_safe.sum(0).tolist(),
                false_reject_channels=false_reject.sum(0).tolist(),
                cost_weighted_sign_regret=(weights[:, None]*abs(y)*(false_safe | false_reject)).sum(0).tolist(),
                false_safe_excess_mass=(weights[:, None]*y*false_safe).sum(0).tolist(),
                false_reject_margin_mass=(weights[:, None]*-y*false_reject).sum(0).tolist(),
                fitting_not_generalization=True))
        del d, warm; source.gc.collect()
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
            raise ValueError('Every parameter must receive full optimizer budget')
        if api.torch.equal(state['model']['encoder.weight'], state['initial_model']['encoder.weight']):
            raise ValueError('Common encoder must really change')
        if not all(api.torch.isfinite(v).all() for v in state['model'].values()):
            raise ValueError('Finite trained weights required')
    fit = fitting_audit(fits); s = summary(result)
    phases = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    source.once(PUBLIC/'result_summary.json', source.stable(dict(s, phases=phases,
        status='fresh_report_from_verified_results', analysis=source.artifact(PUBLIC/'analysis.json'),
        actual_optimizer_and_encoder_changes_verified=True, primary_updates=12000, full_refit_updates=2000,
        real_resume_updates_including_repeats=400, full_historical_test_suite_rerun=False)))
    n = lambda v: 'NA' if v is None else f'{v:.4f}'
    lines = ['# Signed-Cost Neural Admission: Completed Comparison', '',
        'I changed the training objective, not the evaluation contract. Six real',
        'neural continuations compare magnitude-weighted logistic sign learning',
        'with MSE from identical warm weights. Unknown outcomes retain their',
        'original bound-aware MSE. Neither output is a calibrated safety bound.', '',
        '## Held Development Results', '',
        '| Model | Known ADE gain over floor % | Hard gain % | Intervention % | Full risk/support |',
        '|---|---:|---:|---:|---:|']
    for name in ('readout_veto', 'kinematic_veto', 'continued_mse_veto', 'in_source_errors_veto',
            'held_source_errors_veto', 'signed_cost_logistic_veto'):
        p = result['held_policy'][name]; risk = p['safety']
        lines.append(f"| {name} | {n(p['known']['mean'])} | {n(p['hard']['mean'])} | {n(p['switch_percent']['mean'])} | {risk['supported']}/{risk['required']} |")
    lines += ['', f"Signed-cost minus MSE known contrast: **{n(s['known_contrast_pp'])} pp**.",
        f"Missing-label completion bounds: [{n(s['completion_lower_pp'])}, {n(s['completion_upper_pp'])}] pp.",
        'These are not sampling confidence intervals.',
        f"Prospective expansion screen: **{s['expansion_supported']}**. It is not a safety certificate.", '',
        '## Original Safety Accounting', '',
        '| Arm | Known all violations | Known easy violations | Undefined selected easy | Worst defined easy-net % |',
        '|---|---:|---:|---:|---:|']
    for arm in api.ARMS:
        risk = result['held_policy'][arm+'_veto']['safety']
        lines.append(f"| {arm} | {risk['known_all_violations']} | {risk['known_easy_violations']} | {risk['undefined_selected_easy']} | {n(risk['worst_defined_signed_easy_net_percent'])} |")
    lines += ['', 'Full original24group primary remains NA, not passed. Missing supports',
        'and empty same-count controls do not become safety passes. No identity',
        'certificate or evaluation revision. Prior failures remain unchanged.', '',
        '## Runtime And Reproducibility', '',
        '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for phase, p in phases.items():
        lines.append(f"| {phase} | {p['pid']} | {p['seconds']:.2f} | {p['peak_RSS_bytes']} |")
    lines += ['', 'Nativearm64 Torch CPU4/interop1/workers0, no resource probing. Six fits',
        'complete12,000primary updates; exact treatment refit adds2,000. Both',
        'arms resume exactly and all three fresh MSE controls reproduce old weights.',
        'All models freeze before V. New predictions/actions/metrics replay exactly.',
        'Old control actions are cached_verified; their metrics are recomputed',
        'fresh and match. New loss fits and evaluation are fresh_run.', '',
        '## Limits', '',
        'Seven exposed SOURCE units,17recording views, one producer seed. Many',
        'overlapping windows do not add independent scene evidence. This is a',
        'repeated development comparison, not independent confirmation. B has',
        'only2/3/3source groups. Logistic logits are not expected-risk estimates;',
        'unknown-bound MSE makes the complete objective a mixed surrogate.',
        'No new trajectory dynamics, JEPA, held H or reserved external readout.',
        'Nativeobs8/pred12 imagepixels, not verifiedseconds/metric/true3D/foundation.',
        'No deployment change or submission-readiness upgrade. Stage5C/SMCoff.', '',
        '[Protocol](protocol.md) | [Fitting audit](full_fitting_objectives.json) | [Exact replay](verification.json)', '']
    (PUBLIC/'README.md').write_text('\n'.join(lines))
    lines = ['# Fitting Objectives And Decision Errors', '',
        'Complete empirical query weights, not one minibatch. MSE columns applied',
        'to logistic scores are only cross-objective diagnostics, not calibrated',
        'risk errors. Count/regret units are repeated fitting channels.', '',
        '| Context | Model | MSE | Signed-cost mixed objective | All/easy sign regret | False-safe all/easy |',
        '|---|---|---:|---:|---|---|']
    for r in fit:
        lines.append(f"| {r['job']} | {r['arm']} | {r['common_MSE']:.8f} | {r['signed_cost_training_objective']:.8f} | {r['cost_weighted_sign_regret']} | {r['false_safe_channels']} |")
    lines += ['', '## Prospective Context Results', '',
        '| Context | Paired completion lower pp | Expansion condition |', '|---|---:|---|']
    for c in result['per_context']:
        lines.append(f"| {c['job']} | {n(c['completion_lower_pp'])} | {c['expansion_condition']} |")
    lines += ['', 'No favorable-source selection or changed threshold follows this readout.',
        'The ideal fully observed conditional optimum derivation is an objective',
        'motivation, not a finite-network generalization or safety theorem.', '']
    (PUBLIC/'fitting_analysis.md').write_text('\n'.join(lines))
    safety = result['held_policy']['signed_cost_logistic_veto']['safety']
    (PUBLIC/'gates.md').write_text('# Original-Rule Gates\n\n'
        '- Real neural fitting, exact resume/refit and inference replay: pass.\n'
        '- Hash/source exclusion and causal inference schema checks: pass.\n'
        f"- Prospective expansion screen: {s['expansion_supported']}.\n"
        f"- Full original risk/support: {safety['supported']}/{safety['required']} views.\n"
        '- Original24group primary: NA, not passed.\n'
        '- Independent confirmation and reserved-domain evaluation: not_run, reserved.\n'
        '- Deployment changed: false. Submission ready: false.\n'
        '- Stage5C execution: false. SMC: false.\n')
    commands = '\n'.join(f'.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py {p}'
        for p in ('register', 'pilot', 'train', 'evaluate', 'verify'))
    (PUBLIC/'reproducibility.md').write_text('# Reproducibility\n\n```sh\n'+commands+'\n```\n\n'
        'Use --resume only for unfinished phases with unchanged registration.\n'
        'Completed phases refuse overwrite. Checkpoints and prediction caches stay local.\n'
        'No reserved data are needed. `report_m3w_moving_signed_decision.py` reproduces reports.\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
