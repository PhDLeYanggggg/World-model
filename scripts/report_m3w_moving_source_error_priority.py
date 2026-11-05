"""Evidence-bound report for source-held error discovery; never a safety upgrade."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_source_error_priority as run

source, PUBLIC = run.source, run.PUBLIC


def held_action_audit(values):
    from scripts.run_m3w_moving_readout_refit import flow
    _, _, data, jobs, refs, parts, *_ = values
    rows = []
    for job in jobs:
        p = run.old.packet(data, job, refs, parts[job['key']][2], training=False)
        group = source.read(PUBLIC/'groups'/(job['key']+'_val2_propose1.json'))
        with source.np.load(source.checked(group['cache']), allow_pickle=False) as saved:
            source.np.testing.assert_array_equal(saved['ids'], p['ids'])
            proposal = saved['public_router']; before = saved['in_source_errors_veto']; after = saved['held_source_errors_veto']
            ids = p['ids']; target = data['target_eval'][ids]-data['origin'][ids, None]
            truth = run.old.amplitude.targets(p, target, data['valid'][ids], job['easy_cut'])
            for site in sorted(set(p['sites'])):
                at = p['sites'] == site
                local = dict(moments=truth['moments'][at], observed_future=truth['observed_future'][at])
                rows.append(dict(job=job['key'], source=str(site), action_flow=flow(local,
                    proposal[at], before[at], after[at]), posthoc_diagnostic_only=True,
                    frozen_actions_not_reselected=True))
    source.once(PUBLIC/'held_action_flow.json', source.stable(rows))


def fitting_audit(student_rows, discovery_rows):
    from src.world_model.m3w_frozen_readout_refit import query_weights
    api = run.api
    _, values = run.register(lambda **_: None)
    states = {(r['job'], r['arm']): api.base.read_checkpoint(source.checked(r['checkpoint'])) for r in student_rows}
    losses, overlaps, extrapolation = [], [], []
    for job in values[3]:
        d, warm = run.student_data(values, job)
        c = d['causal']; ix = source.np.arange(len(c['ids'])); choice = c['allowed'].argmax(1)
        weights = query_weights(c, warm['sampler']['counts'])
        norm = api.kin.parent.normalizer(d['preprocess'])
        truth = (api.kin.parent.signed(d['truth_by_arm']['bound_aware']).numpy()/norm)[ix, choice]
        dr = next(r for r in discovery_rows if r['job'] == job['key'])
        basis = api.kin.motion_basis(c['x'])
        for ref in dr['teachers']:
            receipt = source.read(source.checked(ref))
            teacher = api.base.read_checkpoint(source.checked(receipt['checkpoint']))
            common = (c['x'][..., :380]-teacher['preprocess']['mean'][:380])/teacher['preprocess']['std'][:380]
            extra = (basis-teacher['extra_preprocess']['mean'])/teacher['extra_preprocess']['std']
            for held, role in ((True, 'held_B_source'), (False, 'teacher_fitting_sources')):
                at = (c['sites'] == receipt['omitted_source']) == held
                extrapolation.append(dict(job=job['key'], omitted_source=receipt['omitted_source'], role=role,
                    rows=int(at.sum()), common_feature_values_clipped_fraction=float((abs(common[at]) > 8).mean()),
                    kinematic_feature_values_clipped_fraction=float((abs(extra[at]) > 8).mean()),
                    common_rows_with_any_clipping_fraction=float((abs(common[at]) > 8).any((1, 2)).mean()),
                    diagnostic_only=True, future_labels_used=False))
            del teacher, common, extra
        masks = {}
        with source.np.load(source.checked(dr['cache']), allow_pickle=False) as saved:
            source.np.testing.assert_array_equal(saved['ids'], c['ids'])
            for arm in api.ARMS:
                prepared = api.prioritize(d, saved[arm]); masks[arm] = prepared['priority_mask'].numpy()[ix, choice]
                state = states[job['key'], arm]
                api.base.exact(state['initial_model'], warm['model'])
                api.base.exact(state['priority_mask_hash'], prepared['priority_mask_hash'])
                api.base.exact(state['warm_prediction_hash'], prepared['warm_prediction_hash'])
        api.assert_student_pair(*(states[job['key'], arm] for arm in api.ARMS))
        left, right = (masks[a] for a in api.ARMS)
        overlaps.append(dict(job=job['key'], intersection=(left & right).sum(0).tolist(),
            in_source_only=(left & ~right).sum(0).tolist(), held_source_only=(right & ~left).sum(0).tolist(),
            label_channels_not_independent_samples=True))
        for name, state in [('warm', warm)]+[(arm, states[job['key'], arm]) for arm in api.ARMS]:
            prediction = api.predict(state, c)[ix, choice]/c['scales'][:, None]/norm
            square = (prediction-truth)**2
            priority = {a: float((weights*(square*m).mean(1)).sum()) for a, m in masks.items()}
            known = (weights > 0) & d['observed_future']
            false_safe = known[:, None] & (prediction <= 0) & (truth > 0)
            losses.append(dict(job=job['key'], arm=name,
                full_anchor_MSE=float((weights*square.mean(1)).sum()), full_priority_MSE=priority,
                false_safe_output_channels=false_safe.sum(0).tolist(),
                observed_proposed_rows=int(known.sum()), bound_proposed_rows=int(((weights > 0) & ~d['observed_future']).sum()),
                resubstitution_not_generalization=True))
        del d, warm; source.gc.collect()
    source.once(PUBLIC/'full_fitting_losses.json', source.stable(dict(losses=losses, mask_overlap=overlaps,
        teacher_extrapolation=extrapolation,
        scope='full_B_empirical_query_weights_not_minibatch_not_validation')))
    held_action_audit(values)
    return losses


def interpretation(result):
    pair = result['contrasts']['held_minus_in_source']
    return dict(known_contrast_pp=pair['known_pp']['mean'],
        completion_lower_pp=pair['completion_lower_pp']['mean'],
        completion_upper_pp=pair['completion_upper_pp']['mean'],
        expansion_supported=result['expansion_supported'], completion_bounds_are_sampling_CI=False,
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        original24group_primary='not_estimable_not_passed')


def main():
    run.api.torch.set_num_threads(4); run.api.torch.set_num_interop_threads(1)
    result = source.read(PUBLIC/'analysis.json'); verification = source.read(PUBLIC/'verification.json')
    if verification['analysis_sha256'] != source.sha(PUBLIC/'analysis.json'):
        raise ValueError('Exact completed result verification required')
    if not verification['all3_predictions_actions_metrics_exact']:
        raise ValueError('All prediction/action/metric replay required')
    teacher = source.read(PUBLIC/'teacher_freeze.json'); student = source.read(PUBLIC/'training_freeze.json')
    if teacher['complete_heads'] != 8 or student['complete_heads'] != 6:
        raise ValueError('All planned real fits required')
    phases = {p: source.read(PUBLIC/(p+'_execution.json')) for p in ('pilot', 'teachers', 'students', 'evaluate', 'verify')}
    teachers = [source.read(source.checked(ref)) for ref in teacher['teachers']]
    students = [source.read(source.checked(ref)) for ref in student['students']]
    discovery = [source.read(source.checked(ref)) for ref in teacher['discovery']]
    for row in teachers+students:
        state = run.api.base.read_checkpoint(source.checked(row['checkpoint']))
        if row['steps'] != 2000 or row['parameters'] != 32130:
            raise ValueError('Actual full-budget neural artifacts required')
        if any(int(v['step']) != 2000 for v in state['optimizer']['state'].values()):
            raise ValueError('Every neural parameter must have completed actual optimizer steps')
        if run.api.torch.equal(state['model']['encoder.weight'], state['initial_model']['encoder.weight']):
            raise ValueError('Common neural encoder must actually change')
        if not all(run.api.torch.isfinite(v).all() for v in state['model'].values()):
            raise ValueError('All trained parameters must be finite')
    for name in ('teacher_full_refit', 'student_full_refit'):
        if not source.read(PUBLIC/(name+'.json'))['exact']:
            raise ValueError('Exact full refits required')
    losses = fitting_audit(students, discovery)
    summary = interpretation(result)
    source.once(PUBLIC/'result_summary.json', source.stable(dict(summary,
        status='fresh_report_from_verified_results', evidence=source.artifact(PUBLIC/'analysis.json'),
        teacher_heads=8, student_heads=6, primary_updates=28000, full_refit_updates=4000,
        phases=phases, actual_optimizer_and_encoder_changes_verified=True,
        full_historical_test_suite_rerun=False)))
    n = lambda v: 'NA' if v is None else f'{v:.4f}'
    lines = ['# Source-Held Error Discovery: Completed', '',
        'I compared errors discovered on an unseen fitting source against errors',
        'on an already seen source using equal-sized discovery networks. Both',
        'student arms start from the same frozen kinematic neural model. The only',
        'changed training ingredient is which observed error channels receive',
        'weight10. Architecture, sampling, optimizer budget and 2% rules are fixed.', '',
        'Eight randomly initialized discovery heads and six matched student heads',
        'completed 28,000 primary updates. Independent full teacher/student refits',
        'add 4,000 updates; the real resume pilot adds200. Predictions, decisions',
        'and metrics reproduce exactly. This is real neural risk-head training,',
        'not new trajectory dynamics, a new JEPA fit or independent confirmation.', '',
        '## Held Development Results', '',
        '| Model | Known ADE gain over floor % | Hard gain % | Intervention % | Full original risk/support |',
        '|---|---:|---:|---:|---:|']
    names = ('readout_veto', 'kinematic_veto', 'continued_mse_veto', 'false_safe_priority_veto',
        'in_source_errors_veto', 'held_source_errors_veto')
    for name in names:
        p = result['held_policy'][name]; s = p['safety']
        lines.append(f"| {name} | {n(p['known']['mean'])} | {n(p['hard']['mean'])} | {n(p['switch_percent']['mean'])} | {s['supported']}/{s['required']} |")
    lines += ['', f"Held-error minus in-source-error known contrast: **{n(summary['known_contrast_pp'])} pp**.",
        f"Missing-outcome completion bounds: [{n(summary['completion_lower_pp'])}, {n(summary['completion_upper_pp'])}] pp.",
        'These bounds are not sampling confidence intervals.',
        f"Prospective expansion screen: **{summary['expansion_supported']}**. Original 24-group primary remains NA.",
        'No model is deployed or certified by this report. No identity certificate',
        'or changed support requirement. Empty matched controls remain NA.', '', '## Safety', '',
        '| Student | Known all violations | Known easy violations | Undefined selected easy | Worst defined easy-net % |',
        '|---|---:|---:|---:|---:|']
    for a in run.api.ARMS:
        s = result['held_policy'][a+'_veto']['safety']
        lines.append(f"| {a} | {s['known_all_violations']} | {s['known_easy_violations']} | {s['undefined_selected_easy']} | {n(s['worst_defined_signed_easy_net_percent'])} |")
    lines += ['', '## Runtime', '', '| Phase | PID | Seconds | Peak RSS bytes |', '|---|---:|---:|---:|']
    for name, p in phases.items():
        lines.append(f"| {name} | {p['pid']} | {p['seconds']:.2f} | {p['peak_RSS_bytes']} |")
    lines += ['', 'Native arm64 Torch CPU4, interop1, workers0; no resource probing.',
        'The first pilot failed before training because the immutable cache',
        'rejects reduced role populations. The repaired interface subsets whole',
        'sources/queries after reading the original cache. Both registrations',
        'and the preflight amendment remain; the evaluation contract is unchanged.', '',
        '## Limits And Provenance', '',
        'New teachers, students and evaluation: fresh_run. Original forecasts,',
        'router, warm heads and old control actions: cached_verified by hashes,',
        'schemas and source exclusion. Old control metrics are freshly recomputed',
        'and exactly matched. No new H/independent test: not_run, reserved.',
        'B contains only two/three/three source groups. Individual teachers see',
        'one/two/two sources; row counts and difficulty differ between teachers.',
        'One producer seed and seven exposed held SOURCE units cannot establish',
        'independent generalization. No favorable-context selection is allowed.',
        'Native obs8/pred12 image pixels, not verified metric or seconds; not',
        'true3D, not a foundation world model. Stage5C and SMC remain disabled.', '',
        '[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Exact replay](verification.json)', '']
    (PUBLIC/'README.md').write_text('\n'.join(lines))
    failure = ['# Error Discovery And Failure Analysis', '',
        'The comparison changes error discovery, not evaluation. Discovery teachers',
        'have no full-B warm weights or normalization; their held source is excluded',
        'from fitting, common normalization, target RMS and kinematic normalization.',
        'The frozen A proposer remains foreign to all B sources.', '',
        '## Supervised Priority Channels', '',
        '| Context | In-source all/easy | Held-source all/easy |', '|---|---:|---:|']
    for d in discovery:
        failure.append(f"| {d['job']} | {d['priority_channels']['in_source_errors']} | {d['priority_channels']['held_source_errors']} |")
    failure += ['', '## Complete Fitting Loss', '',
        'Identical empirical query weighting; not minibatch loss. Priority masks',
        'differ, so both mask-specific losses are shown for every student.', '',
        '| Context | Model | Anchor MSE | In-source-priority MSE | Held-source-priority MSE |',
        '|---|---|---:|---:|---:|']
    for r in losses:
        failure.append(f"| {r['job']} | {r['arm']} | {r['full_anchor_MSE']:.9f} | "
            f"{r['full_priority_MSE']['in_source_errors']:.9f} | {r['full_priority_MSE']['held_source_errors']:.9f} |")
    failure += ['', 'These are supervised output channels across repeated fitting contexts,',
        'not independent people, complete admitted decisions or safety passes.',
        'Unknown labels never receive priority; their bound-aware anchor remains.', '',
        '## Prospective Context Results', '', '| Context | Completion lower contrast pp | Expansion condition |', '|---|---:|---|']
    for c in result['per_context']:
        failure.append(f"| {c['job']} | {n(c['completion_lower_pp'])} | {c['expansion_condition']} |")
    failure += ['', '## Interpretation Limits', '',
        'A reduction in fitting error or a lower switch rate is not proof of safe',
        'held utility. Negative or unsupported views remain in the result.',
        'Teacher source scarcity is a plausible limitation, not a proven sole',
        'cause. This comparison cannot disentangle every source-size/difficulty',
        'effect. Mask cardinality and total priority-gradient mass can differ;',
        'the experiment tests the fixed discovery procedure, not an isolated',
        'equal-weight causal effect of individual error identities. Changing',
        'error discovery does not add new observed motion',
        'information. No new guarantee of calibrated expected harm is claimed.',
        'The unchanged original support deficit remains separate from model',
        'quality; training cannot make an absent reference denominator observed.', '']
    (PUBLIC/'failure_analysis.md').write_text('\n'.join(failure))
    commands = '\n'.join(f'.venv-pytorch/bin/python scripts/run_m3w_moving_source_error_priority.py {p}'
        for p in ('register', 'pilot', 'teachers', 'students', 'evaluate', 'verify'))
    (PUBLIC/'reproducibility.md').write_text('# Reproducibility\n\n```sh\n'+commands+'\n```\n\n'
        'Use --resume only for an interrupted unfinished phase with identical registered files.\n'
        'Completed phases intentionally reject overwriting. Reuse verified artifacts or a\n'
        'separately registered rerun location. Both full refits are already exact.\n'
        'Checkpoints, data, private prediction caches and local environment are not in Git.\n'
        'The operative registration is registration_v2.json. No reserved data are needed.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
