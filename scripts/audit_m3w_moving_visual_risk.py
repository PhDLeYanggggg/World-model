"""Posthoc source-held calibration audit after the immutable full readout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_visual_risk as run

np = run.np
NAMES = ('gain', 'harm', 'reference', 'easy_reference', 'easy_harm')


def score(prediction, target, probability, known, weights):
    p, y, q, mask, w = map(np.asarray, (prediction, target, probability, known, weights))
    n = len(mask)
    if (p.shape != (n, 5) or y.shape != p.shape or q.shape != (n,) or w.shape != (n,)
            or mask.dtype != bool or not mask.any() or (w < 0).any() or w[mask].sum() <= 0
            or not all(np.isfinite(v).all() for v in (p, y, q, w))):
        raise ValueError('Known-label conditional costs and separate support roles required')
    ww = w/w.sum(); wk = w[mask]/w[mask].sum()
    return dict(candidates=n, known_candidates=int(mask.sum()),
        observation_Brier=float(ww@((q-mask)**2)),
        conditional_MSE=dict(zip(NAMES, np.einsum('n,nd->d', wk, (p[mask]-y[mask])**2).tolist())),
        conditional_bias=dict(zip(NAMES, np.einsum('n,nd->d', wk, p[mask]-y[mask]).tolist())),
        mean_observation_probability=float(ww@q), observed_fraction=float(ww@mask))


def tail_case(a, reg):
    rows = [dict(job=r['job'], source=r['source'], seed=r['seed'], recording=k, **v)
        for r in a['rows'] if r['arm'] == 'motion_geometry'
        for k, v in r['policies']['joint_free']['recordings'].items()
        if v['known_easy_net_degradation'] is not None]
    worst = max(rows, key=lambda r: r['known_easy_net_degradation'])
    node = next(n for n in reg['jobs'] if n['job']['key'] == worst['job'])
    own, old = run.own_input(node); data = run.parent.full.load_data(); ix = old['local_ids']
    target = data['target_eval'][ix]-data['origin'][ix, None]; valid = data['valid'][ix]
    errors = run.parent.full.method.original.errors
    be, _ = errors(old['floor'], target, valid); ce, _ = errors(old['cv'], target, valid)
    banks, paths = {}, {}
    for arm in ('quality_geometry', 'motion_geometry'):
        ref = run.read(run.PUBLIC/'inference'/f'{worst["job"]}_{arm}_seed{worst["seed"]}.json')['cache']
        banks[arm] = run.parent.load_cache(ref)
        paths[arm] = run.parent.full.method.paths_for(old['paths'], old['floor'], banks[arm]['joint_free'])
    pe, _ = errors(paths['motion_geometry'], target, valid)
    easy = valid.any(1)&(ce > 0)&(ce <= node['job']['easy_cut'])
    eligible = easy&(old['sites'] == worst['source'])&(old['recordings'].astype(str) == worst['recording'])
    harm = np.where(eligible, pe-be, -np.inf); j = int(np.argmax(harm))
    if not eligible[j] or harm[j] <= 0: raise ValueError('Expected an actual harmed easy case')
    names = sorted(run.read(ROOT/'outputs/publication_readiness_2026_09/moving_development_admission_v1/admission.json')['recordings'])
    result = dict(selection='worst_motion_joint_free_signed_easy_recording_then_largest_known_positive_easy_harm',
        representative_sample=False, posthoc_only=True, job=worst['job'], source=worst['source'], seed=worst['seed'],
        recording=int(old['recordings'][j]), recording_name=names[int(old['recordings'][j])],
        frame=int(old['frames'][j]), global_id=int(old['ids'][j]), known_future_steps=int(valid[j].sum()),
        floor_available_ADE_pixel=float(be[j]), actual_added_ADE_pixel=float(harm[j]), arms={})
    for arm, bank in banks.items():
        choice = int(bank['joint_free'][j]); error, _ = errors(paths[arm][j:j+1], target[j:j+1], valid[j:j+1])
        at = [j, choice] if choice >= 0 else None
        result['arms'][arm] = dict(action=choice, available_ADE_pixel=float(error[0]),
            predicted_moments=dict(zip(NAMES, bank['moments'][j, choice].tolist())) if at else None,
            predicted_observation_probability=float(bank['probability'][j, choice]) if at else None)
    run.once(run.PUBLIC/'failure_case.json', run.stable(result))


def main():
    cfg, reg = run.register()
    a = run.read(run.PUBLIC/'analysis.json')
    if not run.read(run.PUBLIC/'inference_freeze.json')['all36_choices_before_readout']:
        raise ValueError('No outcome audit before all original choices are frozen')
    if a['deployment_changed'] or not a['original_rules_unchanged']:
        raise ValueError('Original complete contract must stay intact')
    rows = []
    for node in reg['jobs']:
        key = node['job']['key']
        ref = next(x['cache'] for x in run.read(run.parent.PUBLIC/'own.json')['banks'] if x['job'] == key)
        own = run.parent.load_cache(ref)
        for seed in (17, 29, 43):
            for arm in run.method.ARMS:
                cache = run.parent.load_cache(run.read(run.PUBLIC/'inference'/f'{key}_{arm}_seed{seed}.json')['cache'])
                at = (own['row'], own['action'])
                p = cache['conditional'][at]/own['scales'][:, None]
                q = cache['probability'][at]
                for source in node['part']['names']:
                    mask = own['sites'] == source
                    rows.append(dict(job=key, seed=seed, arm=arm, source=source,
                        **score(p[mask], own['moments'][mask], q[mask], own['observed'][mask], own['weights'][mask])))
    summary = {}
    for arm in run.method.ARMS:
        rr = [r for r in rows if r['arm'] == arm]
        if len(rr) != 24: raise ValueError('All8source groups and3risk seeds required')
        summary[arm] = dict(observation_Brier=float(np.mean([r['observation_Brier'] for r in rr])),
            conditional_MSE={name: float(np.mean([r['conditional_MSE'][name] for r in rr])) for name in NAMES},
            conditional_bias={name: float(np.mean([r['conditional_bias'][name] for r in rr])) for name in NAMES})
    result = run.stable(dict(result_source='fresh_run_posthoc_analysis_of_frozen_predictions',
        rows=rows, summary=summary, common_units='past_causal_history_scale_not_fit_RMS',
        averaging='equal_source_and_risk_seed_macro_with_within_source_query_balanced_candidate_weights',
        unknown_rows_excluded_from_conditional_MSE_not_imputed_as_zero=True,
        unknown_rows_retained_in_observation_Brier=True,
        used_for_model_selection=False, thresholds_changed=False, original_risk_rules_unchanged=True))
    run.once(run.PUBLIC/'calibration_analysis.json', result)
    tail_case(a, reg)
    lines = ['# Frozen Source-Held Calibration Audit', '',
        'Posthoc analysis after all36choices and the complete original readout.',
        'This does not change the registered decision, select a model or tune a threshold.',
        'Costs use a common past-history scale, not the different fitting RMS.',
        'Conditional MSE uses known labels only. Unknowns stay in observation Brier;',
        'they are not assigned zero-error ground truth. We average eight source groups',
        'and three risk seeds with within-source query-balanced candidate weights.', '',
        '| Context | Observation Brier | Gain MSE | Harm MSE | Reference MSE | Easy reference MSE | Easy harm MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, s in summary.items():
        values = [s['observation_Brier'], *s['conditional_MSE'].values()]
        lines.append(f'| {arm} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', 'These diagnostics can explain a failed source-held policy but cannot',
        'replace per-recording positive harm, signed easy-net and support checks.',
        'They are conditional empirical errors, not calibration guarantees under',
        'selection or on unseen sources. All per-source/seed values remain in JSON.',
        'Same120+56input schema, no future inputs, no new C/H or reserved outcomes.',
        'Native8/12image-coordinate development only; not metric/seconds or physical safety.']
    (run.PUBLIC/'calibration_analysis.md').write_text('\n'.join(lines)+'\n')
    print('All96source/seed/arm conditional diagnostics frozen; no policy or threshold change.')


if __name__ == '__main__':
    main()
