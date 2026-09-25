"""All fixed source-C views; no six-locality or reserved-role outcome access."""
import json
import numpy as np
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
from src.evaluation.m3w_bridge_risk_calibration import evidence
from src.evaluation.m3w_native_metrics import paired_scene_metrics, native_errors
from src.world_model.m3w_bridge_attribution import query_groups


def support_summary(y, moments, bits, sites, bins):
    out = {}; known = np.isfinite(y).all(1)
    for site in sorted(set(sites)):
        out[site] = {}
        for b in range(3):
            pop = (sites == site) & (bins == b); use = pop & known; action = use & bits
            harm = float(y[action, 1].sum()); pred = float(moments[action, 1].sum())
            out[site][str(b)] = dict(rows=int(pop.sum()), supported_rows=int(use.sum()),
                actions=int((bits & pop).sum()), unknown_actions=int((bits & pop & ~known).sum()),
                true_selected_harm=harm, predicted_selected_harm=pred,
                predicted_over_actual=pred/harm if harm > 0 else None,
                actual_easy_selected_harm=float(y[action, 3].sum()),
                actual_reference_mass=float(y[use, 0].sum()))
    return out


def evaluate(run, cfg, identity):
    freeze = run.PUBLIC/'decision_freeze.json'; run.previous.require_committed(freeze)
    training = run.checked_training(identity)
    assert json.loads(freeze.read_text())['manifest'] == run.artifact(run.PRIVATE/'training_complete.json')
    complete = run.PUBLIC/'completion_checks.json'
    if complete.exists():
        old = json.loads(complete.read_text()); assert old['identity'] == identity and old['all_passed']
        for ref in old['groups']+[old['seeds']]: assert run.artifact(run.ROOT/ref['path']) == ref
        run.beat('cached_verified_source_C_readout'); return
    by_pair = {p:[] for p in cfg['pairs']}; refs = []
    for g, data, pairs in run.contexts(identity):
        ci = pairs['C']['ids']; name = g['group']; sites = data['sites'][ci]; roster = sorted(set(sites))
        queries = query_groups(sites, data['recordings'][ci], data['frames'][ci])
        cv = data['baseline_ade'][ci, 1]; masks = dict(all=np.ones(len(ci), bool),
            easy=(cv > 0)&(cv <= g['easy_cut']), hard=cv >= g['hard_cut'], complete=data['valid'][ci].all(1))
        for pair in cfg['pairs']:
            path = run.PUBLIC/'groups'/(name+'_'+pair+'.json'); receipt_path = run.PRIVATE/'eval_receipts'/path.name
            if path.exists():
                assert run.artifact(path) == json.loads(receipt_path.read_text())
                row = json.loads(path.read_text()); assert row['identity'] == identity
                by_pair[pair].append(row); refs.append(run.artifact(path)); continue
            run.beat('source_C_readout', group=name, pair=pair)
            x, env, r, p = pairs['C'][pair]; checks = dict(coordinates=0, metrics=0)
            def errors(pred):
                absolute = pred.astype(float)+data['origin'][ci, None]
                a = native_errors(absolute, data['target_eval'][ci], data['valid'][ci], np.ones(len(ci)))
                b = run.base.cross.independent.coordinate_errors(absolute, data['target_eval'][ci], data['valid'][ci])
                for v, w in zip(a, b): run.base.cross.independent.close(v, w); checks['coordinates'] += 1
                return a
            ra, rf = errors(r); pa, pf = errors(p)
            source = run.previous.PRIVATE/'source'/(name+'_'+pair)
            with np.load(source/'labels.npz', allow_pickle=False) as z:
                for key, v in (('reference', ra), ('candidate', pa), ('cv', cv)):
                    np.testing.assert_array_equal(v, z[key])
            with np.load(source/'scores.npz', allow_pickle=False) as z: old = {k:z[k].copy() for k in z.files}
            new = {}
            for arm in cfg['arms']:
                with np.load(run.PRIVATE/'heads'/(name+'_'+pair+'_'+arm)/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], ci); new[arm] = z['scores'].copy()
            actions = run.decision_bank(old, new, env, queries, ci)
            with np.load(run.PRIVATE/'decisions'/(name+'_'+pair+'.npz'), allow_pickle=False) as z:
                for k, v in actions.items(): np.testing.assert_array_equal(v, z[k])
                bins = z['support_bins'].copy()
            def metric(a, b, mask):
                value = paired_scene_metrics(a[mask], b[mask], sites[mask], expected_scenes=roster,
                    dataset='EuropeanSquares_source_development', coordinate_unit='image_pixel',
                    bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
                run.base.cross.independent.check_metric(a, b, sites, mask, value, cfg); checks['metrics'] += 1
                return value
            preds = {k:(np.where(v, pa, ra), np.where(v, pf, rf)) for k,v in actions.items()}
            target = run.method.event_targets(cv, ra, pa, g['easy_cut']); views = {}
            for k, (ade, fde) in preds.items():
                risk = evidence(actions[k], cv, ra, pa, sites, easy_cut=g['easy_cut'])
                views[k] = dict(ADE_vs_raw={s:metric(ade, preds['raw_neural'][0], m) for s,m in masks.items()},
                    ADE_vs_reference={s:metric(ade, ra, m) for s,m in masks.items()},
                    easy_vs_CV=metric(ade, cv, masks['easy']), FDE_vs_raw=metric(fde, preds['raw_neural'][1], masks['all']),
                    switch_rate=float(actions[k].mean()), risk=risk,
                    tails={s:dict(p95=float(np.nanquantile(ade[sites == s], .95)),
                                  p99=float(np.nanquantile(ade[sites == s], .99))) for s in roster})
            contrasts = {}
            tests = [(f'selected_{mode}', f'mean_{mode}') for mode in ('all', 'dual', 'scene', 'joint')]
            tests += [(f'{arm}_joint', f'{arm}_dual') for arm in cfg['arms']]
            tests += [('selected_joint', 'selected_hash_matched'), ('selected_joint', 'raw_neural'),
                      ('selected_dual', 'raw_neural'), ('mean_all', 'raw_neural')]
            for a, b in tests:
                contrasts[a+'_vs_'+b] = {s:metric(preds[a][0], preds[b][0], m) for s,m in masks.items()}
            diagnostics = dict(raw=support_summary(target, old['neural__all_risk'], actions['raw_neural'], sites, bins))
            for arm in cfg['arms']:
                diagnostics[arm] = support_summary(target, new[arm][:, :2], actions[arm+'_joint'], sites, bins)
            moments = {}
            for arm in cfg['arms']:
                moments[arm] = {}
                for site in roster:
                    use = (sites == site)&np.isfinite(target).all(1); a = new[arm][use]; b = target[use]
                    moments[arm][site] = dict(predicted_sum=a.sum(0).tolist(), actual_sum=b.sum(0).tolist(),
                        mse=((a-b)**2).mean(0).tolist(), rows=int(use.sum()))
            row = dict(identity=identity, group=name, producer=g['producer'], controller=g['controller'],
                seed=int(name.split('_seed')[1].split('_')[0]), pair=pair, views=views, contrasts=contrasts,
                diagnostics=diagnostics, moments=moments, checks=checks, verified=True,
                rows=len(ci), queries=len(queries), source_C_roster=roster,
                result_source='fresh_run_frozen_source_C_evaluation', role=cfg['readout_role'])
            run.immutable_json(path, row); run.immutable_json(receipt_path, run.artifact(path))
            by_pair[pair].append(row); refs.append(run.artifact(path))
    seed_path = run.PUBLIC/'seed_averaged_metrics.json'
    run.immutable_json(seed_path, {p:seed_summary(rows, cfg) for p, rows in by_pair.items()})
    run.immutable_json(complete, dict(identity=identity, groups=refs, seeds=run.artifact(seed_path), all_passed=True,
        checks={k:sum(r['checks'][k] for rows in by_pair.values() for r in rows) for k in ('coordinates', 'metrics')},
        evaluation_role=cfg['readout_role'], selection_outcomes_used=False,
        reserved_calibration_opened=False, confirmation_opened=False, engineering_completion_not_scientific_success=True))
    run.beat('source_C_readout_complete', groups=len(refs))
