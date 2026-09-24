"""Fixed source-only event-moment training on unchanged neural forecasts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_cv_reference as prior
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from src.world_model.m3w_european_conditional_risk import (
    event_labels, fitting_support, nonnegative_moments, decisions, CONTROL_ARMS,
)
from src.world_model.m3w_native_gain_harm import (
    preprocess, fit_ridge, predict_ridge, fit_neural, predict_neural, build_head,
)
from src.world_model.m3w_european_cv_reference import choose_errors
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics

PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_conditional_risk_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_conditional_risk_v1'
CONFIG = 'configs/m3w_european_conditional_risk_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_conditional_risk.py',
    'src/world_model/m3w_european_conditional_risk.py', 'tests/test_m3w_european_conditional_risk.py',
    'outputs/publication_readiness_2026_09/european_conditional_risk_v1/registration.md')


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def assert_identity(identity):
    for path, expected in identity['bindings'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen event-risk code changed: '+path)
    prior.assert_identity(identity['previous_identity'])


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    oldreg, data, oldidentity, designs, qmask = prior.load()
    for key in ('seeds', 'arms', 'ridge_alpha', 'training', 'pair_weight', 'edge_radius_bbox_widths',
                'proximity_threshold_bbox_widths', 'solver_seconds', 'bootstrap_resamples', 'bootstrap_seed'):
        if reg[key] != oldreg[key]:
            raise ValueError('Unregistered simultaneous training/control change: '+key)
    if (reg['events'] != ['all', 'easy'] or reg['support_controls'] != ['no_guard', 'source_zero_guard']
            or reg['predicted_risk_budget'] != .02
            or reg['utility_head'] != 'frozen_european_cv_reference_neural_underharm4'
            or any(reg[k] for k in ('independent_reserved_readout', 'deployment_promotion', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source-only event experiment required')
    for name in ('verification.json', 'checkpoint_replay.json'):
        receipt = json.loads((prior.PUBLIC/name).read_text())
        if receipt['analysis_sha256'] != digest(prior.PUBLIC/'analysis.json'):
            raise ValueError('Verified prior readout required')
    identity = dict(bindings={p: digest(ROOT/p) for p in FILES}, previous_identity=oldidentity,
        previous_analysis_sha256=digest(prior.PUBLIC/'analysis.json'), folds=oldidentity['folds'],
        query_mask_sha256=array_hash(qmask), numpy=np.__version__, torch=torch.__version__)
    immutable_json(PRIVATE/'identity.json', identity)
    immutable_json(PUBLIC/'matrix.json', dict(identity=identity, ridge_fits=18, neural_fits=18,
        neural_steps_per_fit=2000, new_forecaster_fits=0, source_rows=len(data['sites']),
        joint_rows=int(qmask.sum()), events=reg['events'], support_controls=reg['support_controls'],
        reserved_roles_opened=False))
    return reg, data, identity, designs, qmask


def prepare_event(a, data, design, event):
    fit = design['train_ids']
    cv = np.asarray(data['baseline_ade'][fit, 1])
    y = event_labels(cv, a['y'][:, 1], easy_cut=design['easy_cut'], event=event)
    pr = preprocess(a['x'][fit], y, cv, data['sites'][fit], sorted(set(data['sites'][design['held_ids']]))[0])
    for key in ('mean', 'std', 'weights', 'known'):
        np.testing.assert_array_equal(pr[key], a['pr'][key])
    if pr['cost_scale'] != a['pr']['cost_scale']:
        raise ValueError('Unchanged fitting cost scale required')
    support = fitting_support(cv, data['sites'][fit])
    lineage = dict(a['lineage'], event=event, event_labels_sha256=array_hash(y),
        source_support=support, channel_order=['baseline_error_mass', 'positive_harm_mass'])
    return y, pr, lineage


def checked_head(name, identity):
    path = PRIVATE/'heads'/name/'complete.json'
    if not path.exists():
        raise ValueError('All36 fixed endpoints required before readout')
    r = json.loads(path.read_text())
    if r['identity']['identity'] != identity or not prior.previous.artifacts_ok(r):
        raise ValueError('Incomplete or changed event head')
    if 'neural' in name and (r['fit']['step'] != 2000 or not r['fit']['complete']):
        raise ValueError('Incomplete neural budget')
    return r


def train(reg, data, identity, designs, *, resume, pilot):
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        a = prior.assemble(data, identity['previous_identity'], key, design, ti)
        fit, held = design['train_ids'], design['held_ids']
        for event in reg['events']:
            y, pr, lineage = prepare_event(a, data, design, event)
            for arm in (['neural_underharm4'] if pilot else reg['arms']):
                name = key+'_'+event+'_'+arm
                directory = PRIVATE/'heads'/name
                hid = dict(identity=identity, lineage=lineage, arm=arm)
                if (directory/'complete.json').exists():
                    r = checked_head(name, identity)
                    if r['identity'] != hid:
                        raise ValueError('Completed event lineage changed')
                    beat('verified_completed_head', trial=name)
                    continue
                started = time.monotonic()
                beat('event_head_training', trial=name, train_rows=len(fit), held_rows=len(held), pilot=pilot)
                if arm == 'ridge':
                    head = fit_ridge(a['x'][fit], y, pr, alpha=reg['ridge_alpha'])
                    raw = predict_ridge(head, a['x'][held], np.zeros(len(held), bool), pr)
                    cp = directory/'ridge.pt'
                    prior.previous.save_state(cp, dict(identity=hid, head=head, preprocess=pr))
                    report = dict(method='weighted_ridge_closed_form', gradient_updates=0,
                                  supported_rows=int(pr['known'].sum()))
                else:
                    model, report = fit_neural(a['x'][fit], y, data['sites'][fit], np.zeros(len(fit), bool), pr,
                        seed=ti['seed'], arm='underharm4', settings=reg['training'], identity=hid,
                        directory=directory, resume=resume, stop_at=100 if pilot else None,
                        heartbeat=lambda **kw: beat(trial=name, **kw))
                    cp = directory/'checkpoint.pt'
                    if not report['complete']:
                        immutable_json(PRIVATE/'pilot.json', dict(identity=hid, fit=report, checkpoint=artifact(cp),
                            phase_seconds=time.monotonic()-started, result_source='fresh_run_inside_fixed_budget'))
                        assert_identity(identity)
                        beat('pilot_complete_requires_resume', trial=name, steps=report['step'])
                        return
                    raw = predict_neural(model, a['x'][held], np.zeros(len(held), bool), pr)
                path = directory/'scores.npz'
                prior.atomic_arrays(path, ids=held, moments=nonnegative_moments(raw, a['same'][held]))
                r = dict(identity=hid, fit=report, wall_seconds=time.monotonic()-started,
                         artifacts=dict(checkpoint=artifact(cp), scores=artifact(path)))
                immutable_json(directory/'complete.json', r)
                assert_identity(identity)
                beat('event_head_complete', trial=name, seconds=r['wall_seconds'])
            if pilot:
                return
        del a


def frozen_utility(key, identity, held):
    r = prior.checked_head(key+'_neural_underharm4', identity['previous_identity'])
    path = ROOT/r['artifacts']['predicted_costs']['path']
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], held)
        utility = (z['costs'][:, 0]-z['costs'][:, 1])/r['identity']['lineage']['cost_scale']
    p = prior.PRIVATE/'decisions'/(key+'_neural_underharm4.json')
    receipt = json.loads(p.read_text())
    if receipt['identity'] != r['identity'] or digest(ROOT/receipt['path']) != receipt['sha256']:
        raise ValueError('Frozen utility decision changed')
    with np.load(ROOT/receipt['path'], allow_pickle=False) as z:
        old = {k: z[k].copy() for k in z.files}
    np.testing.assert_array_equal(old['pointwise_ids'], held)
    return utility, old


def get_decisions(reg, data, a, held, qmask, moments, utility, name, r, guard, verify):
    path = PRIVATE/'decisions'/(name+'_'+guard+'.npz')
    ident = dict(head_identity=r['identity'], support_guard=guard)
    if path.with_suffix('.json').exists():
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['identity'] != ident or digest(path) != receipt['sha256']:
            raise ValueError('Changed event decisions')
        with np.load(path, allow_pickle=False) as z:
            choice = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(choice['pointwise_ids'], held)
        np.testing.assert_array_equal(choice['ids'], held[qmask[held]])
        return choice, receipt
    if verify:
        raise ValueError('Missing decisions cannot be called verified')
    available = guard == 'no_guard' or r['identity']['lineage']['source_support']['gate_available']
    choice, queries = decisions(reg, history=data['history'], origin=data['origin'], widths=data['width'],
        recordings=data['recordings'], frames=data['frames'], sites=data['sites'], baseline=a['b'],
        candidate=a['p'], utility=utility, moments=moments, held_ids=held, query_mask=qmask,
        support_available=available)
    prior.atomic_arrays(path, **choice)
    receipt = dict(identity=ident, path=str(path.relative_to(ROOT)), sha256=digest(path), queries=queries,
                   source_support_available=available, used_future_inputs=False)
    immutable_json(path.with_suffix('.json'), receipt)
    return choice, receipt


def evaluate(reg, data, identity, designs, qmask, verify):
    reports = {key+'_'+event+'_'+arm: checked_head(key+'_'+event+'_'+arm, identity)
        for key, _, ti in designs if ti['kind'] == 'complement'
        for event in reg['events'] for arm in reg['arms']}
    if len(reports) != 36:
        raise ValueError('Complete fixed matrix required')
    n = len(data['sites'])
    cv, cv_fde = np.asarray(data['baseline_ade'][:, 1]), np.asarray(data['baseline_fde'][:, 1])
    common, outputs, receipts = {}, {}, []
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        a = prior.assemble(data, identity['previous_identity'], key, design, ti)
        held, seed = design['held_ids'], ti['seed']
        utility, old = frozen_utility(key, identity, held)
        s = common.setdefault(seed, dict(ade=np.full(n, np.nan), fde=np.full(n, np.nan),
            strong_ade=np.full(n, np.nan), strong_fde=np.full(n, np.nan), easy=np.zeros(n, bool),
            hard=np.zeros(n, bool), seen=np.zeros(n, bool),
            oldbits={k: np.zeros(n, bool) for k in ('pointwise', *CONTROL_ARMS)}))
        if s['seen'][held].any():
            raise ValueError('Repeated outer source readout')
        s['seen'][held] = True
        s['ade'][held], s['fde'][held] = native_errors(a['p'][held].astype(float)+data['origin'][held, None],
            data['target_eval'][held], data['valid'][held], np.ones(len(held)))
        s['strong_ade'][held] = data['baseline_ade'][held, design['baseline_index']]
        s['strong_fde'][held] = data['baseline_fde'][held, design['baseline_index']]
        s['easy'][held] = (cv[held] > 0) & (cv[held] <= design['easy_cut'])
        s['hard'][held] = cv[held] >= design['hard_cut']
        s['oldbits']['pointwise'][held] = old['pointwise']
        for k in CONTROL_ARMS:
            s['oldbits'][k][old['ids']] = old[k]
        for event in reg['events']:
            _, _, lineage = prepare_event(a, data, design, event)
            truth = event_labels(cv[held], np.maximum(s['ade'][held]-cv[held], 0),
                                 easy_cut=design['easy_cut'], event=event)
            for arm in reg['arms']:
                name = key+'_'+event+'_'+arm
                r = reports[name]
                if r['identity']['lineage'] != lineage:
                    raise ValueError('Event-label producer lineage changed')
                with np.load(ROOT/r['artifacts']['scores']['path'], allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], held)
                    moments = z['moments'].copy()
                for guard in reg['support_controls']:
                    beat('event_control_readout', trial=name, guard=guard, verify=verify)
                    choice, receipt = get_decisions(reg, data, a, held, qmask, moments, utility, name, r, guard, verify)
                    receipts.append(artifact(PRIVATE/'decisions'/(name+'_'+guard+'.json')))
                    v = outputs.setdefault(f'{seed}_{event}_{arm}_{guard}', dict(seed=seed,
                        bits={k: np.zeros(n, bool) for k in ('pointwise', *CONTROL_ARMS)},
                        seen=np.zeros(n, bool), nonzero=np.zeros(n, bool), queries=[], moment_errors={}))
                    if v['seen'][held].any():
                        raise ValueError('Repeated event readout')
                    v['seen'][held] = True
                    v['bits']['pointwise'][held] = choice['pointwise']
                    for k in CONTROL_ARMS:
                        v['bits'][k][choice['ids']] = choice[k]
                    v['nonzero'][choice['ids']] = choice['matched_nonzero']
                    v['queries'].extend(receipt['queries'])
                    for site in sorted(set(data['sites'][held])):
                        use = (data['sites'][held] == site) & np.isfinite(truth).all(1)
                        v['moment_errors'][site] = dict(rows=int(use.sum()),
                            mean_absolute_error=np.abs(moments[use]-truth[use]).mean(0).tolist(),
                            predicted_mean=moments[use].mean(0).tolist(), true_mean=truth[use].mean(0).tolist())
        del a
    roster, fullmask = sorted(identity['folds']), np.ones(n, bool)
    def metric(model, reference, mask):
        return paired_scene_metrics(model[mask], reference[mask], data['sites'][mask], expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
            bootstrap_resamples=reg['bootstrap_resamples'], seed=reg['bootstrap_seed'])
    def describe(s, ade, fde, bits, mask):
        easy = metric(ade, cv, mask & s['easy'])
        zero = mask & np.isfinite(cv) & (cv == 0)
        return dict(ADE_vs_CV=metric(ade, cv, mask), FDE_vs_CV=metric(fde, cv_fde, mask),
            ADE_vs_training_selected_baseline=metric(ade, s['strong_ade'], mask),
            ADE_vs_fixed_damping097=metric(ade, data['baseline_ade'][:, 3], mask),
            hard_ADE_vs_CV=metric(ade, cv, mask & s['hard']), positive_easy_ADE_vs_CV=easy,
            complete_ADE_vs_CV=metric(ade, cv, mask & data['valid'].all(1)),
            zero_CV=dict(rows=int(zero.sum()), harmed_rows=int((ade[zero] > 0).sum()),
                         total_added_error=float(ade[zero].sum())),
            switch_rate=float(bits[mask].mean()), indexed_rows=int(mask.sum()),
            safety_observed_pass=bool(easy['worst_scene_gain_percent'] is not None
                and easy['worst_scene_gain_percent'] >= -2 and not np.any(ade[zero] > 0)),
            zero_safety_supported=bool(zero.any()), calibrated_safety=False)
    summaries, references = {}, {}
    for seed, s in common.items():
        if not s['seen'].all():
            raise ValueError('Incomplete source predictor readout')
        refs = {}
        for name, ade, fde, bits in [
            ('CV', cv, cv_fde, np.zeros(n, bool)),
            ('training_selected_baseline', s['strong_ade'], s['strong_fde'], np.zeros(n, bool)),
            ('fixed_damping097', data['baseline_ade'][:, 3], data['baseline_fde'][:, 3], np.zeros(n, bool)),
            ('neural', s['ade'], s['fde'], np.ones(n, bool)),
            *[(k, choose_errors(bits, s['ade'], cv), choose_errors(bits, s['fde'], cv_fde), bits)
              for k, bits in s['oldbits'].items()]]:
            refs[name] = dict(joint_population=describe(s, ade, fde, bits, qmask))
            if name not in CONTROL_ARMS:
                refs[name]['full'] = describe(s, ade, fde, bits, fullmask)
        references[str(seed)] = refs
    for name, v in outputs.items():
        if not v['seen'].all():
            raise ValueError('Incomplete event population')
        s = common[v['seed']]
        errors = {k: choose_errors(b, s['ade'], cv) for k, b in v['bits'].items()}
        joint = {k: describe(s, errors[k], choose_errors(b, s['fde'], cv_fde), b, qmask)
                 for k, b in v['bits'].items() if k != 'pointwise'}
        olderr = choose_errors(s['oldbits']['pointwise'], s['ade'], cv)
        comparisons = dict(pointwise_vs_previous=metric(errors['pointwise'], olderr, fullmask),
            joint_vs_independent=metric(errors['joint'], errors['independent'], qmask),
            joint_exact_vs_independent=metric(errors['joint_exact'], errors['independent'], qmask & v['nonzero']),
            joint_exact_vs_unary=metric(errors['joint_exact'], errors['unary_exact'], qmask & v['nonzero']))
        summaries[name] = dict(full=describe(s, errors['pointwise'],
            choose_errors(v['bits']['pointwise'], s['fde'], cv_fde), v['bits']['pointwise'], fullmask),
            joint_population=joint, comparisons=comparisons, moment_errors=v['moment_errors'],
            queries=dict(total=len(v['queries']), matched=sum(q['matched'] for q in v['queries']),
                matched_nonzero=sum(q['matched_nonzero'] for q in v['queries']),
                with_edges=sum(q['edges'] > 0 for q in v['queries']),
                solver_failures={k: sum(not q['arms'][k]['solver_optimal'] for q in v['queries']) for k in CONTROL_ARMS}))
    factorial = {}
    for seed in reg['seeds']:
        s = common[seed]
        for arm in reg['arms']:
            for guard in reg['support_controls']:
                left = outputs[f'{seed}_easy_{arm}_{guard}']['bits']
                right = outputs[f'{seed}_all_{arm}_{guard}']['bits']
                factorial[f'{seed}_{arm}_{guard}_easy_vs_all'] = {
                    rule: metric(choose_errors(left[rule], s['ade'], cv), choose_errors(right[rule], s['ade'], cv), mask)
                    for rule, mask in (('pointwise', fullmask), ('joint', qmask))}
    result = dict(result_source='fresh_run_event_moment_heads_and_fixed_factorial_controls',
        identity=identity, training={k: dict(fit=r['fit'], lineage=r['identity']['lineage'],
            artifacts=r['artifacts'], wall_seconds=r['wall_seconds']) for k, r in reports.items()},
        controls=receipts, source_rows=n, joint_rows=int(qmask.sum()), references=references,
        policies=summaries, factorial_contrasts=factorial, actual_cost_heads=36,
        new_forecaster_training=False, forecast_source='cached_verified',
        independent_reserved_readout=False, calibrated_safety=False, deployment_changed=False,
        metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
    assert_identity(identity)
    immutable_json(PUBLIC/'analysis.json', result)
    if verify:
        immutable_json(PUBLIC/'verification.json', dict(result_source='cached_verified',
            analysis_sha256=digest(PUBLIC/'analysis.json'), metrics_recomputed=True, new_training=False))
    beat('event_readout_complete', heads=36, policies=len(summaries), source_rows=n)


def replay(reg, data, identity, designs):
    if not (PUBLIC/'analysis.json').exists():
        raise ValueError('Completed readout required')
    checks = []
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        a = prior.assemble(data, identity['previous_identity'], key, design, ti)
        held = design['held_ids'][:4096]
        for event in reg['events']:
            _, pr, _ = prepare_event(a, data, design, event)
            for arm in reg['arms']:
                name = key+'_'+event+'_'+arm
                r = checked_head(name, identity)
                cp = torch.load(ROOT/r['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
                if cp['identity'] != r['identity']:
                    raise ValueError('Checkpoint identity mismatch')
                for k in ('mean', 'std', 'constant', 'weights', 'known'):
                    np.testing.assert_array_equal(pr[k], cp['preprocess'][k])
                if arm == 'ridge':
                    raw = predict_ridge(cp['head'], a['x'][held], np.zeros(len(held), bool), pr)
                else:
                    model = build_head(reg['training']['width'], pr, ti['seed'])
                    model.load_state_dict(cp['model'])
                    raw = predict_neural(model, a['x'][held], np.zeros(len(held), bool), pr)
                fresh = nonnegative_moments(raw, a['same'][held])
                with np.load(ROOT/r['artifacts']['scores']['path'], allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'][:len(held)], held)
                    saved = z['moments'][:len(held)].copy()
                np.testing.assert_array_equal(fresh, saved)
                checks.append(dict(trial=name, rows=len(held), exact=True, max_difference=0.,
                    ids_sha256=array_hash(held), checkpoint_sha256=r['artifacts']['checkpoint']['sha256']))
        del a
    assert_identity(identity)
    immutable_json(PUBLIC/'checkpoint_replay.json', dict(result_source='fresh_run_checkpoint_inference',
        analysis_sha256=digest(PUBLIC/'analysis.json'), checks=checks, all_passed=True, new_training=False))
    beat('checkpoint_replay_complete', heads=len(checks), maximum_difference=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepare', 'pilot', 'train', 'resume', 'evaluate', 'verify', 'replay'):
        parser.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    if not any((args.prepare, args.pilot, args.train, args.evaluate, args.verify, args.replay)):
        parser.error('Explicit phase required')
    if args.pilot and any((args.train, args.evaluate, args.verify, args.replay)):
        parser.error('Pilot is separate and resumes within budget')
    for path in (PRIVATE, PUBLIC):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlinked experiment destination')
        path.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        reg, data, identity, designs, qmask = load()
        beat('verified_event_inputs', rows=len(data['sites']), architecture=platform.machine(),
             threads=torch.get_num_threads(), workers=0)
        if args.train or args.pilot:
            train(reg, data, identity, designs, resume=args.resume, pilot=args.pilot)
        if args.evaluate or args.verify:
            evaluate(reg, data, identity, designs, qmask, args.verify)
        if args.replay:
            replay(reg, data, identity, designs)
        beat('phase_complete')


if __name__ == '__main__':
    main()
