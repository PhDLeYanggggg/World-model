"""Versioned source-only CV-reference repair; existing predictors are immutable."""
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
from scripts import run_m3w_european_source_intervention as previous
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from src.world_model.m3w_european_cv_reference import cv_reference_design, query_decisions, choose_errors, CONTROL_ARMS
from src.world_model.m3w_native_gain_harm import fit_ridge, predict_ridge, fit_neural, predict_neural, build_head
from src.world_model.m3w_european_source_intervention import paired_cost_labels
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics

PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_cv_reference_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cv_reference_v1'
CONFIG = 'configs/m3w_european_cv_reference_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_cv_reference.py',
    'src/world_model/m3w_european_cv_reference.py', 'tests/test_m3w_european_cv_reference.py',
    'src/evaluation/m3w_native_metrics.py',
    'outputs/publication_readiness_2026_09/european_cv_reference_v1/registration.md')


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    preg, data, prior, designs, qmask = previous.load()
    for key in ('seeds', 'arms', 'ridge_alpha', 'training', 'joint_queries_per_locality',
                'query_salt', 'predicted_positive_harm_budget', 'pair_weight',
                'edge_radius_bbox_widths', 'proximity_threshold_bbox_widths',
                'solver_seconds', 'bootstrap_resamples', 'bootstrap_seed'):
        if reg[key] != preg[key]:
            raise ValueError('Unregistered simultaneous factor change: '+key)
    if reg['reference_baseline_index'] != 1 or any(reg[k] for k in (
            'independent_reserved_readout', 'deployment_promotion', 'stage5c_executed', 'smc_enabled')):
        raise ValueError('Fixed CV reference and source-only boundaries required')
    for name in ('verification.json', 'checkpoint_replay.json'):
        receipt = json.loads((previous.PUBLIC/name).read_text())
        if receipt['analysis_sha256'] != digest(previous.PUBLIC/'analysis.json'):
            raise ValueError('Verified previous result required')
    identity = dict(bindings={p: digest(ROOT/p) for p in FILES}, previous_identity=prior,
        previous_analysis_sha256=digest(previous.PUBLIC/'analysis.json'),
        folds=prior['folds'], query_mask_sha256=array_hash(qmask),
        numpy=np.__version__, torch=torch.__version__, reference_baseline_index=1)
    immutable_json(PRIVATE/'identity.json', identity)
    immutable_json(PUBLIC/'matrix.json', dict(identity=identity, seeds=reg['seeds'],
        outer_folds=3, ridge_fits=9, neural_fits=9, neural_steps_per_fit=2000,
        new_forecaster_fits=0, source_rows=len(data['sites']), joint_rows=int(qmask.sum()),
        held_out_roles_opened=False))
    return reg, data, identity, designs, qmask


def assert_identity(identity):
    for path, sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Completed/active CV-reference code changed: '+path)
    previous.assert_identity(identity['previous_identity'])


def assemble(data, identity, key, design, ti):
    a = previous.assemble(data, identity, key, cv_reference_design(design), ti)
    a['lineage']['frozen_neural_producer_internal_baseline_index'] = design['baseline_index']
    if a['lineage']['baseline_index'] != 1:
        raise ValueError('Decision reference is not causal CV')
    return a


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def atomic_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def checked_head(name, identity):
    r = json.loads((PRIVATE/'heads'/name/'complete.json').read_text())
    if r['identity']['identity'] != identity or not previous.artifacts_ok(r):
        raise ValueError('Incomplete or changed CV-reference head')
    if 'neural' in name and (not r['fit']['complete'] or r['fit']['step'] != 2000):
        raise ValueError('Fixed training budget not complete')
    return r


def train(reg, data, identity, designs, *, resume, pilot):
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        a = assemble(data, identity, key, design, ti)
        fit, held = design['train_ids'], design['held_ids']
        for arm in (['neural_underharm4'] if pilot else reg['arms']):
            name = key+'_'+arm
            directory = PRIVATE/'heads'/name
            hid = dict(identity=identity, lineage=a['lineage'], arm=arm)
            if (directory/'complete.json').exists():
                r = checked_head(name, identity)
                if r['identity'] != hid:
                    raise ValueError('Completed head lineage changed')
                beat('verified_completed_head', trial=name)
                continue
            started = time.monotonic()
            beat('cost_head_training', trial=name, train_rows=len(fit), held_rows=len(held), pilot=pilot)
            if arm == 'ridge':
                head = fit_ridge(a['x'][fit], a['y'], a['pr'], alpha=reg['ridge_alpha'])
                costs = predict_ridge(head, a['x'][held], a['same'][held], a['pr'])
                fit_report = dict(method='weighted_ridge_closed_form', gradient_updates=0,
                    supported_rows=int(a['pr']['known'].sum()))
                cp = directory/'ridge.pt'
                previous.save_state(cp, dict(identity=hid, head=head, preprocess=a['pr']))
            else:
                model, fit_report = fit_neural(a['x'][fit], a['y'], data['sites'][fit], a['same'][fit], a['pr'],
                    seed=ti['seed'], arm='underharm4', settings=reg['training'], identity=hid,
                    directory=directory, resume=resume, stop_at=100 if pilot else None,
                    heartbeat=lambda **kw: beat(trial=name, **kw))
                cp = directory/'checkpoint.pt'
                if not fit_report['complete']:
                    immutable_json(PRIVATE/'pilot.json', dict(identity=hid, fit=fit_report,
                        checkpoint=artifact(cp), total_phase_seconds=time.monotonic()-started,
                        result_source='fresh_run_real_training_inside_fixed_budget'))
                    assert_identity(identity)
                    beat('pilot_complete_requires_resume', trial=name, steps=fit_report['step'])
                    return
                costs = predict_neural(model, a['x'][held], a['same'][held], a['pr'])
            score_path = directory/'scores.npz'
            atomic_arrays(score_path, ids=held, costs=np.maximum(costs, 0))
            r = dict(identity=hid, fit=fit_report, wall_seconds=time.monotonic()-started,
                artifacts=dict(checkpoint=artifact(cp), predicted_costs=artifact(score_path)))
            immutable_json(directory/'complete.json', r)
            assert_identity(identity)
            beat('cost_head_complete', trial=name, seconds=r['wall_seconds'])
        del a
        if pilot:
            return


def get_decisions(reg, data, a, held, costs, qmask, name, receipt, verify):
    path = PRIVATE/'decisions'/(name+'.npz')
    if path.with_suffix('.json').exists():
        r = json.loads(path.with_suffix('.json').read_text())
        if r['identity'] != receipt['identity'] or digest(path) != r['sha256']:
            raise ValueError('Changed decision receipt')
        with np.load(path, allow_pickle=False) as z:
            choice = {k: z[k].copy() for k in z.files}
    else:
        if verify:
            raise ValueError('Cannot reproduce absent control decisions')
        choice, queries = query_decisions(reg, history=data['history'], current_xy=data['origin'],
            widths=data['width'], recordings=data['recordings'], frames=data['frames'], sites=data['sites'],
            baseline=a['b'], candidate=a['p'], costs=costs, held_ids=held, query_mask=qmask,
            cost_scale=a['pr']['cost_scale'], heartbeat=beat)
        atomic_arrays(path, **choice)
        r = dict(identity=receipt['identity'], **artifact(path), queries=queries)
        immutable_json(path.with_suffix('.json'), r)
    np.testing.assert_array_equal(choice['pointwise_ids'], held)
    np.testing.assert_array_equal(choice['ids'], held[qmask[held]])
    return choice, r


def previous_decisions(name, identity, held, qmask):
    path = previous.PRIVATE/'decisions'/(name+'.npz')
    r = json.loads(path.with_suffix('.json').read_text())
    if r['identity']['identity'] != identity['previous_identity'] or digest(path) != r['sha256']:
        raise ValueError('Original control changed')
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['pointwise_ids'], held)
        np.testing.assert_array_equal(z['ids'], held[qmask[held]])
        return {k: z[k].copy() for k in z.files}


def evaluate(reg, data, identity, designs, qmask, verify):
    reports = {key+'_'+arm: checked_head(key+'_'+arm, identity)
               for key, _, ti in designs if ti['kind'] == 'complement' for arm in reg['arms']}
    n = len(data['sites'])
    cv, cv_fde = np.asarray(data['baseline_ade'][:, 1]), np.asarray(data['baseline_fde'][:, 1])
    outputs, decision_receipts = {}, []
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        beat('control_readout', trial=key, verify=verify)
        a = assemble(data, identity, key, design, ti)
        held = design['held_ids']
        ade, fde = native_errors(a['p'][held].astype(float)+data['origin'][held, None],
            data['target_eval'][held], data['valid'][held], np.ones(len(held)))
        for arm in reg['arms']:
            name = key+'_'+arm
            r = reports[name]
            if r['identity']['lineage'] != a['lineage']:
                raise ValueError('Recomputed cost lineage differs')
            with np.load(ROOT/r['artifacts']['predicted_costs']['path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], held)
                costs = z['costs'].copy()
            choice, receipt = get_decisions(reg, data, a, held, costs, qmask, name, r, verify)
            old = previous_decisions(name, identity, held, qmask)
            decision_receipts.append(artifact(PRIVATE/'decisions'/(name+'.json')))
            s = outputs.setdefault((ti['seed'], arm), dict(
                neural_ade=np.full(n, np.nan), neural_fde=np.full(n, np.nan), strong_ade=np.full(n, np.nan),
                strong_fde=np.full(n, np.nan), easy=np.zeros(n, bool), hard=np.zeros(n, bool),
                nonzero=np.zeros(n, bool), seen=np.zeros(n, bool),
                bits={k: np.zeros(n, bool) for k in ('pointwise', *CONTROL_ARMS)},
                oldbits={k: np.zeros(n, bool) for k in ('pointwise', *CONTROL_ARMS)}, queries=[]))
            if s['seen'][held].any():
                raise ValueError('Repeated outer readout')
            s['seen'][held] = True
            s['neural_ade'][held], s['neural_fde'][held] = ade, fde
            s['strong_ade'][held] = data['baseline_ade'][held, design['baseline_index']]
            s['strong_fde'][held] = data['baseline_fde'][held, design['baseline_index']]
            s['easy'][held] = (cv[held] > 0) & (cv[held] <= design['easy_cut'])
            s['hard'][held] = cv[held] >= design['hard_cut']
            s['bits']['pointwise'][held], s['oldbits']['pointwise'][held] = choice['pointwise'], old['pointwise']
            for k in CONTROL_ARMS:
                s['bits'][k][choice['ids']] = choice[k]
                s['oldbits'][k][old['ids']] = old[k]
            s['nonzero'][choice['ids']] = choice['matched_nonzero']
            s['queries'].extend(receipt['queries'])
        del a
    roster = sorted(identity['folds'])
    def metric(model, ref, mask):
        return paired_scene_metrics(model[mask], ref[mask], data['sites'][mask], expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
            bootstrap_resamples=reg['bootstrap_resamples'], seed=reg['bootstrap_seed'])
    def describe(s, ade, fde, bits, mask):
        zero = mask & np.isfinite(cv) & (cv == 0)
        easy = metric(ade, cv, mask & s['easy'])
        return dict(ADE_vs_CV=metric(ade, cv, mask), FDE_vs_CV=metric(fde, cv_fde, mask),
            ADE_vs_training_selected_baseline=metric(ade, s['strong_ade'], mask),
            FDE_vs_training_selected_baseline=metric(fde, s['strong_fde'], mask),
            ADE_vs_fixed_damping097=metric(ade, data['baseline_ade'][:, 3], mask),
            hard_ADE_vs_CV=metric(ade, cv, mask & s['hard']), positive_easy_ADE_vs_CV=easy,
            complete_ADE_vs_CV=metric(ade, cv, mask & data['valid'].all(1)),
            zero_CV=dict(rows=int(zero.sum()), harmed_rows=int((ade[zero] > 0).sum()),
                total_added_error=float(ade[zero].sum())), switch_rate=float(bits[mask].mean()),
            indexed_rows=int(mask.sum()),
            safety_observed_pass=bool(easy['worst_scene_gain_percent'] is not None
                and easy['worst_scene_gain_percent'] >= -2 and not np.any(ade[zero] > 0)))
    summaries = {}
    allmask = np.ones(n, bool)
    for (seed, arm), s in outputs.items():
        if not s['seen'].all():
            raise ValueError('Incomplete outer predictions')
        metrics, errors, switches = {}, {}, {}
        for name, bits, ref, ref_fde in [
            ('CV', np.zeros(n, bool), cv, cv_fde),
            ('training_selected_baseline', np.zeros(n, bool), s['strong_ade'], s['strong_fde']),
            ('fixed_damping097', np.zeros(n, bool), data['baseline_ade'][:, 3], data['baseline_fde'][:, 3]),
            ('neural', np.ones(n, bool), cv, cv_fde),
            *[(k, v, cv, cv_fde) for k, v in s['bits'].items()],
            *[('previous_'+k, v, s['strong_ade'], s['strong_fde']) for k, v in s['oldbits'].items()]]:
            ade = choose_errors(bits, s['neural_ade'], ref)
            fde = choose_errors(bits, s['neural_fde'], ref_fde)
            errors[name], switches[name] = ade, bits
            metrics[name] = describe(s, ade, fde, bits, qmask)
        full = {}
        for name in ('CV', 'training_selected_baseline', 'fixed_damping097', 'neural', 'pointwise', 'previous_pointwise'):
            if name == 'training_selected_baseline':
                fde = s['strong_fde']
            elif name == 'fixed_damping097':
                fde = data['baseline_fde'][:, 3]
            else:
                fde = choose_errors(switches[name], s['neural_fde'], s['strong_fde'] if name.startswith('previous') else cv_fde)
            full[name] = describe(s, errors[name], fde, switches[name], allmask)
        comparisons = dict(
            pointwise_vs_previous=metric(errors['pointwise'], errors['previous_pointwise'], allmask),
            joint_vs_previous=metric(errors['joint'], errors['previous_joint'], qmask),
            joint_vs_independent=metric(errors['joint'], errors['independent'], qmask),
            joint_exact_vs_independent=metric(errors['joint_exact'], errors['independent'], qmask & s['nonzero']),
            joint_exact_vs_unary=metric(errors['joint_exact'], errors['unary_exact'], qmask & s['nonzero']))
        summaries[f'{seed}_{arm}'] = dict(full=full, joint_population=metrics, comparisons=comparisons,
            queries=dict(total=len(s['queries']), matched=sum(v['matched'] for v in s['queries']),
                matched_nonzero=sum(v['matched_nonzero'] for v in s['queries']),
                with_edges=sum(v['edges'] > 0 for v in s['queries']),
                solver_failures={k: sum(not v['arms'][k]['solver_optimal'] for v in s['queries']) for k in CONTROL_ARMS}))
    result = dict(result_source='fresh_run_nested_CV_reference_cost_fitting_and_control_readout', identity=identity,
        training={k: dict(lineage=r['identity']['lineage'], fit=r['fit'], wall_seconds=r['wall_seconds'],
                          artifacts=r['artifacts']) for k, r in reports.items()},
        controls=decision_receipts, seeds=summaries, source_rows=n, joint_rows=int(qmask.sum()),
        new_forecaster_training=False, forecaster_source='cached_verified', actual_cost_heads=len(reports),
        fixed_reference_index=1, positive_easy_limit_percent=2, zero_reference_allowed_added_harm=0,
        independent_reserved_readout=False, calibrated_safety=False, deployment_changed=False,
        metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
    assert_identity(identity)
    immutable_json(PUBLIC/'analysis.json', result)
    if verify:
        immutable_json(PUBLIC/'verification.json', dict(result_source='cached_verified',
            analysis_sha256=digest(PUBLIC/'analysis.json'), metrics_recomputed=True,
            new_training=False, new_forecast_inference=False, new_solver=False))
    beat('cv_reference_readout_complete', heads=len(reports), source_rows=n)


def replay(reg, data, identity, designs):
    if not (PUBLIC/'analysis.json').exists():
        raise ValueError('Completed readout required before replay')
    checks = []
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        a = assemble(data, identity, key, design, ti)
        held = design['held_ids'][:4096]
        for arm in reg['arms']:
            name = key+'_'+arm
            r = checked_head(name, identity)
            cp = torch.load(ROOT/r['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
            if cp['identity'] != r['identity']:
                raise ValueError('Checkpoint lineage changed')
            for k in ('mean', 'std', 'constant', 'weights', 'known'):
                np.testing.assert_array_equal(a['pr'][k], cp['preprocess'][k])
            if arm == 'ridge':
                fresh = predict_ridge(cp['head'], a['x'][held], a['same'][held], cp['preprocess'])
            else:
                model = build_head(reg['training']['width'], cp['preprocess'], ti['seed'])
                model.load_state_dict(cp['model'])
                fresh = predict_neural(model, a['x'][held], a['same'][held], cp['preprocess'])
            with np.load(ROOT/r['artifacts']['predicted_costs']['path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'][:4096], held)
                saved = z['costs'][:4096].copy()
            fresh = np.maximum(fresh, 0)
            np.testing.assert_allclose(fresh, saved, atol=1e-5, rtol=1e-6)
            checks.append(dict(trial=name, rows=len(held), exact=bool(np.array_equal(fresh, saved)),
                max_difference=float(np.abs(fresh-saved).max()), ids_sha256=array_hash(held),
                checkpoint_sha256=r['artifacts']['checkpoint']['sha256']))
        del a
    assert_identity(identity)
    immutable_json(PUBLIC/'checkpoint_replay.json', dict(result_source='fresh_run_checkpoint_inference',
        analysis_sha256=digest(PUBLIC/'analysis.json'), checks=checks, all_passed=True,
        new_training=False, reserved_roles_opened=False))
    beat('checkpoint_replay_complete', checks=len(checks), max_difference=max(v['max_difference'] for v in checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepare', 'pilot', 'train', 'resume', 'evaluate', 'verify', 'replay'):
        parser.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    if not any((args.prepare, args.pilot, args.train, args.evaluate, args.verify, args.replay)):
        parser.error('Explicit phase required')
    if args.pilot and (args.train or args.evaluate or args.verify or args.replay):
        parser.error('Pilot is a separate resume-within-budget step')
    for path in (PRIVATE, PUBLIC):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlinked experiment destination')
        path.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        reg, data, identity, designs, qmask = load()
        beat('verified_CV_reference_source', rows=len(data['sites']), joint_rows=int(qmask.sum()),
             architecture=platform.machine(), threads=torch.get_num_threads(), workers=0)
        if args.train or args.pilot:
            train(reg, data, identity, designs, resume=args.resume, pilot=args.pilot)
        if args.evaluate or args.verify:
            evaluate(reg, data, identity, designs, qmask, args.verify)
        if args.replay:
            replay(reg, data, identity, designs)
        beat('phase_complete')


if __name__ == '__main__':
    main()
