"""Fit source-excluded score heads, calibrate, then read the frozen outer folds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_nested_producers as bank
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from src.world_model.m3w_european_source_forecast import baseline_numpy, fit_design
from src.world_model.m3w_european_source_intervention import causal_cost_features, paired_cost_labels
from src.world_model.m3w_native_gain_harm import preprocess, fit_neural, predict_neural, build_head
from src.world_model.m3w_european_conditional_risk import event_labels, nonnegative_moments, fitting_support
from src.evaluation.m3w_source_risk_calibration import fit_calibration, apply_calibration
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics

PRIVATE, PUBLIC = bank.PRIVATE, bank.PUBLIC
FILES = ('scripts/run_m3w_european_nested_calibration.py', 'src/evaluation/m3w_source_risk_calibration.py',
    'tests/test_m3w_source_risk_calibration.py', 'src/world_model/m3w_native_gain_harm.py',
    'src/world_model/m3w_european_source_intervention.py', 'src/world_model/m3w_european_conditional_risk.py',
    'src/world_model/m3w_supervised_intervention.py', 'src/evaluation/m3w_native_metrics.py')


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    json_write(PRIVATE/'head_heartbeat.json', row)
    with (PRIVATE/'head_events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def assert_identity(identity):
    bank.assert_identity(identity['producer_identity'])
    for path, sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Registered calibration code changed: '+path)


def load():
    reg, preg, data, pid = bank.load()
    receipt = json.loads((PUBLIC/'producer_replay.json').read_text())
    if receipt['identity'] != pid or not receipt['all_passed'] or len(receipt['checks']) != 18:
        raise ValueError('Complete verified inner producer bank required')
    for row in receipt['checks']:
        for field in ('checkpoint', 'prediction'):
            a = row[field]
            if digest(ROOT/a['path']) != a['sha256']:
                raise ValueError('Changed inner-producer artifact')
        if not row['exact'] or row['replay_rows'] <= 0:
            raise ValueError('Exact sampled producer replay required')
    identity = dict(bindings={p: digest(ROOT/p) for p in FILES}, producer_identity=pid,
        producer_replay_sha256=digest(PUBLIC/'producer_replay.json'))
    immutable_json(PRIVATE/'head_identity.json', identity)
    immutable_json(PUBLIC/'head_matrix.json', dict(identity=identity, heads=54, updates=108000,
        calibration_maps=72, policy_views=72, pointwise_only=True, reserved_roles_opened=False))
    return reg, preg, data, identity


def jobs(reg, data, identity):
    folds = identity['producer_identity']['parent_identity']['folds']
    for fold in range(3):
        sites = sorted(s for s, f in folds.items() if f == fold)
        design = fit_design(data, sites, data['baseline_ade'])
        for seed in reg['seeds']:
            for candidate in reg['candidates']:
                yield f'{candidate}_fold{fold}_seed{seed}', candidate, fold, seed, design


def read_predictions(artifact, ids):
    path = ROOT/artifact['path']
    if digest(path) != artifact['sha256']:
        raise ValueError('Forecast artifact changed')
    with np.load(path, allow_pickle=False) as z:
        pos = np.searchsorted(z['ids'], ids)
        np.testing.assert_array_equal(z['ids'][pos], ids)
        return z['prediction'][pos].copy()


def assemble(candidate, fold, seed, design, data, identity):
    n = len(data['sites'])
    fit, held = design['train_ids'], design['held_ids']
    pid = identity['producer_identity']
    b = baseline_numpy(data['history'], 1)-data['origin'][:, None]
    lineage = dict(candidate=candidate, fit_fold=fold, seed=seed, training_sites=sorted(set(data['sites'][fit])),
        excluded_sites=sorted(set(data['sites'][held])), train_ids_sha256=array_hash(fit),
        held_ids_sha256=array_hash(held), baseline_index=1, easy_cut=design['easy_cut'], hard_cut=design['hard_cut'])
    if candidate == 'damping097':
        p = baseline_numpy(data['history'], 3)-data['origin'][:, None]
        lineage['producer'] = 'fixed_causal_damping097'
    else:
        p = np.empty((n, 12, 2), np.float32)
        seen = np.zeros(n, bool)
        producers = []
        for side in range(2):
            key = f'fold{fold}_half{side}_seed{seed}'
            path = PRIVATE/'producers'/key/'opposite_half.json'
            r = json.loads(path.read_text())
            ids = np.flatnonzero(np.isin(data['sites'], pid['halves'][str(fold)][1-side]))
            if set(r['identity']['training_sites']) & set(data['sites'][ids]):
                raise ValueError('OOF producer exposed to target source')
            if set(r['identity']['training_sites']) & set(data['sites'][held]):
                raise ValueError('OOF producer exposed to calibration/readout')
            p[ids] = read_predictions(r, ids)
            seen[ids] = True
            producers.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path)))
        final = pid['frozen_final_producers'][f'single{fold}_seed{seed}']
        if set(final['training_sites']) != set(data['sites'][fit]):
            raise ValueError('Final producer fitting roster differs from score head')
        p[held] = read_predictions(final['prediction'], held)
        seen[held] = True
        if not seen.all():
            raise ValueError('Incomplete causal candidate population')
        lineage['inner_producers'], lineage['final_producer'] = producers, final
    x, scale = causal_cost_features(data['geometry'], b, p)
    same = np.all(b == p, axis=(1, 2))
    ade, _ = native_errors(p[fit].astype(float)+data['origin'][fit, None], data['target_eval'][fit],
        data['valid'][fit], np.ones(len(fit)))
    cv = np.asarray(data['baseline_ade'][fit, 1])
    y = paired_cost_labels(cv, ade)
    lineage.update(feature_train_sha256=array_hash(x[fit]), label_train_sha256=array_hash(y),
        fitting_support=fitting_support(cv, data['sites'][fit]))
    return dict(x=x, p=p, b=b, same=same, y=y, lineage=lineage)


def task_data(a, design, data, task):
    fit = design['train_ids']
    cv = np.asarray(data['baseline_ade'][fit, 1])
    y = a['y'] if task == 'utility' else event_labels(cv, a['y'][:, 1], easy_cut=design['easy_cut'], event=task)
    pr = preprocess(a['x'][fit], y, cv, data['sites'][fit], str(data['sites'][design['held_ids'][0]]))
    return y, pr


def checked(name, task, identity, steps):
    r = json.loads((PRIVATE/'heads'/(name+'_'+task)/'complete.json').read_text())
    if r['identity']['identity'] != identity or not r['fit']['complete'] or r['fit']['step'] != steps:
        raise ValueError('Incomplete or changed score head')
    for artifact in r['artifacts'].values():
        if digest(ROOT/artifact['path']) != artifact['sha256']:
            raise ValueError('Score artifact changed')
    return r


def scores(r, ids):
    with np.load(ROOT/r['artifacts']['scores']['path'], allow_pickle=False) as z:
        pos = np.searchsorted(z['ids'], ids)
        np.testing.assert_array_equal(ids, z['ids'][pos])
        return z['scores'][pos].copy()


def train(reg, data, identity, resume, pilot):
    for name, candidate, fold, seed, design in jobs(reg, data, identity):
        a = assemble(candidate, fold, seed, design, data, identity)
        fit, held = design['train_ids'], design['held_ids']
        for task in ('utility', *reg['events']):
            directory = PRIVATE/'heads'/(name+'_'+task)
            y, pr = task_data(a, design, data, task)
            hid = dict(identity=identity, lineage=a['lineage'], task=task, target_sha256=array_hash(y), cost_scale=pr['cost_scale'])
            if (directory/'complete.json').exists():
                r = checked(name, task, identity, reg['head_training']['steps'])
                if r['identity'] != hid:
                    raise ValueError('Score-head lineage changed')
                continue
            same = a['same'][fit] if task == 'utility' else np.zeros(len(fit), bool)
            beat('score_training', trial=name, task=task)
            model, report = fit_neural(a['x'][fit], y, data['sites'][fit], same, pr, seed=seed,
                arm='mse', settings=reg['head_training'], identity=hid, directory=directory, resume=resume,
                stop_at=100 if pilot else None, heartbeat=lambda **kw: beat(trial=name, task=task, **kw))
            if pilot:
                immutable_json(PRIVATE/'head_pilot.json', dict(identity=hid, fit=report, checkpoint=bank.artifact(directory/'checkpoint.pt')))
                return
            raw = predict_neural(model, a['x'][held], a['same'][held] if task == 'utility' else np.zeros(len(held), bool), pr)
            pred = np.maximum(raw, 0) if task == 'utility' else nonnegative_moments(raw, a['same'][held])
            path = directory/'scores.npz'
            tmp = path.with_suffix('.tmp.npz')
            np.savez(tmp, ids=held, scores=pred)
            os.replace(tmp, path)
            immutable_json(directory/'complete.json', dict(identity=hid, fit=report, artifacts=dict(
                checkpoint=bank.artifact(directory/'checkpoint.pt'), scores=bank.artifact(path))))
            assert_identity(identity)
            beat('score_complete', trial=name, task=task, updates=report['step'])


def all_heads(reg, data, identity):
    return {name+'_'+task: checked(name, task, identity, reg['head_training']['steps'])
        for name, _, _, _, _ in jobs(reg, data, identity) for task in ('utility', *reg['events'])}


def calibrate(reg, data, identity):
    heads = all_heads(reg, data, identity)
    receipts = []
    for name, candidate, fold, seed, design in jobs(reg, data, identity):
        a = assemble(candidate, fold, seed, design, data, identity)
        for cal in range(3):
            if cal == fold:
                continue
            roles = bank.role_sets(identity['producer_identity']['parent_identity']['folds'], fold, cal)
            ids = np.flatnonzero(np.isin(data['sites'], roles['calibration']))
            ade, _ = native_errors(a['p'][ids].astype(float)+data['origin'][ids, None],
                data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            utility_head = heads[name+'_utility']
            costs = scores(utility_head, ids)
            u = (costs[:, 0]-costs[:, 1])/utility_head['identity']['cost_scale']
            moving = np.linalg.norm(np.diff(data['history'][ids], axis=1), axis=2).sum(1) > 0
            for event in reg['events']:
                moment_head = heads[name+'_'+event]
                m = scores(moment_head, ids)
                rule = fit_calibration(u, m, moving, np.asarray(data['baseline_ade'][ids, 1]), ade,
                    np.asarray(data['sites'][ids]), easy_cut=design['easy_cut'], event=event,
                    grid=reg['calibration_grid'], budget=reg['predicted_risk_budget'])
                path = PRIVATE/'calibration'/(name+'_'+event+f'_cal{cal}.json')
                immutable_json(path, dict(identity=identity, roles=roles, candidate=candidate, seed=seed,
                    event=event, calibration_ids_sha256=array_hash(ids), rule=rule,
                    utility=utility_head['artifacts'], risk=moment_head['artifacts'], result_source='fresh_run'))
                receipts.append(bank.artifact(path))
                beat('calibration_frozen', trial=name, calibration_fold=cal, event=event,
                    selected_rule=rule['rules']['selected_risk_grid'])
    immutable_json(PUBLIC/'calibration_receipts.json', dict(identity=identity, receipts=receipts,
        count=len(receipts), outer_metrics_read=False, formal_guarantee=False))


def evaluate(reg, data, identity, verify):
    heads = all_heads(reg, data, identity)
    manifest = json.loads((PUBLIC/'calibration_receipts.json').read_text())
    if manifest['identity'] != identity or manifest['count'] != 72:
        raise ValueError('All calibration maps must freeze before outer readout')
    for item in manifest['receipts']:
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Calibration map changed')
    n = len(data['sites'])
    cv, cvf = np.asarray(data['baseline_ade'][:, 1]), np.asarray(data['baseline_fde'][:, 1])
    populations, receipts = {}, []
    for name, candidate, fold, seed, design in jobs(reg, data, identity):
        a = assemble(candidate, fold, seed, design, data, identity)
        for cal in range(3):
            if cal == fold:
                continue
            roles = bank.role_sets(identity['producer_identity']['parent_identity']['folds'], fold, cal)
            ids = np.flatnonzero(np.isin(data['sites'], roles['readout']))
            direction = (cal-fold) % 3
            ade, fde = native_errors(a['p'][ids].astype(float)+data['origin'][ids, None],
                data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            uh = heads[name+'_utility']
            cost = scores(uh, ids)
            u = (cost[:, 0]-cost[:, 1])/uh['identity']['cost_scale']
            moving = np.linalg.norm(np.diff(data['history'][ids], axis=1), axis=2).sum(1) > 0
            for event in reg['events']:
                path = PRIVATE/'calibration'/(name+'_'+event+f'_cal{cal}.json')
                calibration = json.loads(path.read_text())
                if calibration['identity'] != identity or calibration['roles'] != roles:
                    raise ValueError('Mismatched calibration lineage')
                m = scores(heads[name+'_'+event], ids)
                for rule, config in calibration['rule']['rules'].items():
                    bits = apply_calibration(u, m, moving, config, reg['predicted_risk_budget'])
                    receipt_path = PRIVATE/'decisions'/(name+'_'+event+f'_cal{cal}_'+rule+'.npz')
                    if receipt_path.exists():
                        r = json.loads(receipt_path.with_suffix('.json').read_text())
                        if digest(receipt_path) != r['sha256'] or r['identity'] != identity or r['calibration'] != bank.artifact(path):
                            raise ValueError('Changed frozen outer decisions')
                        with np.load(receipt_path, allow_pickle=False) as z:
                            np.testing.assert_array_equal(ids, z['ids'])
                            np.testing.assert_array_equal(bits, z['switch'])
                    else:
                        if verify:
                            raise ValueError('Cannot verify absent decisions')
                        receipt_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp = receipt_path.with_suffix('.tmp.npz')
                        np.savez(tmp, ids=ids, switch=bits)
                        os.replace(tmp, receipt_path)
                        immutable_json(receipt_path.with_suffix('.json'), dict(identity=identity,
                            calibration=bank.artifact(path), **bank.artifact(receipt_path), uses_future_input=False))
                    receipts.append(bank.artifact(receipt_path.with_suffix('.json')))
                    key = f'{seed}_direction{direction}_{candidate}_{event}_{rule}'
                    p = populations.setdefault(key, dict(ade=np.full(n, np.nan), fde=np.full(n, np.nan),
                        raw_ade=np.full(n, np.nan), raw_fde=np.full(n, np.nan), bits=np.zeros(n, bool),
                        easy=np.zeros(n, bool), hard=np.zeros(n, bool), seen=np.zeros(n, bool)))
                    if p['seen'][ids].any():
                        raise ValueError('Repeated outer population')
                    p['seen'][ids] = True
                    p['bits'][ids] = bits
                    p['ade'][ids] = np.where(bits, ade, cv[ids])
                    p['fde'][ids] = np.where(bits, fde, cvf[ids])
                    p['raw_ade'][ids], p['raw_fde'][ids] = ade, fde
                    p['easy'][ids] = (cv[ids] > 0) & (cv[ids] <= design['easy_cut'])
                    p['hard'][ids] = cv[ids] >= design['hard_cut']
            beat('outer_readout', trial=name, calibration_fold=cal, rows=len(ids), verify=verify)
    full = np.ones(n, bool)
    roster = sorted(identity['producer_identity']['parent_identity']['folds'])
    def metric(model, reference, mask):
        return paired_scene_metrics(model[mask], reference[mask], data['sites'][mask], expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
            bootstrap_resamples=reg['bootstrap_resamples'], seed=reg['bootstrap_seed'])
    summaries, contrasts, calibration_lift = {}, {}, {}
    for key, p in populations.items():
        if not p['seen'].all():
            raise ValueError('Missing complete outer population')
        easy = metric(p['ade'], cv, p['easy'])
        zero = np.isfinite(cv) & (cv == 0)
        summaries[key] = dict(ADE_vs_CV=metric(p['ade'], cv, full), FDE_vs_CV=metric(p['fde'], cvf, full),
            hard_ADE_vs_CV=metric(p['ade'], cv, p['hard']), positive_easy_ADE_vs_CV=easy,
            complete_ADE_vs_CV=metric(p['ade'], cv, data['valid'].all(1)),
            raw_candidate_ADE_vs_CV=metric(p['raw_ade'], cv, full),
            positive_harm_sum=float(np.maximum(p['ade'][np.isfinite(cv)]-cv[np.isfinite(cv)], 0).sum()),
            zero_CV=dict(rows=int(zero.sum()), harmed_rows=int((p['ade'][zero] > 0).sum()), total_added_error=float(p['ade'][zero].sum())),
            switch_rate=float(p['bits'].mean()), indexed_rows=n,
            safety_observed_pass=bool(easy['worst_scene_gain_percent'] is not None and easy['worst_scene_gain_percent'] >= -2
                and not np.any(p['ade'][zero] > 0)), calibrated_safety=False)
        tokens = key.split('_', 4)
        seed, direction, candidate, event, rule = tokens
        raw = populations['_'.join((seed, direction, candidate, event, 'none'))]
        calibration_lift[key] = dict(all=metric(p['ade'], raw['ade'], full), hard=metric(p['ade'], raw['ade'], p['hard']))
        if candidate == 'neural':
            other = populations['_'.join((seed, direction, 'damping097', event, rule))]
            contrasts[key] = dict(all=metric(p['ade'], other['ade'], full), hard=metric(p['ade'], other['ade'], p['hard']),
                easy=metric(p['ade'], other['ade'], p['easy']))
    result = dict(identity=identity, result_source='fresh_run_nested_head_fit_source_calibration_and_outer_development_readout',
        policies=summaries, neural_vs_damping=contrasts, calibration_vs_none=calibration_lift,
        training=heads, decisions=receipts, calibration=manifest, source_rows=n,
        new_score_heads=54, new_score_updates=108000, new_inner_predictors=18, new_forecast_updates=72000,
        independent_reserved_readout=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    assert_identity(identity)
    immutable_json(PUBLIC/'analysis.json', result)
    if verify:
        immutable_json(PUBLIC/'verification.json', dict(analysis_sha256=digest(PUBLIC/'analysis.json'),
            metrics_recomputed=True, decisions_exact=216, result_source='cached_verified', all_passed=True))
    beat('nested_readout_complete', policies=len(summaries), verify=verify)


def replay(reg, data, identity):
    rows = []
    samplers = {}
    all_heads(reg, data, identity)
    for name, candidate, fold, seed, design in jobs(reg, data, identity):
        a = assemble(candidate, fold, seed, design, data, identity)
        fit, ids = design['train_ids'], design['held_ids'][:4096]
        for task in ('utility', *reg['events']):
            r = checked(name, task, identity, reg['head_training']['steps'])
            y, pr = task_data(a, design, data, task)
            state = torch.load(ROOT/r['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
            if state['identity'] != r['identity'] or int(state['draws'].sum()) != 512000 or state['draws'][~pr['known']].any():
                raise ValueError('Invalid score-head checkpoint exposure')
            for field in ('mean', 'std', 'weights', 'known', 'constant'):
                np.testing.assert_array_equal(pr[field], state['preprocess'][field])
            group = (fold, seed)
            if group not in samplers:
                samplers[group] = dict(draws=state['draws'].copy(), rng=state['sampler_rng'].clone(), heads=0)
            np.testing.assert_array_equal(state['draws'], samplers[group]['draws'])
            if not torch.equal(state['sampler_rng'], samplers[group]['rng']):
                raise ValueError('Candidate/event fitting exposure differs')
            samplers[group]['heads'] += 1
            model = build_head(reg['head_training']['width'], pr, seed)
            model.load_state_dict(state['model'])
            raw = predict_neural(model, a['x'][ids], a['same'][ids] if task == 'utility' else np.zeros(len(ids), bool), pr)
            fresh = np.maximum(raw, 0) if task == 'utility' else nonnegative_moments(raw, a['same'][ids])
            np.testing.assert_array_equal(fresh, scores(r, ids))
            rows.append(dict(head=name+'_'+task, rows=len(ids), exact=True, supervised_draws=512000, unknown_draws=0))
    if len(samplers) != 9 or any(s['heads'] != 6 for s in samplers.values()):
        raise ValueError('Every candidate/task exposure comparison required')
    immutable_json(PUBLIC/'checkpoint_replay.json', dict(analysis_sha256=digest(PUBLIC/'analysis.json'),
        checks=rows, matched_sampler_groups=9, heads_per_sampler_group=6,
        all_passed=True, result_source='fresh_run_checkpoint_replay'))
    beat('head_replay_complete', heads=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare', 'pilot', 'train', 'resume', 'calibrate', 'evaluate', 'verify', 'replay'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    if not any((args.prepare, args.pilot, args.train, args.calibrate, args.evaluate, args.verify, args.replay)):
        parser.error('Explicit phase required')
    if args.pilot and any((args.train, args.calibrate, args.evaluate, args.verify, args.replay)):
        parser.error('Pilot is a separate resumable phase')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'heads.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        reg, _, data, identity = load()
        beat('inputs_verified', rows=len(data['sites']), architecture=platform.machine(), threads=4, workers=0)
        if args.train or args.pilot:
            train(reg, data, identity, args.resume, args.pilot)
        if args.calibrate:
            calibrate(reg, data, identity)
        if args.evaluate or args.verify:
            evaluate(reg, data, identity, args.verify)
        if args.replay:
            replay(reg, data, identity)
        beat('phase_complete')


if __name__ == '__main__':
    main()
