"""Matched neural kinematic-basis repair; original SOURCE and risk rules remain."""
import argparse
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_readout_refit as parent
from src.world_model import m3w_kinematic_risk as api

source, old, direct = parent.source, parent.old, parent.parent
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_kinematic_risk_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_kinematic_risk_v1'
CONFIG = ROOT/'configs/m3w_moving_kinematic_risk_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_kinematic_risk.py',
    ROOT/'tests/test_m3w_kinematic_risk.py')
POLICIES = parent.POLICIES + tuple(a+s for a in api.ARMS for s in ('_veto', '_matched'))


def register(beat):
    cfg = source.read(CONFIG)
    _, original, data, jobs, refs, parts, points, routers, critics, fitted = parent.register(beat)
    if (cfg['arms'] != list(api.ARMS) or cfg['new_heads'] != 6 or cfg['primary_updates'] != 12000
            or cfg['feature_dimensions'] != len(api.NAMES) or cfg['training'] != source.read(direct.CONFIG)['training']
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed matched model repair only; original scientific contract unchanged')
    previous = ROOT/'outputs/publication_readiness_2026_09/moving_nonlinear_readout_v1'
    if source.read(previous/'verification.json')['analysis_sha256'] != source.sha(previous/'analysis.json'):
        raise ValueError('Current negative result must remain verified')
    refits = {source.read(source.checked(r))['job']: source.read(source.checked(r))
        for r in source.read(parent.PUBLIC/'training_freeze.json')['fits']}
    source.once(REG, source.stable(dict(configuration=cfg, jobs=jobs,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(parent.REG), warm_training=source.artifact(direct.PUBLIC/'training_freeze.json'),
        negative_readout=source.artifact(previous/'analysis.json'),
        feature_provenance=[source.artifact(ROOT/p) for p in
            ('src/world_model/m3w_supervised_intervention.py', 'src/world_model/m3w_native_gain_harm.py',
             'src/world_model/m3w_floor_relative.py', 'src/data_unification/m3w_moving_forecast_data.py')],
        schema_names=api.NAMES, exposed_SOURCE_only=True, proposer_part=1, fitting_part=0, validation_part=2,
        original_rules_unchanged=True, no_new_H=True, original24group_primary='not_estimable_not_passed',
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)))
    return cfg, original, data, jobs, refs, parts, points, routers, critics, fitted, refits


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def fitting(data, job, refs, parts, points, routers, fitted):
    p, _ = direct.parent.training_packet(data, job, refs, parts[job['key']][0], 1, points, routers)
    warm = api.base.read_checkpoint(source.checked(fitted[job['key'], 'mean']['checkpoint']))
    if warm['step'] != 2000 or not all(int(s['step']) == 2000 for s in warm['optimizer']['state'].values()):
        raise ValueError('Completed real Torch mean critic required')
    prepared = api.prepare(p['learning'], warm)
    return prepared, warm


def one(data, warm, job, arm, cfg, fitted, beat, guard, kind='fits', resume=False, stop_at=None):
    return api.fit(data, warm, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job,
            warm_checkpoint=fitted[job['key'], 'mean']['checkpoint']),
        path=path_for(job, arm, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v))


def pilot(cfg, original, data, jobs, refs, parts, points, routers, critics, fitted, refits, beat, guard, resume):
    job = jobs[0]; d, warm = fitting(data, job, refs, parts, points, routers, fitted)
    model = api.initialize(warm); group = d['causal']['groups'][0]; seg = api.torch.zeros(len(group), dtype=api.torch.long)
    prior = api.parent.DirectRiskHead(428, cfg['training']['width'], job['seed']); prior.load_state_dict(warm['model'])
    for arm in api.ARMS:
        with api.torch.no_grad():
            api.base.exact(model(api.batch_features(d, arm)[group], seg, 1), prior(d['z'][group], seg, 1))
    rows = []
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume'); step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(d, warm, job, arm, cfg, fitted, beat, guard, 'pilot_resume', resume and path.exists(), 50)
        a = one(d, warm, job, arm, cfg, fitted, beat, guard, 'pilot_resume', True, 100)
        b = one(d, warm, job, arm, cfg, fitted, beat, guard, 'pilot_direct',
            resume and path_for(job, arm, 'pilot_direct').exists(), 100)
        api.parent.parent.router.exact_state(a, b)
        source.np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
        rows.append(dict(arm=arm, steps=100, seconds=b['seconds'], parameters=b['parameters'], exact_resume=True))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, actual_updates_including_repeats=400,
        both_resume_and_prediction_exact=True, initial_warm_predictions_exact=True, no_validation_used=True))


def train(cfg, original, data, jobs, refs, parts, points, routers, critics, fitted, refits, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['both_resume_and_prediction_exact']:
        raise ValueError('Real native Torch resume pilot first')
    receipts = []
    for job in jobs:
        d, warm = fitting(data, job, refs, parts, points, routers, fitted); states = []
        for arm in api.ARMS:
            state = one(d, warm, job, arm, cfg, fitted, beat, guard,
                resume=resume and path_for(job, arm).exists())
            if state['step'] != 2000 or state['budget'] != 2000 or state['parameters'] != 32130:
                raise ValueError('Full registered neural update budget required')
            for opt in state['optimizer']['state'].values():
                if int(opt['step']) != 2000:
                    raise ValueError('Every trained parameter needs actual optimizer updates')
            if api.torch.equal(state['model']['encoder.weight'], warm['model']['encoder.weight']):
                raise ValueError('Actual neural encoder updates required')
            if bool(state['model']['kinematics.weight'].any()) != (arm == 'kinematic'):
                raise ValueError('Only the unmasked feature branch should learn')
            for key in ('sampler', 'draw_hash', 'preprocess', 'fit_ids', 'source_sites'):
                api.base.exact(state[key], warm[key])
            if not receipts:
                repeat = one(d, warm, job, arm, cfg, fitted, beat, guard, 'full_refit',
                    resume and path_for(job, arm, 'full_refit').exists())
                api.parent.parent.router.exact_state(state, repeat)
                source.once(PUBLIC/'first_full_refit.json', dict(exact=True, checkpoint=source.artifact(path_for(job, arm, 'full_refit'))))
            rowpath = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(rowpath, source.stable(dict(job=job['key'], arm=arm, checkpoint=source.artifact(path_for(job, arm)),
                steps=state['step'], parameters=state['parameters'], seconds=state['seconds'], trace=state['trace'],
                common_input_hash=state['common_input_hash'], extra_hash=state['extra_hash'],
                extra_raw_hash=state['extra_raw_hash'], target_hash=state['target_hash'], proposal_hash=state['proposal_hash'],
                warm_checkpoint=fitted[job['key'], 'mean']['checkpoint'], observed_draws=state['observed_supervised_draws'],
                unknown_bound_draws=state['unknown_bound_supervised_draws'], source_sites=state['source_sites'],
                features_available=True, validation_readout=False, neural_encoder_updated=True)))
            receipts.append(source.artifact(rowpath)); states.append(state)
            beat(state='kinematic_risk_fit_frozen', complete=len(receipts), total=6)
        api.assert_matched(*states)
        del d, warm, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=receipts, complete_heads=6, primary_updates=12000,
        additional_full_refit_updates=2000, all_before_validation=True, original_sampler_and_target_unchanged=True))


def assert_excluded(state, causal, names):
    api.parent.parent.assert_excluded(state, causal, names)
    if set(state['extra_preprocess']['source_names']) != set(state['source_sites']) or set(names) & set(state['extra_preprocess']['source_names']):
        raise ValueError('Extra preprocessing must be own-fitting-only and source-excluded')


def macro(rows, getter):
    by_source = {}
    for s in sorted({r['site'] for r in rows}):
        values = [getter(r) for r in rows if r['site'] == s]
        by_source[s] = float(source.np.mean(values)) if all(v is not None for v in values) else None
    return dict(by_source=by_source, mean=float(source.np.mean(list(by_source.values())))
        if by_source and all(v is not None for v in by_source.values()) else None, CI95=None, descriptive_pilot=True)


def evaluate(cfg, original, data, jobs, refs, parts, points, routers, critics, fitted, refits, beat, guard, *, verify):
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 6 or not freeze['all_before_validation']:
        raise ValueError('Freeze all six registered neural heads before any held readout')
    trained = {}
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); trained[r['job'], r['arm']] = r
    held_rows, fitting_rows, receipts = [], [], []
    for job in jobs:
        for part_id in (0, 2):
            part = parts[job['key']][part_id]
            p = old.packet(data, job, refs, part, training=False); c, ids = p['causal'], p['ids']
            choices, predictions, hashes = direct.decisions(p, job, part, points, routers, critics, fitted, part_id == 2)
            affine = api.base.read_checkpoint(source.checked(refits[job['key']]['checkpoint']))
            if part_id == 2:
                api.parent.parent.assert_excluded(affine, c, part['names'])
            predictions['readout'] = parent.api.predict(affine, c)
            hashes['readout'] = source.array_hash(predictions['readout'])
            choices['readout_veto'], choices['readout_matched'] = api.parent.choose(predictions['router'], predictions['all_action'], predictions['readout'], c)
            for arm in api.ARMS:
                state = api.base.read_checkpoint(source.checked(trained[job['key'], arm]['checkpoint']))
                if part_id == 2:
                    assert_excluded(state, c, part['names'])
                predictions[arm] = api.predict(state, c); hashes[arm] = source.array_hash(predictions[arm])
                choices[arm+'_veto'], choices[arm+'_matched'] = api.parent.choose(predictions['router'], predictions['all_action'], predictions[arm], c)
            target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
            truth = old.amplitude.targets(p, target, valid, job['easy_cut'])
            paths = {name: old.rollout(p['paths'], p['floor'], chosen) for name, chosen in choices.items()}
            rows = []
            for site in sorted(set(p['sites'])):
                at = p['sites'] == site
                policies = {name: source.readout.policy(v[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                    p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, v in paths.items()}
                local_truth = dict(moments=truth['moments'][at], observed_future=truth['observed_future'][at])
                contrasts = {name: source.readout.paired(paths[left][at], paths[right][at], p['floor'][at], target[at], valid[at])
                    for name, left, right in (('kinematic_minus_masked', 'masked_veto', 'kinematic_veto'),
                        ('kinematic_minus_affine', 'readout_veto', 'kinematic_veto'), ('masked_minus_mean', 'mean_veto', 'masked_veto'))}
                q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                matched = {arm: source.matched_utility(paths[arm+'_matched'][at], paths[arm+'_veto'][at], p['floor'][at],
                    target[at], valid[at], choices[arm+'_matched'][at], choices[arm+'_veto'][at], q) for arm in api.ARMS}
                reliability = {arm: direct.reliability(predictions[arm][at], local_truth,
                    dict(proposals=choices['public_router'][at], admitted=choices[arm+'_veto'][at])) for arm in ('mean', 'readout', *api.ARMS)}
                flow = {arm: parent.flow(local_truth, choices['public_router'][at], choices['readout_veto'][at], choices[arm+'_veto'][at]) for arm in api.ARMS}
                rows.append(dict(job=job['key'], site=str(site), policies=policies, contrasts=contrasts,
                    matched=matched, reliability=reliability, action_flow=flow))
            if part_id == 0:
                fitting_rows.extend(rows)
            else:
                key = job['key']+'_val2_propose1'; previous = source.read(parent.PUBLIC/'groups'/(key+'.json'))
                for name, h in previous['prediction_hashes'].items():
                    api.base.exact(h, hashes[name])
                for name, h in previous['choice_hashes'].items():
                    api.base.exact(h, source.array_hash(choices[name]))
                for row in rows:
                    prev = next(r for r in previous['rows'] if r['site'] == row['site'])
                    for name in parent.POLICIES:
                        api.base.exact(row['policies'][name], prev['policies'][name])
                cache = PRIVATE/'choices'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(ids=ids, **{a+'_prediction': predictions[a] for a in api.ARMS}, **choices)
                if cache.exists():
                    with source.np.load(cache, allow_pickle=False) as saved:
                        for name, value in payload.items():
                            source.np.testing.assert_array_equal(saved[name], value)
                elif verify:
                    raise ValueError('Original prediction cache required')
                else:
                    source.np.savez_compressed(cache, **payload)
                out = PUBLIC/'groups'/(key+'.json')
                source.once(out, source.stable(dict(rows=rows, prediction_hashes=hashes,
                    choice_hashes={k: source.array_hash(v) for k, v in choices.items()}, cache=source.artifact(cache),
                    old_controls_exact=True)))
                receipts.append(source.artifact(out)); held_rows.extend(rows)
                beat(state='kinematic_risk_readout', complete=len(receipts), total=3)
            del p, predictions, paths, truth; source.gc.collect(); guard()
    source.once(PUBLIC/'fitting_readout.json', source.stable(fitting_rows))
    policy = lambda rows: {name: dict(known=macro(rows, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
        hard=macro(rows, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
        switch_percent=macro(rows, lambda r, n=name: 100*r['policies'][n]['switch_rate']), safety=direct.safety(rows, name)) for name in POLICIES}
    contrasts = {scope: {name: {k: macro(rows, lambda r, n=name, k=k: r['contrasts'][n][k])
        for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for name in rows[0]['contrasts']}
        for scope, rows in (('fit', fitting_rows), ('held', held_rows))}
    contexts = []
    for job in jobs:
        rows = [r for r in held_rows if r['job'] == job['key']]
        masked, treated = direct.safety(rows, 'masked_veto'), direct.safety(rows, 'kinematic_veto')
        lower = macro(rows, lambda r: r['contrasts']['kinematic_minus_masked']['completion_lower_pp'])['mean']
        active = sum(r['policies']['kinematic_veto']['switch_rate'] for r in rows) > 0
        expanded = lower is not None and lower > 0 and active and all(treated[k] <= masked[k]
            for k in ('known_all_violations', 'known_easy_violations'))
        contexts.append(dict(job=job['key'], completion_lower_pp=lower, masked_safety=masked,
            kinematic_safety=treated, expansion_condition=bool(expanded)))
    gain = contrasts['held']['kinematic_minus_affine']['completion_lower_pp']['mean']
    result = dict(held_policy=policy(held_rows), fitting_policy=policy(fitting_rows), contrasts=contrasts,
        per_context=contexts, group_receipts=receipts, sources=len({r['site'] for r in held_rows}), source_views=len(held_rows),
        expansion_supported=all(c['expansion_condition'] for c in contexts) and gain is not None and gain > 0,
        unavailable_matched_source_views={a: sum(r['matched'][a]['contrast']['known_pp'] is None for r in held_rows) for a in api.ARMS},
        original24group_primary='not_estimable_not_passed', no_new_H=True, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'analysis.json', source.stable(result))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all3_predictions_choices_metrics_exact=True,
            all_parent_controls_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        values = register(beat)
        if args.phase == 'register':
            return
        for p in (*FILES, REG):
            content = source.subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(content).hexdigest() != source.sha(p):
                raise ValueError('Commit the exact model comparison before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase remains immutable')
        if args.phase in ('pilot', 'train'):
            globals()[args.phase](*values, beat, guard, args.resume)
        else:
            evaluate(*values, beat, guard, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_torch=True, no_new_H=True, new_encoder_training=args.phase in ('pilot', 'train')))
        beat(state='complete')


if __name__ == '__main__':
    main()
