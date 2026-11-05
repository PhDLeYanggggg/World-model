"""Matched score-only / latent-conditional risk readout with original rules."""
import argparse
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_readout_refit as parent
from src.world_model import m3w_nonlinear_risk_readout as api

source, old, direct = parent.source, parent.old, parent.parent
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_nonlinear_readout_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_nonlinear_readout_v1'
CONFIG = ROOT/'configs/m3w_moving_nonlinear_readout_v1.json'
REG = PUBLIC/'registration.json'
AMENDMENT = PUBLIC/'evaluation_interface_amendment.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
    ROOT/'src/world_model/m3w_nonlinear_risk_readout.py', ROOT/'tests/test_m3w_nonlinear_risk_readout.py')
POLICIES = parent.POLICIES + tuple(arm+suffix for arm in api.ARMS for suffix in ('_veto', '_matched'))


def register(beat):
    spec = source.read(CONFIG)
    _, cfg, data, jobs, refs, parts, points, routers, critics, fitted = parent.register(beat)
    if (spec['arms'] != list(api.ARMS) or spec['primary_fits'] != 6
            or spec['primary_scalar_boosting_rounds'] != 1536 or spec['risk_budget'] != .02
            or spec['training'] != dict(rounds=128, checkpoint_every=8, learning_rate=.05,
                max_leaf_nodes=15, min_samples_leaf=64)
            or any(spec[k] for k in ('new_encoder_training', 'new_H_readout', 'threshold_search',
                'identity_certification', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Original rules and complete fixed matched control required')
    if source.read(parent.PUBLIC/'verification.json')['analysis_sha256'] != source.sha(parent.PUBLIC/'analysis.json'):
        raise ValueError('Verified original affine control required')
    refits = {}
    for ref in source.read(parent.PUBLIC/'training_freeze.json')['fits']:
        row = source.read(source.checked(ref)); refits[row['job']] = row
    payload = source.stable(dict(configuration=spec, jobs=jobs,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(parent.REG), parent_freeze=source.artifact(parent.PUBLIC/'training_freeze.json'),
        parent_analysis=source.artifact(parent.PUBLIC/'analysis.json'),
        sklearn=api.sklearn.__version__, original_rules_unchanged=True, no_new_H=True,
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    if REG.exists() and source.read(REG) != payload:
        prior = source.read(REG); runner = str(Path(__file__).relative_to(ROOT))
        api.base.exact({k: v for k, v in prior.items() if k != 'bindings'},
            {k: v for k, v in payload.items() if k != 'bindings'})
        api.base.exact(payload['bindings'], dict(prior['bindings'], **{runner: payload['bindings'][runner]}))
        api.base.exact(source.read(AMENDMENT), dict(original_registration_sha256=source.sha(REG),
            frozen_training_sha256=source.sha(PUBLIC/'training_freeze.json'),
            original_runner_sha256=prior['bindings'][runner], amended_runner_sha256=payload['bindings'][runner],
            scope='nested_preprocess_source_exclusion_interface_only', training_changed=False,
            predictions_changed_by_interface_fix=False, scientific_rules_changed=False))
    else:
        source.once(REG, payload)
    return spec, cfg, data, jobs, refs, parts, points, routers, critics, fitted, refits


def assert_readout_excluded(state, causal, names):
    """The readout reuses its frozen parent's preprocessor, not a new one."""
    for key in ('fit_ids', 'source_sites'):
        api.base.exact(state[key], state['frozen'][key])
    direct.api.parent.assert_excluded(dict(state, preprocess=state['frozen']['preprocess']), causal, names)


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.joblib')


def fitting(data, job, refs, parts, points, routers, fitted, refits, beat, guard):
    packet, _ = direct.parent.training_packet(data, job, refs, parts[job['key']][0], 1, points, routers)
    warm = api.base.read_checkpoint(source.checked(fitted[job['key'], 'mean']['checkpoint']))
    frozen = api.base.read_checkpoint(source.checked(refits[job['key']]['checkpoint']))
    m = api.matrix(packet['learning'], warm, frozen, heartbeat=beat, guard=guard)
    return m, frozen


def one(m, frozen, job, arm, spec, refits, beat, guard, kind, resume, stop_at=None):
    path = path_for(job, arm, kind)
    return api.fit(m, frozen, arm=arm, settings=spec['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job, frozen_checkpoint=refits[job['key']]['checkpoint']),
        path=path, resume=resume, stop_at=stop_at,
        heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v), guard=guard)


def pilot(spec, cfg, data, jobs, refs, parts, points, routers, critics, fitted, refits, beat, guard, resume):
    job = jobs[0]
    m, frozen = fitting(data, job, refs, parts, points, routers, fitted, refits, beat, guard)
    rows = []
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume')
        done = api.joblib.load(path)['rounds'] if path.exists() else 0
        if done < 8:
            one(m, frozen, job, arm, spec, refits, beat, guard, 'pilot_resume', resume and path.exists(), 8)
        a = one(m, frozen, job, arm, spec, refits, beat, guard, 'pilot_resume', True, 16)
        b = one(m, frozen, job, arm, spec, refits, beat, guard, 'pilot_direct',
            resume and path_for(job, arm, 'pilot_direct').exists(), 16)
        api.exact_fit(a, b)
        source.np.testing.assert_array_equal(api.correction(a['models'], m[arm]), api.correction(b['models'], m[arm]))
        rows.append(dict(arm=arm, rounds=16, scalar_rounds=32, seconds=b['seconds'], rows=m['stats']['rows'], exact_resume=True))
    source.once(PUBLIC/'pilot.json', source.stable(dict(rows=rows, real_scalar_rounds_including_repeat=128,
        both_resume_and_inference_exact=True, no_validation_used=True)))


def train(spec, cfg, data, jobs, refs, parts, points, routers, critics, fitted, refits, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['both_resume_and_inference_exact']:
        raise ValueError('Real runtime/resume pilot required')
    receipts = []
    for job in jobs:
        m, frozen = fitting(data, job, refs, parts, points, routers, fitted, refits, beat, guard)
        for arm in api.ARMS:
            state = one(m, frozen, job, arm, spec, refits, beat, guard, 'fits',
                resume and path_for(job, arm).exists())
            if not receipts:
                repeat = one(m, frozen, job, arm, spec, refits, beat, guard, 'independent_refit',
                    resume and path_for(job, arm, 'independent_refit').exists())
                api.exact_fit(state, repeat)
                source.np.testing.assert_array_equal(api.correction(state['models'], m[arm]), api.correction(repeat['models'], m[arm]))
                source.once(PUBLIC/'first_full_refit.json', dict(exact=True,
                    checkpoint=source.artifact(path_for(job, arm, 'independent_refit'))))
            output = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(output, source.stable(dict(job=job['key'], arm=arm, checkpoint=source.artifact(path_for(job, arm)),
                frozen_checkpoint=refits[job['key']]['checkpoint'], rounds=state['rounds'], scalar_rounds=2*state['rounds'],
                seconds=state['seconds'], trace=state['trace'], stats=state['stats'], matrix_hashes=state['matrix_hashes'],
                nodes=state['node_count'], validation_readout=False, new_encoder_updates=0)))
            receipts.append(source.artifact(output))
            beat(state='model_frozen', fits=len(receipts), total=6)
        del m, frozen, state; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=receipts, complete_fits=6,
        primary_scalar_boosting_rounds=1536, additional_full_refit_scalar_rounds=256,
        all_before_validation=True, first_full_refit_exact=True, new_encoder_updates=0, new_adam_updates=0))


def evaluate(spec, cfg, data, jobs, refs, parts, points, routers, critics, fitted, refits, beat, guard, *, verify):
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_fits'] != 6 or not freeze['all_before_validation']:
        raise ValueError('All six model fits must freeze before V readout')
    heads = {}
    for ref in freeze['fits']:
        row = source.read(source.checked(ref)); heads[row['job'], row['arm']] = row
    held_rows, fit_rows, receipts = [], [], []
    for job in jobs:
        for part_id in (0, 2):
            part = parts[job['key']][part_id]
            p = old.packet(data, job, refs, part, training=False); c, ids = p['causal'], p['ids']
            chosen, predictions, hashes = direct.decisions(p, job, part, points, routers, critics, fitted, part_id == 2)
            frozen = api.base.read_checkpoint(source.checked(refits[job['key']]['checkpoint']))
            if part_id == 2:
                direct.api.parent.assert_excluded(frozen, c, part['names'])
            predictions['readout'] = api.linear.predict(frozen, c)
            hashes['readout'] = source.array_hash(predictions['readout'])
            chosen['readout_veto'], chosen['readout_matched'] = direct.api.choose(
                predictions['router'], predictions['all_action'], predictions['readout'], c)
            for arm in api.ARMS:
                state = api.joblib.load(source.checked(heads[job['key'], arm]['checkpoint']))
                if part_id == 2:
                    assert_readout_excluded(state, c, part['names'])
                predictions[arm] = api.predict(state, c, chosen['public_router'])
                hashes[arm] = source.array_hash(predictions[arm])
                chosen[arm+'_veto'], chosen[arm+'_matched'] = direct.api.choose(
                    predictions['router'], predictions['all_action'], predictions[arm], c)
            # All actions are fixed before target/validity access, including fit diagnostics.
            target = data['target_eval'][ids]-data['origin'][ids, None]
            valid = data['valid'][ids]
            truth = old.amplitude.targets(p, target, valid, job['easy_cut'])
            paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in chosen.items()}
            rows = []
            for site in sorted(set(p['sites'])):
                at = p['sites'] == site
                policies = {name: source.readout.policy(path[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                    p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, path in paths.items()}
                local = dict(moments=truth['moments'][at], observed_future=truth['observed_future'][at])
                pairs = {arm+'_minus_readout': source.readout.paired(paths['readout_veto'][at], paths[arm+'_veto'][at],
                    p['floor'][at], target[at], valid[at]) for arm in api.ARMS}
                pairs['latent_minus_score'] = source.readout.paired(paths['score_only_veto'][at], paths['latent_score_veto'][at],
                    p['floor'][at], target[at], valid[at])
                q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                matched = {arm: source.matched_utility(paths[arm+'_matched'][at], paths[arm+'_veto'][at], p['floor'][at],
                    target[at], valid[at], chosen[arm+'_matched'][at], chosen[arm+'_veto'][at], q) for arm in api.ARMS}
                reliability = {arm: direct.reliability(predictions[arm][at], local,
                    dict(proposals=chosen['public_router'][at], admitted=chosen[arm+'_veto'][at]))
                    for arm in ('readout', *api.ARMS)}
                rows.append(dict(job=job['key'], site=str(site), policies=policies, pairs=pairs, matched=matched,
                    reliability=reliability, action_flow={arm: parent.flow(local, chosen['public_router'][at],
                        chosen['readout_veto'][at], chosen[arm+'_veto'][at]) for arm in api.ARMS}))
            if part_id == 0:
                fit_rows.extend(rows)
                del p, paths, predictions, truth; source.gc.collect(); guard()
                continue
            key = job['key']+'_val2_propose1'
            previous = source.read(parent.PUBLIC/'groups'/(key+'.json'))
            for name in parent.POLICIES:
                if source.array_hash(chosen[name]) != previous['choice_hashes'][name]:
                    raise ValueError('Frozen parent action changed')
            for name, value in previous['prediction_hashes'].items():
                if hashes[name] != value:
                    raise ValueError('Frozen parent prediction changed')
            for row in rows:
                original = next(r for r in previous['rows'] if r['site'] == row['site'])
                for name in parent.POLICIES:
                    if row['policies'][name] != original['policies'][name]:
                        raise ValueError('Original evaluation changed')
            cache = PRIVATE/'choices'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(ids=ids, **{arm+'_prediction': predictions[arm] for arm in api.ARMS},
                **{arm+'_veto': chosen[arm+'_veto'] for arm in api.ARMS})
            if cache.exists():
                with source.np.load(cache, allow_pickle=False) as saved:
                    for name, value in payload.items():
                        source.np.testing.assert_array_equal(saved[name], value)
            elif verify:
                raise ValueError('Original inference cache missing')
            else:
                source.np.savez_compressed(cache, **payload)
            output = PUBLIC/'groups'/(key+'.json')
            source.once(output, source.stable(dict(rows=rows, prediction_hashes=hashes,
                choice_hashes={k: source.array_hash(v) for k, v in chosen.items()},
                cache=source.artifact(cache), parent_controls_exact=True)))
            held_rows.extend(rows); receipts.append(source.artifact(output))
            beat(state='readout_complete', groups=len(receipts), total=3)
            del p, paths, predictions, truth; source.gc.collect(); guard()
    source.once(PUBLIC/'fitting_readout.json', source.stable(fit_rows))

    def macro(rows, getter):
        values = {}
        for site in sorted({r['site'] for r in rows}):
            parts = [getter(r) for r in rows if r['site'] == site]
            values[site] = float(source.np.mean(parts)) if all(v is not None for v in parts) else None
        return dict(by_source=values, mean=float(source.np.mean(list(values.values())))
            if all(v is not None for v in values.values()) else None, CI95=None, descriptive_pilot=True)

    def summarize(rows):
        return {name: dict(known=macro(rows, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
            hard=macro(rows, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
            switch_percent=macro(rows, lambda r, n=name: 100*r['policies'][n]['switch_rate']),
            safety=direct.safety(rows, name)) for name in POLICIES}

    contrasts = {scope: {pair: {metric: macro(rows, lambda r, p=pair, m=metric: r['pairs'][p][m])
        for metric in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')}
        for pair in ('score_only_minus_readout', 'latent_score_minus_readout', 'latent_minus_score')}
        for scope, rows in (('fit', fit_rows), ('held', held_rows))}
    contexts = []
    for job in jobs:
        rows = [r for r in held_rows if r['job'] == job['key']]
        before, after = direct.safety(rows, 'readout_veto'), direct.safety(rows, 'latent_score_veto')
        lower = macro(rows, lambda r: r['pairs']['latent_score_minus_readout']['completion_lower_pp'])['mean']
        contexts.append(dict(job=job['key'], direct_lower_pp=lower, original_safety=before, latent_safety=after,
            expansion_condition=bool(lower is not None and lower > 0
                and any(r['policies']['latent_score_veto']['switch_rate'] > 0 for r in rows)
                and all(after[k] <= before[k] for k in ('known_all_violations', 'known_easy_violations')))))
    latent_lower = contrasts['held']['latent_minus_score']['completion_lower_pp']['mean']
    analysis = dict(held_policy=summarize(held_rows), fit_policy=summarize(fit_rows), contrasts=contrasts,
        per_context=contexts, expansion_supported=bool(all(c['expansion_condition'] for c in contexts)
            and latent_lower is not None and latent_lower > 0),
        sources=len({r['site'] for r in held_rows}), source_views=len(held_rows), group_receipts=receipts,
        unavailable_matched_source_views={arm: sum(r['matched'][arm]['contrast']['known_pp'] is None for r in held_rows) for arm in api.ARMS},
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        no_new_H=True, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'analysis.json', source.stable(analysis))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all3_inference_choice_and_metric_blocks_exact=True,
            all_parent_controls_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        start, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        values = register(beat)
        if args.phase == 'register':
            return
        for path in (*FILES, REG, *((AMENDMENT,) if AMENDMENT.exists() else ())):
            body = source.subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(body).hexdigest() != source.sha(path):
                raise ValueError('Commit exact registered experiment before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase receipt is immutable')
        if args.phase in ('pilot', 'train'):
            (pilot if args.phase == 'pilot' else train)(*values, beat, guard, args.resume)
        else:
            evaluate(*values, beat, guard, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            sklearn=api.sklearn.__version__, new_encoder_training=False, no_new_H=True))
        beat(state='complete')


if __name__ == '__main__':
    main()
