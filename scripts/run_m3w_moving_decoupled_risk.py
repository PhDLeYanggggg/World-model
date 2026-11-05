"""Matched shared/separate risk representations under the original contract."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_full_objective_risk as previous
from scripts.run_m3w_moving_kinematic_risk import assert_excluded, macro
from src.world_model import m3w_decoupled_risk as api

source, old, direct = previous.source, previous.old, previous.direct
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_decoupled_risk_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_decoupled_risk_v1'
CONFIG = ROOT/'configs/m3w_moving_decoupled_risk_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_decoupled_risk.py',
    ROOT/'tests/test_m3w_decoupled_risk.py')


def register(beat):
    cfg = source.read(CONFIG); _, values = previous.register(beat)
    original = source.read(previous.previous.CONFIG)['training']
    if (cfg['arms'] != list(api.ARMS) or cfg['new_heads'] != 9
            or cfg['training'] != dict(original, learning_rate=.0001)
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Prespecified model contrast and original risk/support rules required')
    api.base.exact(source.read(previous.PUBLIC/'verification.json')['analysis_sha256'],
        source.sha(previous.PUBLIC/'analysis.json'))
    source.once(REG, source.stable(dict(configuration=cfg,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        prior_registration=source.artifact(previous.REG), prior_result=source.artifact(previous.PUBLIC/'analysis.json'),
        starts=[source.artifact(previous.previous.PUBLIC/'fits'/(j['key']+'_constant.json')) for j in values[3]],
        jobs=values[3], partitions={j['key']: values[5][j['key']] for j in values[3]},
        main_contrast='separate64_minus_shared128', secondary='separate64_minus_shared64',
        fitting_screen='all3_separate64_mean_MSE_improves_1percent_and_both_sign_costs_nonworse',
        original_rules_unchanged=True, no_new_H=True, independent_confirmation=False,
        original24group_primary='not_estimable_not_passed', submission_ready=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(d, origin, job, arm, cfg, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(d, origin, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key']),
        path=path_for(job, arm, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v))


def prediction_equal(a, b, causal):
    for left, right in zip(api.predict(a, causal), api.predict(b, causal)):
        source.np.testing.assert_array_equal(left, right)


def pilot(cfg, values, beat, guard, resume):
    job = values[3][-1]; d, origin = previous.inputs(values, job); rows = []
    reference = api.kin.predict(origin, d['causal'])
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume')
        step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(d, origin, job, arm, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=50)
        a = one(d, origin, job, arm, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        b = one(d, origin, job, arm, cfg, beat, guard, kind='pilot_direct',
            resume=resume and path_for(job, arm, 'pilot_direct').exists(), stop_at=100)
        api.exact_state(a, b); prediction_equal(a, b, d['causal'])
        initial = dict(b, model=b['initial_model'])
        risk, decision = api.predict(initial, d['causal'])
        source.np.testing.assert_allclose(risk, reference, rtol=1e-5, atol=1e-5)
        scaled = decision*api.kin.parent.normalizer(d['preprocess'])[None, None, :]*d['causal']['scales'][:, None, None]
        source.np.testing.assert_allclose(scaled, reference, rtol=1e-5, atol=1e-5)
        rows.append(dict(job=job['key'], arm=arm, exact_resume=True, initial_function_close=True,
            steps=100, parameters=b['parameters'], direct_seconds=b['seconds'], rows=len(d['causal']['x'])))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, all3_resume_predictions_exact=True,
        actual_updates_including_repeats=600, no_validation_used=True))


def train(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['all3_resume_predictions_exact']:
        raise ValueError('Actual largest-context resume pilot required')
    rows, refs = [], []
    for index, job in enumerate(values[3]):
        d, origin = previous.inputs(values, job); states = []
        for arm in api.ARMS:
            s = one(d, origin, job, arm, cfg, beat, guard, resume=resume and path_for(job, arm).exists())
            if s['step'] != 2000 or not all(int(v['step']) == 2000 for v in s['optimizer']['state'].values()):
                raise ValueError('Complete actual optimizer budget required')
            names = ('risk.encoder.weight', 'decision.encoder.weight') if arm == 'separate64' else ('shared.encoder.weight',)
            if any(api.torch.equal(s['model'][k], s['initial_model'][k]) for k in names):
                raise ValueError('All relevant neural encoders must actually train')
            if not all(api.torch.isfinite(v).all() for v in s['model'].values()):
                raise ValueError('Finite learned parameters required')
            for k in ('sampler', 'draw_hash', 'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash'):
                api.base.exact(s[k], origin[k])
            if index == 0 and arm == 'separate64':
                repeat = one(d, origin, job, arm, cfg, beat, guard, kind='full_refit',
                    resume=resume and path_for(job, arm, 'full_refit').exists())
                api.exact_state(s, repeat); prediction_equal(s, repeat, d['causal'])
                source.once(PUBLIC/'full_refit.json', dict(exact=True, steps=repeat['step'],
                    checkpoint=source.artifact(path_for(job, arm, 'full_refit'))))
                del repeat
            row = dict(job=job['key'], arm=arm, checkpoint=source.artifact(path_for(job, arm)),
                **{k: s[k] for k in ('step', 'seconds', 'trace', 'parameters', 'source_sites', 'input_hash',
                    'target_hash', 'proposal_hash', 'observed_mask_hash', 'draw_hash', 'observed_supervised_draws',
                    'unknown_bound_supervised_draws')},
                initial_fitting=api.full(api.model_from(s, True), d, s['sampler']['counts'], guard),
                final_fitting=api.full(api.model_from(s), d, s['sampler']['counts'], guard),
                validation_readout=False, neural_encoders_updated=True)
            out = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(out, source.stable(row)); rows.append(row); refs.append(source.artifact(out)); states.append(s)
            beat(state='dual_fit_frozen', complete=len(rows), total=9)
        api.assert_matched(states)
        del d, origin, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=refs, complete_heads=9, primary_updates=18000,
        additional_full_refit_updates=2000, matched_draws_and_bounds=True, all_before_validation=True))
    source.once(PUBLIC/'fitting_screen.json', previous.api.fitting_screen([r for r in rows if r['arm'] == 'separate64']))


def verify_fitting(values, guard):
    for job in values[3]:
        d, _ = previous.inputs(values, job)
        for arm in api.ARMS:
            r = source.read(PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
            s = api.base.read_checkpoint(source.checked(r['checkpoint']))
            for initial, key in ((True, 'initial_fitting'), (False, 'final_fitting')):
                api.base.exact(r[key], api.full(api.model_from(s, initial), d, s['sampler']['counts'], guard))
        del d, s; source.gc.collect(); guard()


def evaluate(cfg, values, beat, guard, verify=False):
    freeze = source.read(PUBLIC/'training_freeze.json'); screen = source.read(PUBLIC/'fitting_screen.json')
    if freeze['complete_heads'] != 9 or not freeze['all_before_validation']:
        raise ValueError('Freeze all nine fits before development readout')
    if verify:
        verify_fitting(values, guard)
    boundary = dict(fitting_screen=screen, original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False, no_new_H=True,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    if not screen['followup_readout_allowed']:
        source.once(PUBLIC/'analysis.json', dict(status='fitting_complete_held_readout_not_run',
            reason='Prespecified separate64 fitting screen failed; no additional V or H predictions', **boundary))
        if verify:
            source.once(PUBLIC/'verification.json', dict(all9_initial_final_fitting_metrics_exact=True,
                held_predictions='not_run', analysis_sha256=source.sha(PUBLIC/'analysis.json')))
        return
    _, _, data, jobs, refs, parts, points, routers, critics, fitted, *_ = values
    rows, receipts = [], []
    for job in jobs:
        part = parts[job['key']][2]; p = old.packet(data, job, refs, part, training=False)
        c, ids = p['causal'], p['ids']; key = job['key']+'_val2_propose1'
        old_row = source.read(previous.previous.PUBLIC/'groups'/(key+'.json'))
        _, original, _ = direct.decisions(p, job, part, points, routers, critics, fitted, True)
        with source.np.load(source.checked(old_row['cache']), allow_pickle=False) as saved:
            source.np.testing.assert_array_equal(saved['ids'], ids)
            choices = {name: saved[name].copy() for name in old_row['choice_hashes']}
        for name, choice in choices.items():
            api.base.exact(source.array_hash(choice), old_row['choice_hashes'][name])
        predictions = {}
        for arm in api.ARMS:
            receipt = source.read(PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
            s = api.base.read_checkpoint(source.checked(receipt['checkpoint']))
            assert_excluded(s, c, part['names'])
            risk, decision = api.predict(s, c)
            predictions[arm+'_risk'], predictions[arm+'_decision'] = risk, decision
            for suffix, score in (('', api.combined(risk, decision)), ('_risk_only', risk)):
                name = arm+suffix
                choices[name+'_veto'], choices[name+'_matched'] = api.kin.parent.choose(
                    original['router'], original['all_action'], score, c)
        paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in choices.items()}
        # Decisions are complete before any evaluation target is fetched.
        target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
        group_rows = []
        pairs = [('separation', 'shared128_veto', 'separate64_veto'),
            ('vs_narrow', 'shared64_veto', 'separate64_veto'), ('vs_constant', 'constant_veto', 'separate64_veto')]
        for arm in api.ARMS:
            pairs.append((arm+'_decision_head', arm+'_risk_only_veto', arm+'_veto'))
        for site in sorted(set(p['sites'])):
            at = p['sites'] == site
            policies = {name: source.readout.policy(v[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, v in paths.items()}
            for name, policy in next(r for r in old_row['rows'] if r['site'] == site)['policies'].items():
                api.base.exact(policies[name], policy)
            contrasts = {label: source.readout.paired(paths[left][at], paths[right][at], p['floor'][at], target[at], valid[at])
                for label, left, right in pairs}
            groups = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
            matched = {a: source.matched_utility(paths[a+'_matched'][at], paths[a+'_veto'][at], p['floor'][at],
                target[at], valid[at], choices[a+'_matched'][at], choices[a+'_veto'][at], groups) for a in api.ARMS}
            group_rows.append(dict(job=job['key'], site=str(site), policies=policies, contrasts=contrasts, matched=matched))
        payload = dict(ids=ids, **choices, **{a+'_prediction': v for a, v in predictions.items()})
        cache = PRIVATE/'evaluation'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            with source.np.load(cache, allow_pickle=False) as saved:
                for k, v in payload.items():
                    source.np.testing.assert_array_equal(saved[k], v)
        elif verify:
            raise ValueError('Original prediction cache required')
        else:
            source.np.savez_compressed(cache, **payload)
        out = PUBLIC/'groups'/(key+'.json')
        source.once(out, source.stable(dict(rows=group_rows, cache=source.artifact(cache),
            prediction_hashes={a: source.array_hash(v) for a, v in predictions.items()},
            choice_hashes={a: source.array_hash(v) for a, v in choices.items()},
            old_controls='cached_verified_actions_fresh_metric_recomputation_exact')))
        rows.extend(group_rows); receipts.append(source.artifact(out)); beat(state='readout', completed=len(receipts))
        del p, s, predictions, paths; source.gc.collect(); guard()
    policies = {name: dict(known=macro(rows, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
        hard=macro(rows, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
        switch_percent=macro(rows, lambda r, n=name: 100*r['policies'][n]['switch_rate']),
        safety=direct.safety(rows, name)) for name in rows[0]['policies']}
    contrasts = {name: {k: macro(rows, lambda r, n=name, k=k: r['contrasts'][n][k])
        for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for name in rows[0]['contrasts']}
    source.once(PUBLIC/'analysis.json', source.stable(dict(status='fresh_run', held_policy=policies,
        contrasts=contrasts, group_receipts=receipts, **boundary)))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all9_initial_final_fitting_metrics_exact=True,
            all_predictions_actions_metrics_exact=True, old_cached_control_metrics_exact=True,
            analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        cfg, values = register(beat)
        if args.phase == 'register':
            return
        for p in (*FILES, REG):
            content = source.subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(content).hexdigest() != source.sha(p):
                raise ValueError('Commit prospective protocol before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phases remain immutable')
        if args.phase in ('pilot', 'train'):
            globals()[args.phase](cfg, values, beat, guard, args.resume)
        else:
            evaluate(cfg, values, beat, guard, args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_torch=True, no_new_H=True, new_encoder_training=args.phase in ('pilot', 'train')))
        beat(state='complete')


if __name__ == '__main__':
    main()
