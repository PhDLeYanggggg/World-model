"""Fixed full-fitting neural optimization before conditional development readout."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_annealed_risk as previous
from src.world_model import m3w_full_objective_risk as api

source, old, direct = previous.source, previous.old, previous.direct
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_full_objective_risk_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_full_objective_risk_v1'
CONFIG = ROOT/'configs/m3w_moving_full_objective_risk_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_full_objective_risk.py',
    ROOT/'tests/test_m3w_full_objective_risk.py', ROOT/'src/evaluation/m3w_all_easy_gradient.py')


def register(beat):
    cfg = source.read(CONFIG); _, values = previous.register(beat)
    if (cfg['new_heads'] != 3 or cfg['training'] != dict(outer_steps=200, history_size=20,
            tolerance_grad=1e-7, tolerance_change=1e-12, checkpoint_every=5, heartbeat_every=10)
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed optimization diagnostic and original risk/support rules required')
    api.base.exact(source.read(previous.PUBLIC/'verification.json')['analysis_sha256'],
        source.sha(previous.PUBLIC/'analysis.json'))
    source.once(REG, source.stable(dict(configuration=cfg,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        previous_registration=source.artifact(previous.REG), previous_result=source.artifact(previous.PUBLIC/'analysis.json'),
        starts=[source.artifact(previous.PUBLIC/'fits'/(j['key']+'_constant.json')) for j in values[3]],
        jobs=values[3], partitions={j['key']: values[5][j['key']] for j in values[3]},
        main_contrast='full_objective_final_minus_precision_only_initial',
        fitting_screen='all3_mean_MSE_improves_1percent_and_both_sign_regrets_do_not_worsen',
        original_rules_unchanged=True, no_new_H=True, independent_confirmation=False,
        original24group_primary='not_estimable_not_passed', submission_ready=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def path_for(job, kind='fits'):
    return PRIVATE/kind/(job['key']+'.pt.gz')


def inputs(values, job):
    d, _ = previous.previous.parent.prior.student_data(values, job)
    r = source.read(previous.PUBLIC/'fits'/(job['key']+'_constant.json'))
    return d, api.base.read_checkpoint(source.checked(r['checkpoint']))


def one(d, origin, job, cfg, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(d, origin, settings=cfg['training'], identity=dict(registration_sha256=source.sha(REG), job=job['key']),
        path=path_for(job, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v))


def equivalent(a, b):
    for key in ('model', 'optimizer', 'torch_rng', 'step', 'changed_parameter_steps', 'full_gradient_passes',
            'trace', 'initial_fitting', 'final_fitting', 'stop_reason', 'input_hash', 'target_hash', 'proposal_hash'):
        api.base.exact(a[key], b[key])


def pilot(cfg, values, beat, guard, resume):
    rows = []
    for job in values[3]:
        d, origin = inputs(values, job); path = path_for(job, 'pilot_resume')
        step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 3:
            one(d, origin, job, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=3)
        a = one(d, origin, job, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=6)
        b = one(d, origin, job, cfg, beat, guard, kind='pilot_direct',
            resume=resume and path_for(job, 'pilot_direct').exists(), stop_at=6)
        equivalent(a, b)
        source.np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
        rows.append(dict(job=job['key'], exact_resume=True, outer_steps=b['step'],
            actual_changed_steps_including_repeat=a['changed_parameter_steps']+b['changed_parameter_steps'],
            full_gradient_passes_including_repeat=a['full_gradient_passes']+b['full_gradient_passes'],
            direct_seconds=b['seconds'], fit_rows=len(d['causal']['x']), parameters=b['parameters']))
        del d, origin, a, b; source.gc.collect(); guard()
    source.once(PUBLIC/'pilot.json', dict(rows=rows, all3_resume_predictions_exact=True, no_validation_used=True))


def train(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['all3_resume_predictions_exact']:
        raise ValueError('Actual complete-objective resume pilot required')
    rows, refs = [], []
    for index, job in enumerate(values[3]):
        d, origin = inputs(values, job)
        s = one(d, origin, job, cfg, beat, guard, resume=resume and path_for(job).exists())
        if (s['parameters'] != 32130 or s['changed_parameter_steps'] == 0
                or s['stop_reason'] not in ('fitting_stationarity', 'budget_exhausted_not_proven_converged')):
            raise ValueError('Complete real fitting budget or measured stationarity required')
        if api.torch.equal(s['model']['encoder.weight'], s['initial_model']['encoder.weight']):
            raise ValueError('Encoder must actually update')
        if index == 0:
            repeat = one(d, origin, job, cfg, beat, guard, kind='full_refit',
                resume=resume and path_for(job, 'full_refit').exists())
            equivalent(s, repeat)
            source.np.testing.assert_array_equal(api.predict(s, d['causal']), api.predict(repeat, d['causal']))
            source.once(PUBLIC/'full_refit.json', dict(exact=True, checkpoint=source.artifact(path_for(job, 'full_refit')),
                actual_changed_steps=repeat['changed_parameter_steps'], full_gradient_passes=repeat['full_gradient_passes']))
            del repeat
        model = api.model_from(s); prepared = api.prepare(d, origin)
        api.base.exact(s['final_fitting'], api.full(model, prepared, True, guard))
        model.load_state_dict(s['initial_model'])
        api.base.exact(s['initial_fitting'], api.full(model, prepared, True, guard))
        out = PUBLIC/'fits'/(job['key']+'.json')
        row = dict(job=job['key'], checkpoint=source.artifact(path_for(job)),
            **{k: s[k] for k in ('step', 'changed_parameter_steps', 'full_gradient_passes', 'lbfgs_iterations',
                'lbfgs_function_evaluations', 'seconds', 'trace', 'parameters', 'source_sites', 'input_hash',
                'target_hash', 'proposal_hash', 'initial_fitting', 'final_fitting', 'stop_reason')},
            validation_readout=False, neural_encoder_updated=True)
        source.once(out, source.stable(row)); rows.append(row); refs.append(source.artifact(out))
        beat(state='complete_objective_fit_frozen', complete=len(rows), total=3)
        del d, origin, s, model, prepared; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=refs, complete_heads=3, all_before_validation=True,
        primary_changed_steps=sum(r['changed_parameter_steps'] for r in rows),
        primary_full_gradient_passes=sum(r['full_gradient_passes'] for r in rows)))
    source.once(PUBLIC/'fitting_screen.json', api.fitting_screen(rows))


def verify_fitting(values, guard):
    for job in values[3]:
        d, origin = inputs(values, job)
        r = source.read(PUBLIC/'fits'/(job['key']+'.json'))
        s = api.base.read_checkpoint(source.checked(r['checkpoint']))
        model = api.model_from(s); prepared = api.prepare(d, origin)
        api.base.exact(r['final_fitting'], api.full(model, prepared, True, guard))
        model.load_state_dict(s['initial_model'])
        api.base.exact(r['initial_fitting'], api.full(model, prepared, True, guard))
        del d, origin, s, prepared, model; source.gc.collect(); guard()


def evaluate(cfg, values, beat, guard, verify=False):
    freeze = source.read(PUBLIC/'training_freeze.json'); screen = source.read(PUBLIC/'fitting_screen.json')
    if freeze['complete_heads'] != 3 or not freeze['all_before_validation']:
        raise ValueError('Freeze all fits before any held readout')
    if verify:
        verify_fitting(values, guard)
    if not screen['followup_readout_allowed']:
        source.once(PUBLIC/'analysis.json', dict(status='fitting_complete_held_readout_not_run',
            reason='Prospective all3 fitting screen failed; no additional V or H readout',
            fitting_screen=screen, original_rules_unchanged=True, identity_certification=False,
            original24group_primary='not_estimable_not_passed', independent_confirmation=False,
            no_new_H=True, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
        if verify:
            source.once(PUBLIC/'verification.json', dict(all3_initial_final_fitting_metrics_exact=True,
                held_predictions='not_run', analysis_sha256=source.sha(PUBLIC/'analysis.json')))
        return
    _, _, data, jobs, refs, parts, points, routers, critics, fitted, *_ = values
    rows, receipts = [], []
    for job in jobs:
        part = parts[job['key']][2]; p = old.packet(data, job, refs, part, training=False)
        c, ids = p['causal'], p['ids']; key = job['key']+'_val2_propose1'
        old_row = source.read(previous.PUBLIC/'groups'/(key+'.json'))
        _, original, _ = direct.decisions(p, job, part, points, routers, critics, fitted, True)
        with source.np.load(source.checked(old_row['cache']), allow_pickle=False) as saved:
            source.np.testing.assert_array_equal(saved['ids'], ids)
            choices = {name: saved[name].copy() for name in old_row['choice_hashes']}
        for name, choice in choices.items():
            api.base.exact(source.array_hash(choice), old_row['choice_hashes'][name])
        receipt = source.read(PUBLIC/'fits'/(job['key']+'.json'))
        s = api.base.read_checkpoint(source.checked(receipt['checkpoint']))
        previous.previous.parent.prior.api.assert_excluded(s, c, part['names'])
        predictions = {}
        for arm, initial in (('precision_only', True), ('full_objective', False)):
            predictions[arm] = api.predict(s, c, initial=initial)
            choices[arm+'_veto'], choices[arm+'_matched'] = api.parent.parent.choose(
                original['router'], original['all_action'], predictions[arm], c)
        paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in choices.items()}
        target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
        rs = []
        for site in sorted(set(p['sites'])):
            at = p['sites'] == site
            policies = {name: source.readout.policy(v[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, v in paths.items()}
            for name, policy in next(r for r in old_row['rows'] if r['site'] == site)['policies'].items():
                api.base.exact(policies[name], policy)
            contrasts = {label: source.readout.paired(paths[left][at], paths[right][at], p['floor'][at], target[at], valid[at])
                for label, left, right in (('training', 'precision_only_veto', 'full_objective_veto'),
                    ('precision', 'constant_veto', 'precision_only_veto'), ('vs_affine', 'readout_veto', 'full_objective_veto'))}
            groups = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
            matched = {a: source.matched_utility(paths[a+'_matched'][at], paths[a+'_veto'][at], p['floor'][at],
                target[at], valid[at], choices[a+'_matched'][at], choices[a+'_veto'][at], groups) for a in predictions}
            rs.append(dict(job=job['key'], site=str(site), policies=policies, contrasts=contrasts, matched=matched))
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
        source.once(out, source.stable(dict(rows=rs, cache=source.artifact(cache),
            prediction_hashes={a: source.array_hash(v) for a, v in predictions.items()},
            choice_hashes={a: source.array_hash(v) for a, v in choices.items()},
            old_controls='cached_verified_actions_fresh_metric_recomputation_exact')))
        rows.extend(rs); receipts.append(source.artifact(out)); beat(state='readout', completed=len(receipts))
        del p, s, predictions, paths; source.gc.collect(); guard()
    macro = previous.previous.parent.prior.parent.parent.macro
    policies = {name: dict(known=macro(rows, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
        hard=macro(rows, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
        switch_percent=macro(rows, lambda r, n=name: 100*r['policies'][n]['switch_rate']),
        safety=direct.safety(rows, name)) for name in rows[0]['policies']}
    contrasts = {name: {k: macro(rows, lambda r, n=name, k=k: r['contrasts'][n][k])
        for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for name in rows[0]['contrasts']}
    source.once(PUBLIC/'analysis.json', source.stable(dict(status='fresh_run', held_policy=policies,
        contrasts=contrasts, group_receipts=receipts, fitting_screen=screen,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        no_new_H=True, original_rules_unchanged=True, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all3_initial_final_fitting_metrics_exact=True,
            all3_predictions_actions_metrics_exact=True, old_cached_control_metrics_exact=True,
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
