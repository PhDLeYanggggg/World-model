"""Fixed-budget learning-rate contrast under the unchanged risk contract."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_scale_ablation as previous
from src.world_model import m3w_annealed_risk as api
from src.evaluation import m3w_all_easy_gradient as diagnostic

source, old, direct = previous.source, previous.old, previous.direct
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_annealed_risk_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_annealed_risk_v1'
CONFIG = ROOT/'configs/m3w_moving_annealed_risk_v1.json'
REG = PUBLIC/'registration.json'
AUDIT = ROOT/'outputs/publication_readiness_2026_09/moving_all_easy_gradient_v1'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_annealed_risk.py',
    ROOT/'tests/test_m3w_annealed_risk.py', ROOT/'src/evaluation/m3w_all_easy_gradient.py')


def register(beat):
    cfg = source.read(CONFIG); _, values = previous.register(beat)
    if (cfg['arms'] != list(api.ARMS) or cfg['new_heads'] != 6 or cfg['minimum_lr_ratio'] != .01
            or cfg['training'] != source.read(previous.CONFIG)['training']
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed learning schedule and original risk/support rules required')
    api.base.exact(source.read(AUDIT/'verification.json')['analysis_sha256'], source.sha(AUDIT/'analysis.json'))
    source.once(REG, source.stable(dict(configuration=cfg,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        previous_registration=source.artifact(previous.REG), previous_result=source.artifact(previous.PUBLIC/'analysis.json'),
        fitting_diagnosis=source.artifact(AUDIT/'analysis.json'),
        jobs=values[3], partitions={j['key']: values[5][j['key']] for j in values[3]},
        warm_freeze=source.artifact(previous.parent.prior.parent.parent.PUBLIC/'training_freeze.json'),
        main_contrast='cosine_minus_constant', original_rules_unchanged=True, no_new_H=True,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed',
        submission_ready=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(d, warm, job, arm, cfg, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(d, warm, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key']),
        path=path_for(job, arm, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v))


def pilot(cfg, values, beat, guard, resume):
    job = values[3][0]; d, warm = previous.parent.prior.student_data(values, job); rows = []
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume'); step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(d, warm, job, arm, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=50)
        a = one(d, warm, job, arm, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        b = one(d, warm, job, arm, cfg, beat, guard, kind='pilot_direct',
            resume=resume and path_for(job, arm, 'pilot_direct').exists(), stop_at=100)
        api.parent.parent.parent.router.exact_state(a, b)
        source.np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
        rows.append(dict(arm=arm, exact_resume=True, steps=100, parameters=b['parameters'],
            final_pilot_rate=b['optimizer']['param_groups'][0]['lr']))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, both_resume_predictions_exact=True,
        actual_updates_including_repeats=400, no_validation_used=True))


def train(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['both_resume_predictions_exact']:
        raise ValueError('Real checkpoint-resume pilot first')
    receipts = []
    for job in values[3]:
        d, warm = previous.parent.prior.student_data(values, job); states = []
        for arm in api.ARMS:
            s = one(d, warm, job, arm, cfg, beat, guard, resume=resume and path_for(job, arm).exists())
            if s['step'] != 2000 or s['parameters'] != 32130 or not all(int(v['step']) == 2000 for v in s['optimizer']['state'].values()):
                raise ValueError('Full actual optimizer budget required')
            if api.torch.equal(s['model']['encoder.weight'], warm['model']['encoder.weight']):
                raise ValueError('Neural encoder must actually train')
            if not all(api.torch.isfinite(v).all() for v in s['model'].values()):
                raise ValueError('All neural parameters must be finite')
            if arm == 'constant':
                ref = source.read(previous.parent.PUBLIC/'fits'/(job['key']+'_continued_mse.json'))
                control = api.base.read_checkpoint(source.checked(ref['checkpoint']))
                for key in ('model', 'optimizer', 'sampler', 'draw_hash', 'input_hash', 'target_hash', 'proposal_hash'):
                    api.base.exact(s[key], control[key])
                del control
            if arm == 'cosine' and len(receipts) == 1:
                repeat = one(d, warm, job, arm, cfg, beat, guard, kind='full_refit',
                    resume=resume and path_for(job, arm, 'full_refit').exists())
                api.parent.parent.parent.router.exact_state(s, repeat)
                source.once(PUBLIC/'full_refit.json', dict(exact=True, checkpoint=source.artifact(path_for(job, arm, 'full_refit'))))
                del repeat
            model = api.parent.KinematicRiskHead(s['settings']['width'], s['seed'])
            model.load_state_dict(s['model']); model.eval()
            fit, _ = diagnostic.full(model, d, s['sampler']['counts'], False, guard)
            del model
            out = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(out, source.stable(dict(job=job['key'], arm=arm,
                checkpoint=source.artifact(path_for(job, arm)), steps=s['step'], parameters=s['parameters'],
                seconds=s['seconds'], trace=s['trace'], source_sites=s['source_sites'],
                input_hash=s['input_hash'], target_hash=s['target_hash'], proposal_hash=s['proposal_hash'],
                final_learning_rate=s['optimizer']['param_groups'][0]['lr'], full_fitting=fit,
                neural_encoder_updated=True, validation_readout=False)))
            receipts.append(source.artifact(out)); states.append(s)
            beat(state='annealed_risk_fit_frozen', complete=len(receipts), total=6)
        api.assert_matched(*states)
        del d, warm, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=receipts, complete_heads=6,
        primary_updates=12000, additional_full_refit_updates=2000, all_before_validation=True,
        all3_fresh_constant_controls_exactly_reproduce_previous=True))


def evaluate(cfg, values, beat, guard, verify=False):
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 6 or not freeze['all_before_validation']:
        raise ValueError('All six fits frozen before V')
    trained = {}
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); trained[r['job'], r['arm']] = r
    _, _, data, jobs, refs, parts, points, routers, critics, fitted, *_ = values
    all_rows, receipts = [], []
    for job in jobs:
        part = parts[job['key']][2]; p = old.packet(data, job, refs, part, training=False)
        c, ids = p['causal'], p['ids']; key = job['key']+'_val2_propose1'
        previous_row = source.read(previous.parent.PUBLIC/'groups'/(key+'.json'))
        _, original, _ = direct.decisions(p, job, part, points, routers, critics, fitted, True)
        with source.np.load(source.checked(previous_row['cache']), allow_pickle=False) as saved:
            source.np.testing.assert_array_equal(saved['ids'], ids)
            choices = {name: saved[name].copy() for name in previous_row['choice_hashes']}
            for name, choice in choices.items():
                api.base.exact(source.array_hash(choice), previous_row['choice_hashes'][name])
        predictions = {}
        for arm in api.ARMS:
            state = api.base.read_checkpoint(source.checked(trained[job['key'], arm]['checkpoint']))
            previous.parent.prior.api.assert_excluded(state, c, part['names'])
            predictions[arm] = api.predict(state, c)
            chosen, matched = api.parent.parent.choose(original['router'], original['all_action'], predictions[arm], c)
            if arm == 'constant':
                source.np.testing.assert_array_equal(chosen, choices['continued_mse_veto'])
                source.np.testing.assert_array_equal(matched, choices['continued_mse_matched'])
            choices[arm+'_veto'], choices[arm+'_matched'] = chosen, matched
        paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in choices.items()}
        target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
        rows = []
        for site in sorted(set(p['sites'])):
            at = p['sites'] == site
            policies = {name: source.readout.policy(v[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, v in paths.items()}
            old_row = next(r for r in previous_row['rows'] if r['site'] == site)
            for name, policy in old_row['policies'].items():
                api.base.exact(policies[name], policy)
            contrasts = {label: source.readout.paired(paths[left][at], paths['cosine_veto'][at],
                p['floor'][at], target[at], valid[at]) for label, left in
                (('cosine_minus_constant', 'constant_veto'), ('cosine_minus_affine', 'readout_veto'))}
            groups = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
            matched = {a: source.matched_utility(paths[a+'_matched'][at], paths[a+'_veto'][at], p['floor'][at],
                target[at], valid[at], choices[a+'_matched'][at], choices[a+'_veto'][at], groups) for a in api.ARMS}
            rows.append(dict(job=job['key'], site=str(site), policies=policies, contrasts=contrasts, matched=matched))
        payload = dict(ids=ids, **{a+'_prediction': v for a, v in predictions.items()}, **choices)
        cache = PRIVATE/'evaluation'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            with source.np.load(cache, allow_pickle=False) as saved:
                for k, v in payload.items():
                    source.np.testing.assert_array_equal(saved[k], v)
        elif verify:
            raise ValueError('Original predictions required for replay')
        else:
            source.np.savez_compressed(cache, **payload)
        out = PUBLIC/'groups'/(key+'.json')
        source.once(out, source.stable(dict(rows=rows, cache=source.artifact(cache),
            prediction_hashes={a: source.array_hash(v) for a, v in predictions.items()},
            choice_hashes={a: source.array_hash(v) for a, v in choices.items()},
            old_controls='cached_verified_actions_fresh_metric_recomputation_exact')))
        all_rows.extend(rows); receipts.append(source.artifact(out))
        beat(state='annealed_risk_readout', complete=len(receipts), total=3)
        del p, state, predictions, paths; source.gc.collect(); guard()
    macro = previous.parent.prior.parent.parent.macro
    policies = {name: dict(known=macro(all_rows, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
        hard=macro(all_rows, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
        switch_percent=macro(all_rows, lambda r, n=name: 100*r['policies'][n]['switch_rate']),
        safety=direct.safety(all_rows, name)) for name in all_rows[0]['policies']}
    contrasts = {name: {k: macro(all_rows, lambda r, n=name, k=k: r['contrasts'][n][k])
        for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for name in all_rows[0]['contrasts']}
    contexts = []
    for job in jobs:
        rs = [r for r in all_rows if r['job'] == job['key']]
        a, b = (direct.safety(rs, name) for name in ('constant_veto', 'cosine_veto'))
        lower = macro(rs, lambda r: r['contrasts']['cosine_minus_constant']['completion_lower_pp'])['mean']
        active = sum(r['policies']['cosine_veto']['switch_rate'] for r in rs) > 0
        ok = lower is not None and lower > 0 and active and all(b[k] <= a[k] for k in
            ('known_all_violations', 'known_easy_violations', 'undefined_selected_easy', 'undefined_full_easy_net'))
        contexts.append(dict(job=job['key'], completion_lower_pp=lower, control_safety=a,
            treatment_safety=b, expansion_condition=bool(ok)))
    lower = contrasts['cosine_minus_affine']['completion_lower_pp']['mean']
    result = dict(status='fresh_run', held_policy=policies, contrasts=contrasts, per_context=contexts,
        sources=len({r['site'] for r in all_rows}), group_receipts=receipts,
        expansion_supported=all(v['expansion_condition'] for v in contexts) and lower is not None and lower > 0,
        unavailable_matched_source_views={a: sum(r['matched'][a]['contrast']['known_pp'] is None for r in all_rows) for a in api.ARMS},
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        no_new_H=True, original_rules_unchanged=True, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'analysis.json', source.stable(result))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all3_predictions_actions_metrics_exact=True,
            old_cached_control_metrics_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


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
                raise ValueError('Commit exact prospective protocol before fitting')
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
