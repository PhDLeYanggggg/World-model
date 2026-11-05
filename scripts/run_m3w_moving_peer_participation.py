"""Complete a matched 2x2 neural peer-context study, no evaluation-rule change."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_proposal_context as previous
from src.world_model import m3w_peer_participation_risk as api
from src.evaluation import m3w_fixed_risk_readout as readout

source, old = previous.source, previous.old
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_peer_participation_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_peer_participation_v1'
CONFIG = ROOT/'configs/m3w_moving_peer_participation_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_peer_participation_risk.py',
    ROOT/'tests/test_m3w_peer_participation_risk.py')
inputs = previous.inputs


def register(beat):
    cfg = source.read(CONFIG); _, values = previous.register(beat)
    if (cfg['new_arms'] != list(api.NEW_ARMS) or cfg['all_arms'] != list(api.ARMS) or cfg['new_heads'] != 6
            or cfg['training'] != source.read(previous.CONFIG)['training'] or cfg['risk_budget'] != .02
            or cfg['easy_net_budget'] != .02 or any(cfg[k] for k in ('threshold_search', 'new_H_readout',
                'identity_certification', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed factorial completion and original rules required')
    api.base.exact(source.read(previous.PUBLIC/'verification.json')['analysis_sha256'], source.sha(previous.PUBLIC/'analysis.json'))
    refs = source.read(previous.PUBLIC/'training_freeze.json')['fits']
    for ref in refs:
        source.checked(source.read(source.checked(ref))['checkpoint'])
    source.once(REG, source.stable(dict(configuration=cfg,
        files={str(f.relative_to(ROOT)):source.sha(f) for f in FILES},
        previous_registration=source.artifact(previous.REG), previous_analysis=source.artifact(previous.PUBLIC/'analysis.json'),
        previous_diagnosis=source.artifact(previous.PUBLIC/'fitting_diagnosis.json'), cached_controls=refs,
        jobs=values[3], partitions={j['key']:values[5][j['key']] for j in values[3]},
        main_contrast='proposed_all_minus_same_action', parameters_per_model=76548,
        fresh_primary_updates=6000, cached_models=6, original_rules_unchanged=True,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed',
        no_new_H=True, identity_certification=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def cached(job, arm):
    if arm not in api.previous.ARMS:
        raise ValueError('Only the completed original two arms are cached')
    r = source.read(previous.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
    return r, api.base.read_checkpoint(source.checked(r['checkpoint']))


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(data, warm, job, arm, cfg, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(data, warm, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key']), path=path_for(job, arm, kind),
        resume=resume, stop_at=stop_at, guard=guard, heartbeat=lambda **v:beat(job=job['key'], kind=kind, **v))


def pilot(cfg, values, beat, guard, resume):
    audits = []
    for job in values[3]:
        data, _ = inputs(values, job); c = data['causal']
        api.base.exact(c['x'][:, 0, 306:330], c['x'][:, 0, 330:354])
        if c['envelope'][:, 0].any() or c['allowed'][:, 0].any():
            raise ValueError('Action zero is a floor context, never an intervention')
        audits.append(dict(job=job['key'], rows=len(c['x']), zero_action_floor_exact=True,
            zero_action_not_proposed=True, proposal_hash=source.array_hash(api.proposal_from_mask(c['allowed']))))
        del data; source.gc.collect(); guard()
    source.once(PUBLIC/'floor_context_audit.json', dict(rows=audits, status='fresh_run_causal_audit', future_labels_used=False))
    job = values[3][-1]; data, warm = inputs(values, job); rows = []
    proposal = api.proposal_from_mask(data['causal']['allowed'])
    control = one(data, warm, job, 'same_action', cfg, beat, guard, kind='engine_control',
        resume=resume and path_for(job, 'same_action', 'engine_control').exists(), stop_at=100)
    original = api.base.read_checkpoint(previous.path_for(job, 'same_action', 'pilot_direct'))
    api.parent.parent.exact_state(control, original)
    api.base.exact(api.predict(control, data['causal'], proposal), api.previous.predict(original, data['causal'], proposal))
    for arm in api.NEW_ARMS:
        path = path_for(job, arm, 'pilot_resume'); step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(data, warm, job, arm, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=50)
        a = one(data, warm, job, arm, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        b = one(data, warm, job, arm, cfg, beat, guard, kind='pilot_direct',
            resume=resume and path_for(job, arm, 'pilot_direct').exists(), stop_at=100)
        api.parent.parent.exact_state(a, b)
        api.base.exact(api.predict(a, data['causal'], proposal), api.predict(b, data['causal'], proposal))
        api.base.exact(api.predict(dict(a, model=a['initial_model']), data['causal'], proposal), api.parent.predict(warm, data['causal']))
        rows.append(dict(arm=arm, steps=100, exact_resume=True, seconds=b['seconds'], parameters=b['parameters']))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, all2_exact_resume=True, legacy_engine_100_steps_exact=True,
        initial_parent_predictions_exact=True, actual_updates_including_repeats=500, no_validation_used=True))


def train(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['legacy_engine_100_steps_exact']:
        raise ValueError('Real engine-equivalence and resume pilot required')
    rows, refs = [], []
    for index, job in enumerate(values[3]):
        data, warm = inputs(values, job); states = []; proposal = api.proposal_from_mask(data['causal']['allowed'])
        for arm in api.ARMS:
            if arm in api.previous.ARMS:
                r, s = cached(job, arm)
                api.base.exact(r['final_fitting'], api.full(s, data, guard))
                api.base.exact(r['joint_attribution'], api.attribution(s, data, guard))
                row = dict(r, result_source='cached_verified', cached_receipt=source.artifact(previous.PUBLIC/'fits'/(job['key']+'_'+arm+'.json')))
            else:
                s = one(data, warm, job, arm, cfg, beat, guard, resume=resume and path_for(job, arm).exists())
                if s['step'] != 1000 or s['parameters'] != 76548 or not all(int(v['step']) == 1000 for v in s['optimizer']['state'].values()):
                    raise ValueError('Full matched optimizer budget required')
                for head in ('risk', 'decision'):
                    for key in ('encoder.weight', 'bridge.weight'):
                        if api.torch.equal(s['model'][head+'.'+key], s['initial_model'][head+'.'+key]):
                            raise ValueError('Both encoders and bridges must update')
                if index == 0 and arm == 'proposed_all':
                    repeat = one(data, warm, job, arm, cfg, beat, guard, kind='full_refit',
                        resume=resume and path_for(job, arm, 'full_refit').exists())
                    api.parent.parent.exact_state(s, repeat)
                    api.base.exact(api.predict(s, data['causal'], proposal), api.predict(repeat, data['causal'], proposal))
                    source.once(PUBLIC/'full_refit.json', dict(exact=True, steps=1000, checkpoint=source.artifact(path_for(job, arm, 'full_refit')))); del repeat
                row = dict(job=job['key'], arm=arm, result_source='fresh_run', checkpoint=source.artifact(path_for(job, arm)),
                    **{k:s[k] for k in ('step', 'seconds', 'parameters', 'trace', 'parent_step', 'parent_model_hash',
                        'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws')},
                    initial_fitting=api.full(s, data, guard, initial=True), final_fitting=api.full(s, data, guard),
                    joint_attribution=api.attribution(s, data, guard), validation_readout=False)
            out = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(out, source.stable(row)); refs.append(source.artifact(out)); rows.append(row); states.append(s)
            beat(state='fit_frozen_or_verified', complete=len(rows), total=12, result_source=row['result_source'])
        api.assert_matched(states); del data, warm, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=refs, fresh_heads=6, cached_verified_heads=6,
        primary_updates=6000, additional_full_refit_updates=1000, all_before_validation=True, matched_factorial_verified=True))
    source.once(PUBLIC/'fitting_screen.json', api.screen(rows, values[3]))


def evaluate(cfg, values, beat, guard, verify=False):
    freeze = source.read(PUBLIC/'training_freeze.json'); screen = source.read(PUBLIC/'fitting_screen.json')
    if freeze['fresh_heads'] != 6 or freeze['cached_verified_heads'] != 6 or not freeze['all_before_validation']:
        raise ValueError('Freeze complete factorial comparison before readout')
    models = {}
    if verify or screen['followup_readout_allowed']:
        for job in values[3]:
            data, _ = inputs(values, job)
            for arm in api.ARMS:
                r = source.read(PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                s = api.base.read_checkpoint(source.checked(r['checkpoint'])); models[job['key'], arm] = s
                if verify:
                    for initial, prefix in ((True, 'initial'), (False, 'final')):
                        api.base.exact(r[prefix+'_fitting'], api.full(s, data, guard, initial=initial))
                    api.base.exact(r['joint_attribution'], api.attribution(s, data, guard))
            del data; source.gc.collect(); guard()
    boundary = dict(fitting_screen=screen, original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False, no_new_H=True,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    if screen['followup_readout_allowed']:
        def predict(s, c):
            return api.predict(s, c, previous.causal_proposal(values, s['identity']['job'], c))
        result = readout.evaluate(values, models, predict, names=('same_action', 'proposed_all'),
            control_public=old.old.previous.previous.previous.previous.previous.previous.PUBLIC,
            public=PUBLIC, private=PRIVATE, guard=guard, beat=beat, verify=verify)
        source.once(PUBLIC/'analysis.json', source.stable(dict(status='fresh_run_exposed_development', **result, **boundary)))
    else:
        source.once(PUBLIC/'analysis.json', dict(status='factorial_fitting_complete_held_readout_not_run',
            reason='Fixed proposed-all versus same-all fitting screen failed; no additional nested-C or H readout', **boundary))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all12_fitting_metrics_and_attributions_exact=True,
            held_predictions='exact_replay' if screen['followup_readout_allowed'] else 'not_run', analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        runtime = old.old.old.previous.teacher.lineage; runtime.PRIVATE = PRIVATE
        started, guard, beat = runtime.runtime({}, source.read(CONFIG), args.phase)
        cfg, values = register(beat)
        if args.phase == 'register':
            return
        for f in (*FILES, REG):
            committed = source.subprocess.run(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if source.hashlib.sha256(committed).hexdigest() != source.sha(f):
                raise ValueError('Commit prospective bindings before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase is immutable')
        if args.phase in ('pilot', 'train'):
            globals()[args.phase](cfg, values, beat, guard, args.resume)
        else:
            evaluate(cfg, values, beat, guard, args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            new_encoder_training=args.phase in ('pilot', 'train'), real_torch=True, no_new_H=True))
        beat(state='complete')


if __name__ == '__main__':
    main()
