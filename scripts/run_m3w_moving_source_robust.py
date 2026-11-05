"""Six fixed neural source-loss continuations; original admission rules retained."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_rollout_history as old
from src.world_model import m3w_source_robust_risk as api
from src.evaluation import m3w_fixed_risk_readout as readout

source = old.source
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_source_robust_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_source_robust_v1'
CONFIG = ROOT/'configs/m3w_moving_source_robust_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_source_robust_risk.py',
    ROOT/'tests/test_m3w_source_robust_risk.py')


def register(beat):
    cfg = source.read(CONFIG); _, values = old.register(beat)
    if (cfg['arms'] != list(api.ARMS) or cfg['new_heads'] != 6 or cfg['coefficient'] != api.COEFFICIENT
            or cfg['training']['minimum_updates'] != 1000 or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('threshold_search', 'new_H_readout', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed two-arm continuation and original rules required')
    for name in ('moving_candidate_neighbor_v1', 'moving_rollout_history_v1'):
        p = PUBLIC.parent/name
        api.base.exact(source.read(p/'verification.json')['analysis_sha256'], source.sha(p/'analysis.json'))
    checkpoints = {job['key']:source.read(old.PUBLIC/'fits'/(job['key']+'_relational.json'))['checkpoint'] for job in values[3]}
    for ref in checkpoints.values():
        source.checked(ref)
    source.once(REG, source.stable(dict(configuration=cfg,
        files={str(f.relative_to(ROOT)):source.sha(f) for f in FILES},
        parent_registration=source.artifact(old.REG), parent_result=source.artifact(old.PUBLIC/'analysis.json'),
        failed_feature_probe=source.artifact(PUBLIC.parent/'moving_candidate_neighbor_v1/analysis.json'),
        warm_checkpoints=checkpoints, jobs=values[3], partitions={j['key']:values[5][j['key']] for j in values[3]},
        main_contrast='source_robust_minus_mean_continue', original_rules_unchanged=True,
        initial_function_identical=True, parameters_per_model=68356, primary_updates=6000,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed',
        no_new_H=True, identity_certification=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def inputs(values, job):
    data, _ = old.inputs(values, job)
    ref = source.read(old.PUBLIC/'fits'/(job['key']+'_relational.json'))
    return data, api.base.read_checkpoint(source.checked(ref['checkpoint']))


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(data, warm, job, arm, cfg, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(data, warm, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key']),
        path=path_for(job, arm, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v:beat(job=job['key'], kind=kind, **v))


def pilot(cfg, values, beat, guard, resume):
    job = values[3][-1]; data, warm = inputs(values, job); rows = []
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume'); step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(data, warm, job, arm, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=50)
        a = one(data, warm, job, arm, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        b = one(data, warm, job, arm, cfg, beat, guard, kind='pilot_direct',
            resume=resume and path_for(job, arm, 'pilot_direct').exists(), stop_at=100)
        api.parent.parent.exact_state(a, b); api.base.exact(api.predict(a, data['causal']), api.predict(b, data['causal']))
        api.base.exact(a['initial_model'], warm['model'])
        rows.append(dict(arm=arm, steps=100, exact_resume=True, seconds=b['seconds'], parameters=b['parameters']))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, all2_exact_resume=True, identical_parent_initial_function=True,
        actual_updates_including_repeats=400, no_validation_used=True))


def train(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['all2_exact_resume']:
        raise ValueError('Completed real-data pilot required')
    rows, refs = [], []
    for index, job in enumerate(values[3]):
        data, warm = inputs(values, job); states = []
        for arm in api.ARMS:
            s = one(data, warm, job, arm, cfg, beat, guard, resume=resume and path_for(job, arm).exists())
            if s['step'] != 1000 or s['parameters'] != 68356 or not all(int(v['step']) == 1000 for v in s['optimizer']['state'].values()):
                raise ValueError('Every parameter must complete the fixed optimizer budget')
            for head in ('risk', 'decision'):
                if api.torch.equal(s['model'][head+'.encoder.weight'], s['initial_model'][head+'.encoder.weight']):
                    raise ValueError('Both neural encoders must update')
            if index == 0 and arm == 'source_robust':
                repeat = one(data, warm, job, arm, cfg, beat, guard, kind='full_refit',
                    resume=resume and path_for(job, arm, 'full_refit').exists())
                api.parent.parent.exact_state(s, repeat); api.base.exact(api.predict(s, data['causal']), api.predict(repeat, data['causal']))
                source.once(PUBLIC/'full_refit.json', dict(exact=True, steps=1000,
                    checkpoint=source.artifact(path_for(job, arm, 'full_refit'))))
                del repeat
            row = dict(job=job['key'], arm=arm, checkpoint=source.artifact(path_for(job, arm)),
                **{k:s[k] for k in ('step', 'seconds', 'parameters', 'trace', 'parent_step', 'parent_model_hash',
                    'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws')},
                initial_fitting=api.full(s, data, guard, initial=True), final_fitting=api.full(s, data, guard),
                initial_sources=api.source_metrics(s, data, guard, initial=True), final_sources=api.source_metrics(s, data, guard),
                joint_attribution=api.attribution(s, data, guard), validation_readout=False)
            out = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(out, source.stable(row)); refs.append(source.artifact(out)); rows.append(row); states.append(s)
            beat(state='fit_frozen', complete=len(rows), total=6)
        api.assert_matched(states); del data, warm, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=refs, complete_heads=6, primary_updates=6000,
        additional_full_refit_updates=1000, all_before_validation=True, matched_design_verified=True))
    source.once(PUBLIC/'fitting_screen.json', api.screen(rows, values[3]))


def evaluate(cfg, values, beat, guard, verify=False):
    freeze = source.read(PUBLIC/'training_freeze.json'); screen = source.read(PUBLIC/'fitting_screen.json')
    if freeze['complete_heads'] != 6 or not freeze['all_before_validation']:
        raise ValueError('Freeze all six models before readout')
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
                        api.base.exact(r[prefix+'_sources'], api.source_metrics(s, data, guard, initial=initial))
                    api.base.exact(r['joint_attribution'], api.attribution(s, data, guard))
            del data; source.gc.collect(); guard()
    boundary = dict(fitting_screen=screen, original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False, no_new_H=True,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    if screen['followup_readout_allowed']:
        result = readout.evaluate(values, models, api.predict, names=api.ARMS,
            control_public=old.old.previous.previous.previous.previous.previous.previous.PUBLIC,
            public=PUBLIC, private=PRIVATE, guard=guard, beat=beat, verify=verify)
        source.once(PUBLIC/'analysis.json', source.stable(dict(status='fresh_run_exposed_development', **result, **boundary)))
    else:
        source.once(PUBLIC/'analysis.json', dict(status='fitting_complete_held_readout_not_run',
            reason='Fixed source-robust fitting screen failed; no additional nested-C or H readout', **boundary))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all6_fitting_source_metrics_and_attributions_exact=True,
            held_predictions='exact_replay' if screen['followup_readout_allowed'] else 'not_run',
            analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.old.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = old.old.old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        cfg, values = register(beat)
        if args.phase == 'register':
            return
        for f in (*FILES, REG):
            committed = source.subprocess.run(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(committed).hexdigest() != source.sha(f):
                raise ValueError('Commit prospective bindings before real fitting')
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
