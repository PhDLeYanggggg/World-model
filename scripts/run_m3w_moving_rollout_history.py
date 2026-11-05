"""Six matched neural relation-basis fits under the unchanged original rules."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_paired_decision as old
from src.world_model import m3w_rollout_history_risk as api
from src.world_model.m3w_full_objective_risk import fitting_screen
from src.evaluation import m3w_fixed_risk_readout as readout

source = old.source
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_rollout_history_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_rollout_history_v1'
CONFIG = ROOT/'configs/m3w_moving_rollout_history_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_rollout_history_risk.py',
    ROOT/'tests/test_m3w_rollout_history_risk.py')


def register(beat):
    cfg = source.read(CONFIG); _, values = old.register(beat)
    if (cfg['arms'] != list(api.ARMS) or cfg['new_heads'] != 6 or cfg['feature_dimensions'] != len(api.NAMES)
            or cfg['training'] != source.read(old.CONFIG)['training'] or cfg['risk_budget'] != .02
            or cfg['easy_net_budget'] != .02 or any(cfg[k] for k in ('threshold_search', 'new_H_readout',
                'identity_certification', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Matched feature contrast and original rules required')
    for name in ('moving_paired_gradient_v1', 'moving_neighbor_order_v1'):
        p = PUBLIC.parent/name
        api.base.exact(source.read(p/'verification.json')['analysis_sha256'], source.sha(p/'analysis.json'))
    source.once(REG, source.stable(dict(configuration=cfg, feature_names=api.NAMES,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent=source.artifact(old.REG), parent_result=source.artifact(old.PUBLIC/'analysis.json'),
        jobs=values[3], partitions={j['key']: values[5][j['key']] for j in values[3]},
        main_contrast='history_relational_minus_history_masked', original_rules_unchanged=True,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed', no_new_H=True,
        identity_certification=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def inputs(values, job):
    data, origin = old.inputs(values, job)
    return api.prepare(data, origin), origin


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(data, origin, job, arm, cfg, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(data, origin, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key']), path=path_for(job, arm, kind),
        resume=resume, stop_at=stop_at, guard=guard, heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v))


def control_equal(state, control, causal):
    for key, value in control['model'].items():
        api.base.exact(state['model'][key], value)
    for key in ('sampler', 'draw_hash', 'step', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        api.base.exact(state[key], control[key])
    def moments(s, model):
        ids = s['optimizer']['param_groups'][0]['params']
        return {name: s['optimizer']['state'][i] for (name, _), i in zip(model.named_parameters(), ids)}
    a = moments(state, api.model_from(state)); b = moments(control, old.api.model_from(control))
    for key, value in b.items():
        api.base.exact(a[key], value)
    api.base.exact(api.predict(state, causal), old.api.predict(control, causal))


def pilot(cfg, values, beat, guard, resume):
    job = values[3][-1]; data, origin = inputs(values, job); rows = []
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume'); step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(data, origin, job, arm, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=50)
        a = one(data, origin, job, arm, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        b = one(data, origin, job, arm, cfg, beat, guard, kind='pilot_direct',
            resume=resume and path_for(job, arm, 'pilot_direct').exists(), stop_at=100)
        api.parent.exact_state(a, b); api.base.exact(api.predict(a, data['causal']), api.predict(b, data['causal']))
        if arm == 'masked':
            path = PRIVATE/'pilot_control.pt.gz'
            control = old.api.fit(data, origin, arm='joint', settings=cfg['training'], identity={},
                path=path, stop_at=100, resume=resume and path.exists(), guard=guard, heartbeat=beat)
            control_equal(b, control, data['causal'])
        rows.append(dict(arm=arm, steps=100, seconds=b['seconds'], exact_resume=True, parameters=b['parameters']))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, all2_exact_resume=True, masked_old_control_exact=True,
        actual_updates_including_repeats=500, no_validation_used=True))


def train(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['all2_exact_resume']:
        raise ValueError('Actual resume pilot required')
    refs, rows = [], []
    for index, job in enumerate(values[3]):
        data, origin = inputs(values, job); states = []
        for arm in api.ARMS:
            s = one(data, origin, job, arm, cfg, beat, guard, resume=resume and path_for(job, arm).exists())
            if s['step'] != 2000 or s['parameters'] != 68356 or not all(int(v['step']) == 2000 for v in s['optimizer']['state'].values()):
                raise ValueError('Full fixed budget and capacity required')
            for name in ('risk', 'decision'):
                if bool(s['model'][name+'.relation.weight'].any()) != (arm == 'relational'):
                    raise ValueError('Only the active relation branch should learn')
                if api.torch.equal(s['model'][name+'.encoder.weight'], s['initial_model'][name+'.encoder.weight']):
                    raise ValueError('Both neural encoders must update')
            if arm == 'masked':
                saved = source.read(old.PUBLIC/'fits'/(job['key']+'_joint.json'))
                control_equal(s, api.base.read_checkpoint(source.checked(saved['checkpoint'])), data['causal'])
            if index == 0 and arm == 'relational':
                repeat = one(data, origin, job, arm, cfg, beat, guard, kind='full_refit',
                    resume=resume and path_for(job, arm, 'full_refit').exists())
                api.parent.exact_state(s, repeat); api.base.exact(api.predict(s, data['causal']), api.predict(repeat, data['causal']))
                source.once(PUBLIC/'full_refit.json', dict(exact=True, steps=2000,
                    checkpoint=source.artifact(path_for(job, arm, 'full_refit'))))
                del repeat
            row = dict(job=job['key'], arm=arm, checkpoint=source.artifact(path_for(job, arm)),
                **{k: s[k] for k in ('step', 'seconds', 'trace', 'parameters', 'source_sites', 'input_hash',
                    'relation_hash', 'relation_raw_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash',
                    'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws')},
                initial_fitting=api.full(s, data, guard, initial=True), final_fitting=api.full(s, data, guard),
                joint_attribution=api.attribution(s, data, guard), validation_readout=False,
                masked_reproduces_old_control=arm == 'masked')
            out = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(out, source.stable(row)); refs.append(source.artifact(out)); rows.append(row); states.append(s)
            beat(state='fit_frozen', complete=len(rows), total=6)
        api.assert_matched(states); del data, origin, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=refs, complete_heads=6, primary_updates=12000,
        additional_full_refit_updates=2000, all3_masked_controls_exact=True, all_before_validation=True))
    source.once(PUBLIC/'fitting_screen.json', fitting_screen([r for r in rows if r['arm'] == 'relational']))


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
                s = api.base.read_checkpoint(source.checked(r['checkpoint'])); models[job['key'], 'history_'+arm] = s
                if verify:
                    api.base.exact(r['initial_fitting'], api.full(s, data, guard, initial=True))
                    api.base.exact(r['final_fitting'], api.full(s, data, guard))
                    api.base.exact(r['joint_attribution'], api.attribution(s, data, guard))
            del data; source.gc.collect(); guard()
    boundary = dict(fitting_screen=screen, original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False, no_new_H=True,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    if screen['followup_readout_allowed']:
        result = readout.evaluate(values, models, api.predict, names=('history_masked', 'history_relational'),
            control_public=old.previous.previous.previous.previous.previous.previous.PUBLIC,
            public=PUBLIC, private=PRIVATE, guard=guard, beat=beat, verify=verify)
        source.once(PUBLIC/'analysis.json', source.stable(dict(status='fresh_run_exposed_development', **result, **boundary)))
    else:
        source.once(PUBLIC/'analysis.json', dict(status='fitting_complete_held_readout_not_run',
            reason='Prospective relational fitting screen failed; no additional V/H predictions', **boundary))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all6_fitting_metrics_and_attributions_exact=True,
            held_predictions='exact_replay' if screen['followup_readout_allowed'] else 'not_run',
            analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = old.old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        cfg, values = register(beat)
        if args.phase == 'register':
            return
        for f in (*FILES, REG):
            committed = source.subprocess.run(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
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
