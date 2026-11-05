"""Fixed harm-weight allocation contrast with original safety rules unchanged."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_joint_decision as previous
from scripts import diagnose_m3w_harm_admission as diagnosis
from src.world_model import m3w_harm_floor_risk as api
from src.evaluation import m3w_fixed_risk_readout as readout
from src.world_model.m3w_full_objective_risk import fitting_screen
from scripts.run_m3w_moving_decoupled_risk import prediction_equal

source, old = previous.source, previous.old
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_harm_floor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_harm_floor_v1'
CONFIG = ROOT/'configs/m3w_moving_harm_floor_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_harm_floor_risk.py',
    ROOT/'src/evaluation/m3w_harm_admission.py', ROOT/'tests/test_m3w_harm_floor_risk.py',
    ROOT/'tests/test_m3w_harm_admission.py')
inputs = previous.inputs


def register(beat):
    cfg = source.read(CONFIG); _, values = previous.register(beat)
    if (cfg['arms'] != list(api.ARMS) or cfg['new_heads'] != 9 or cfg['composition_coefficient'] != .5
            or cfg['training'] != source.read(previous.CONFIG)['training']
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed model-loss contrast and original safety rules required')
    for directory in (previous.PUBLIC, diagnosis.PUBLIC):
        api.base.exact(source.read(directory/'verification.json')['analysis_sha256'], source.sha(directory/'analysis.json'))
    source.once(REG, source.stable(dict(configuration=cfg,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(previous.REG), parent_result=source.artifact(previous.PUBLIC/'analysis.json'),
        harm_admission_attribution=source.artifact(diagnosis.PUBLIC/'analysis.json'),
        controls=[source.artifact(previous.PUBLIC/'fits'/(j['key']+'_joint.json')) for j in values[3]],
        jobs=values[3], partitions={j['key']: values[5][j['key']] for j in values[3]},
        main_contrast='harm_floor_minus_uniform_harm', original_rules_unchanged=True, no_new_H=True,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed',
        submission_ready=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(d, origin, job, arm, cfg, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(d, origin, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key']), path=path_for(job, arm, kind),
        resume=resume, stop_at=stop_at, guard=guard, heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v))


def pilot(cfg, values, beat, guard, resume):
    job = values[3][-1]; d, origin = inputs(values, job); rows = []
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume'); step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(d, origin, job, arm, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=50)
        a = one(d, origin, job, arm, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        b = one(d, origin, job, arm, cfg, beat, guard, kind='pilot_direct',
            resume=resume and path_for(job, arm, 'pilot_direct').exists(), stop_at=100)
        api.parent.exact_state(a, b); prediction_equal(a, b, d['causal'])
        rows.append(dict(arm=arm, steps=100, exact_resume=True, parameters=b['parameters'], direct_seconds=b['seconds']))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, all3_exact_resume=True,
        actual_updates_including_repeats=600, no_validation_used=True))


def train(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['all3_exact_resume']:
        raise ValueError('Actual complete-group resume pilot required')
    rows, refs = [], []
    for index, job in enumerate(values[3]):
        d, origin = inputs(values, job); states = []
        for arm in api.ARMS:
            s = one(d, origin, job, arm, cfg, beat, guard, resume=resume and path_for(job, arm).exists())
            if s['step'] != 2000 or s['parameters'] != 64260 or not all(int(v['step']) == 2000 for v in s['optimizer']['state'].values()):
                raise ValueError('Complete optimizer budget and unchanged capacity required')
            if any(api.torch.equal(s['model'][k], s['initial_model'][k]) for k in ('risk.encoder.weight', 'decision.encoder.weight')):
                raise ValueError('Both neural encoders must update')
            if not all(api.torch.isfinite(v).all() for v in s['model'].values()):
                raise ValueError('Finite learned weights required')
            if arm == 'joint':
                control = source.read(previous.PUBLIC/'fits'/(job['key']+'_joint.json'))
                cached = api.base.read_checkpoint(source.checked(control['checkpoint']))
                api.previous.previous.control_equal(s, cached); prediction_equal(s, cached, d['causal'])
                del cached
            if index == 0 and arm == 'harm_floor':
                repeat = one(d, origin, job, arm, cfg, beat, guard, kind='full_refit',
                    resume=resume and path_for(job, arm, 'full_refit').exists())
                api.parent.exact_state(s, repeat); prediction_equal(s, repeat, d['causal'])
                source.once(PUBLIC/'full_refit.json', dict(exact=True, steps=repeat['step'],
                    checkpoint=source.artifact(path_for(job, arm, 'full_refit'))))
                del repeat
            row = dict(job=job['key'], arm=arm, checkpoint=source.artifact(path_for(job, arm)),
                **{k: s[k] for k in ('step', 'seconds', 'trace', 'parameters', 'source_sites', 'input_hash',
                    'target_hash', 'proposal_hash', 'observed_mask_hash', 'draw_hash', 'observed_supervised_draws',
                    'unknown_bound_supervised_draws')},
                initial_fitting=api.full(api.model_from(s, True), d, s['sampler']['counts'], guard),
                final_fitting=api.full(api.model_from(s), d, s['sampler']['counts'], guard),
                joint_attribution=api.attribution(s, d, guard),
                fitting_harm_weighting=s['fitting_harm_weighting'],
                validation_readout=False, both_neural_encoders_updated=True,
                control_reproduced_previous_exactly=arm == 'joint')
            out = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(out, source.stable(row)); rows.append(row); refs.append(source.artifact(out)); states.append(s)
            beat(state='harm_floor_fit_frozen', complete=len(rows), total=9)
        api.assert_matched(states)
        del d, origin, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=refs, complete_heads=9, primary_updates=18000,
        additional_full_refit_updates=2000, all3_controls_reproduce_previous=True, all_before_validation=True))
    source.once(PUBLIC/'fitting_screen.json', fitting_screen([r for r in rows if r['arm'] == 'harm_floor']))


def evaluate(cfg, values, beat, guard, verify=False):
    freeze = source.read(PUBLIC/'training_freeze.json'); screen = source.read(PUBLIC/'fitting_screen.json')
    if freeze['complete_heads'] != 9 or not freeze['all_before_validation']:
        raise ValueError('Freeze all nine fits before additional readout')
    models = {}
    if verify or screen['followup_readout_allowed']:
        for job in values[3]:
            d, _ = inputs(values, job)
            for arm in api.ARMS:
                r = source.read(PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                s = api.base.read_checkpoint(source.checked(r['checkpoint'])); models[job['key'], arm] = s
                if verify:
                    for initial, key in ((True, 'initial_fitting'), (False, 'final_fitting')):
                        api.base.exact(r[key], api.full(api.model_from(s, initial), d, s['sampler']['counts'], guard))
                    api.base.exact(r['joint_attribution'], api.attribution(s, d, guard))
            del d; source.gc.collect(); guard()
    boundary = dict(fitting_screen=screen, original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False, no_new_H=True,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    if screen['followup_readout_allowed']:
        result = readout.evaluate(values, models, api.predict, names=('uniform_harm', 'harm_floor', 'joint'),
            control_public=previous.previous.previous.previous.previous.PUBLIC, public=PUBLIC, private=PRIVATE,
            guard=guard, beat=beat, verify=verify)
        source.once(PUBLIC/'analysis.json', source.stable(dict(status='fresh_run', **result, **boundary)))
    else:
        source.once(PUBLIC/'analysis.json', dict(status='fitting_complete_held_readout_not_run',
            reason='Prespecified harm-floor fitting screen failed; no additional V or H predictions', **boundary))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all9_initial_final_fitting_metrics_exact=True,
            all9_joint_attributions_exact=True,
            held_predictions='exact_replay' if screen['followup_readout_allowed'] else 'not_run',
            analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        cfg, values = register(beat)
        if args.phase == 'register':
            return
        for f in (*FILES, REG):
            content = source.subprocess.run(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(content).hexdigest() != source.sha(f):
                raise ValueError('Commit prospective protocol before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase is immutable')
        if args.phase in ('pilot', 'train'):
            globals()[args.phase](cfg, values, beat, guard, args.resume)
        else:
            evaluate(cfg, values, beat, guard, args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_torch=True, new_encoder_training=args.phase in ('pilot', 'train'), no_new_H=True))
        beat(state='complete')


if __name__ == '__main__':
    main()

