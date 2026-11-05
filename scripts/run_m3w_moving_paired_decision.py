"""Fixed paired decision experiment; no evaluation-rule or threshold change."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_harm_floor as previous
from scripts import diagnose_m3w_discrimination as diagnosis
from src.world_model import m3w_paired_decision_risk as api
from src.evaluation import m3w_fixed_risk_readout as readout
from src.evaluation.m3w_discrimination_audit import audit
from src.world_model.m3w_full_objective_risk import fitting_screen

source, old = previous.source, previous.old
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_paired_decision_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_paired_decision_v1'
CONFIG = ROOT/'configs/m3w_moving_paired_decision_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_paired_decision_risk.py',
    ROOT/'tests/test_m3w_paired_decision_risk.py', ROOT/'src/evaluation/m3w_discrimination_audit.py',
    ROOT/'tests/test_m3w_discrimination_audit.py')
inputs = previous.inputs


def register(beat):
    cfg = source.read(CONFIG); _, values = previous.register(beat)
    if (cfg['arms'] != list(api.ARMS) or cfg['new_heads'] != 9 or cfg['paired_coefficient'] != .5
            or cfg['training'] != source.read(previous.CONFIG)['training']
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed loss contrast and unchanged original safety required')
    for directory in (previous.PUBLIC, diagnosis.PUBLIC):
        api.base.exact(source.read(directory/'verification.json')['analysis_sha256'], source.sha(directory/'analysis.json'))
    api.base.exact(source.read(diagnosis.PUBLIC/'context_verification.json')['sha256'],
        source.sha(diagnosis.PUBLIC/'context_alias.json'))
    source.once(REG, source.stable(dict(configuration=cfg,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(previous.REG), parent_result=source.artifact(previous.PUBLIC/'analysis.json'),
        discrimination=source.artifact(diagnosis.PUBLIC/'analysis.json'),
        context_alias=source.artifact(diagnosis.PUBLIC/'context_alias.json'),
        jobs=values[3], partitions={j['key']: values[5][j['key']] for j in values[3]},
        main_contrast='paired_rank_minus_paired_point', original_rules_unchanged=True, no_new_H=True,
        independent_confirmation=False, original24group_primary='not_estimable_not_passed',
        submission_ready=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(data, origin, job, arm, cfg, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(data, origin, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key']), path=path_for(job, arm, kind),
        resume=resume, stop_at=stop_at, guard=guard, heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v))


def same_prediction(a, b, causal):
    for x, y in zip(api.predict(a, causal), api.predict(b, causal)):
        source.np.testing.assert_array_equal(x, y)


def discrimination(state, data, guard):
    c = data['causal']; z = api.parent.validate(data, state); model = api.model_from(state)
    risk, decision = (source.np.empty((len(c['ids']), 7, 2), source.np.float32) for _ in range(2))
    for ix, seg in api.parent.batches(c):
        guard()
        with api.torch.no_grad():
            r, d = model(z[ix], api.torch.from_numpy(seg), int(seg.max())+1)
        risk[ix], decision[ix] = r.numpy(), d.numpy()
    y = (api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
        api.torch.from_numpy(api.parent.kin.parent.normalizer(data['preprocess']))).numpy()
    return audit(risk, decision, y, c['allowed'], data['observed_future'],
        api.parent.query_weights(c, state['sampler']['counts']), c['sites'])


def pilot(cfg, values, beat, guard, resume):
    job = values[3][-1]; data, origin = inputs(values, job); rows = []
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume'); step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(data, origin, job, arm, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=50)
        a = one(data, origin, job, arm, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        b = one(data, origin, job, arm, cfg, beat, guard, kind='pilot_direct',
            resume=resume and path_for(job, arm, 'pilot_direct').exists(), stop_at=100)
        api.parent.exact_state(a, b); same_prediction(a, b, data['causal'])
        rows.append(dict(arm=arm, steps=100, exact_resume=True, parameters=b['parameters'], direct_seconds=b['seconds']))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, all3_exact_resume=True,
        actual_updates_including_repeats=600, no_validation_used=True))


def train(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['all3_exact_resume']:
        raise ValueError('Actual complete-query resume pilot required')
    rows, refs = [], []
    for index, job in enumerate(values[3]):
        data, origin = inputs(values, job); states = []
        for arm in api.ARMS:
            s = one(data, origin, job, arm, cfg, beat, guard, resume=resume and path_for(job, arm).exists())
            if s['step'] != 2000 or s['parameters'] != 64260 or not all(int(v['step']) == 2000 for v in s['optimizer']['state'].values()):
                raise ValueError('Full optimizer budget and unchanged capacity required')
            if any(api.torch.equal(s['model'][k], s['initial_model'][k]) for k in ('risk.encoder.weight', 'decision.encoder.weight')):
                raise ValueError('Both neural encoders must update')
            if arm == 'joint':
                ref = source.read(previous.PUBLIC/'fits'/(job['key']+'_joint.json'))
                cached = api.base.read_checkpoint(source.checked(ref['checkpoint']))
                api.previous.previous.control_equal(s, cached); same_prediction(s, cached, data['causal'])
                del cached
            if index == 0 and arm == 'paired_rank':
                repeat = one(data, origin, job, arm, cfg, beat, guard, kind='full_refit',
                    resume=resume and path_for(job, arm, 'full_refit').exists())
                api.parent.exact_state(s, repeat); same_prediction(s, repeat, data['causal'])
                source.once(PUBLIC/'full_refit.json', dict(exact=True, steps=repeat['step'],
                    checkpoint=source.artifact(path_for(job, arm, 'full_refit'))))
                del repeat
            row = dict(job=job['key'], arm=arm, checkpoint=source.artifact(path_for(job, arm)),
                **{k: s[k] for k in ('step', 'seconds', 'trace', 'parameters', 'source_sites', 'input_hash',
                    'target_hash', 'proposal_hash', 'observed_mask_hash', 'draw_hash', 'observed_supervised_draws',
                    'unknown_bound_supervised_draws')},
                initial_fitting=api.full(api.model_from(s, True), data, s['sampler']['counts'], guard),
                final_fitting=api.full(api.model_from(s), data, s['sampler']['counts'], guard),
                joint_attribution=api.attribution(s, data, guard),
                discrimination=discrimination(s, data, guard), validation_readout=False,
                both_neural_encoders_updated=True, control_reproduced_previous_exactly=arm == 'joint')
            out = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(out, source.stable(row)); rows.append(row); refs.append(source.artifact(out)); states.append(s)
            beat(state='paired_fit_frozen', complete=len(rows), total=9)
        api.assert_matched(states)
        del data, origin, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=refs, complete_heads=9, primary_updates=18000,
        additional_full_refit_updates=2000, all3_controls_reproduce_previous=True, all_before_validation=True))
    source.once(PUBLIC/'fitting_screen.json', fitting_screen([r for r in rows if r['arm'] == 'paired_rank']))


def evaluate(cfg, values, beat, guard, verify=False):
    freeze = source.read(PUBLIC/'training_freeze.json'); screen = source.read(PUBLIC/'fitting_screen.json')
    if freeze['complete_heads'] != 9 or not freeze['all_before_validation']:
        raise ValueError('Freeze all nine fits before readout')
    models = {}
    if verify or screen['followup_readout_allowed']:
        for job in values[3]:
            data, _ = inputs(values, job)
            for arm in api.ARMS:
                r = source.read(PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                s = api.base.read_checkpoint(source.checked(r['checkpoint'])); models[job['key'], arm] = s
                if verify:
                    for initial, key in ((True, 'initial_fitting'), (False, 'final_fitting')):
                        api.base.exact(r[key], api.full(api.model_from(s, initial), data, s['sampler']['counts'], guard))
                    api.base.exact(r['joint_attribution'], api.attribution(s, data, guard))
                    api.base.exact(r['discrimination'], discrimination(s, data, guard))
            del data; source.gc.collect(); guard()
    boundary = dict(fitting_screen=screen, original_rules_unchanged=True, identity_certification=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False, no_new_H=True,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    if screen['followup_readout_allowed']:
        result = readout.evaluate(values, models, api.predict, names=('paired_point', 'paired_rank', 'joint'),
            control_public=previous.previous.previous.previous.previous.previous.PUBLIC,
            public=PUBLIC, private=PRIVATE, guard=guard, beat=beat, verify=verify)
        source.once(PUBLIC/'analysis.json', source.stable(dict(status='fresh_run', **result, **boundary)))
    else:
        source.once(PUBLIC/'analysis.json', dict(status='fitting_complete_held_readout_not_run',
            reason='Prespecified paired-rank fitting screen failed; no additional V or H predictions', **boundary))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all9_initial_final_fitting_metrics_exact=True,
            all9_joint_attributions_exact=True, all9_discrimination_exact=True,
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
                raise ValueError('Commit prospective scientific bindings before fitting')
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
