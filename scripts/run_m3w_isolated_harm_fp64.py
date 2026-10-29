"""Fixed matched continuation from frozen shared-utility cost heads."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from scripts import evaluate_m3w_decoder_risk_factorial as ancestor_eval

parent = ancestor_eval.run
ROOT, base = parent.ROOT, parent.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_isolated_harm_fp64_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_isolated_harm_fp64_v1'
CONFIG = ROOT/'configs/m3w_european_isolated_harm_fp64_v1.json'
REG = PUBLIC/'registration.json'


def register():
    old, old_cfg, ancestor = parent.admission()
    cfg = json.loads(CONFIG.read_text())
    if cfg['arms'] != ['utility_continue', 'joint_risk', 'isolated_risk'] or cfg['fits'] != 54:
        raise ValueError('Fixed three-arm54-fit design required')
    if any(cfg[k] for k in ('outer_development_read', 'independent_roles_read', 'deployment_changed',
                           'stage5c_executed', 'smc_enabled')):
        raise ValueError('TRAIN-only diagnostic boundaries required')
    if any(cfg[k] != old_cfg[k] for k in ('head_seeds', 'head_training', 'objective')):
        raise ValueError('Unchanged update budget and objective constants required')
    files = [Path(__file__).resolve(), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_isolated_harm_fp64.py',
        ROOT/'tests/test_m3w_isolated_harm_fp64.py', ROOT/'scripts/evaluate_m3w_isolated_harm_fp64.py',
        ROOT/'tests/test_m3w_isolated_harm_fp64_readout.py', ROOT/'tests/test_m3w_harm_ratio_numerics.py']
    reg = dict(bindings={str(p.relative_to(ROOT)): base.sha(p) for p in files},
        parent_registration_sha256=base.sha(parent.REG),
        parent_training_freeze_sha256=base.sha(parent.PUBLIC/'training_freeze.json'),
        parent_readout_complete_sha256=base.sha(ancestor_eval.PUBLIC/'complete.json'),
        contexts=old['contexts'], seeds=cfg['head_seeds'], arms=cfg['arms'], fits=54,
        anchor_arm='shared_utility', new_optimizer=True, continue_anchor_sampling=True,
        outer_development_read=False, independent_roles_read=False, deployment_changed=False)
    base.once(REG, reg)
    return reg, cfg, ancestor


def admission():
    reg, cfg, ancestor = register()
    for rel in [*reg['bindings'], str(REG.relative_to(ROOT))]:
        result = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=ROOT, capture_output=True)
        if result.returncode or result.stdout != (ROOT/rel).read_bytes():
            raise ValueError('Commit fixed training and evaluation before real execution')
    return reg, cfg, ancestor


def storage(cfg):
    used = sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))
    free = shutil.disk_usage(ROOT).free
    if used > cfg['checkpoint_cap_bytes'] or free < cfg['disk_reserve_bytes']+cfg['checkpoint_cap_bytes']:
        raise OSError('Keep10GiB reserve and256MiB checkpoint cap')
    return dict(checkpoint_bytes=used, free_bytes=free)


def execute(phase, resume=False):
    from src.world_model import m3w_isolated_harm_fp64 as api
    reg, cfg, ancestor = admission()
    _, controls = ancestor_eval.models()
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    finish = PUBLIC/('pilot.json' if phase == 'pilot' else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed phase immutable')
    if phase == 'train' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real fit/resume/protection pilot required')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume saved atomic checkpoint')))
    started, refs = time.monotonic(), []

    def beat(**kw):
        row = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), phase=phase, **kw)
        temp = PRIVATE/'heartbeat.tmp'
        temp.write_text(json.dumps(row, allow_nan=False)+'\n')
        os.replace(temp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h limit; preserve checkpoints and resume, do not shrink scope')

    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        storage(cfg)
        contexts = sorted(reg['contexts'])[:1] if phase == 'pilot' else sorted(reg['contexts'])
        seeds = reg['seeds'][:1] if phase == 'pilot' else reg['seeds']
        for context in contexts:
            fit, _ = parent.prior.packets(reg['contexts'][context], ancestor, beat)
            for seed in seeds:
                reference = controls[context, seed, 'shared_utility']['checkpoint']
                anchor = api.core.read_checkpoint(ROOT/reference['path'])
                identity = dict(context=context, seed=seed, partitions={i['packet']['identity']['source']: i['partition']
                                for i in reg['contexts'][context]})
                common = dict(anchor=anchor, anchor_sha256=reference['sha256'], settings=cfg['head_training'],
                    objective_config=cfg['objective'], identity=identity, experiment_sha256=base.sha(REG),
                    heartbeat=beat, guard=lambda: storage(cfg))
                states = []
                for arm in cfg['arms']:
                    key = context+'_head'+str(seed)+'_'+arm
                    path = PRIVATE/('pilot' if phase == 'pilot' else 'heads')/(key+'.pt.gz')
                    kwargs = dict(**common, arm=arm, path=path, resume=resume and path.exists(),
                                  stop_at=100 if phase == 'pilot' else None)
                    if phase == 'pilot' and arm == 'isolated_risk' and not path.exists():
                        api.fit(fit, **{**kwargs, 'stop_at': 50})
                        kwargs['resume'] = True
                    state = api.fit(fit, **kwargs)
                    states.append(state)
                    if phase == 'train':
                        doc = dict(identity=identity, arm=arm, step=state['step'], anchor=reference,
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=base.sha(path), bytes=path.stat().st_size),
                            registration_sha256=base.sha(REG), total_parameters=state['total_parameters'],
                            trainable_parameters=state['trainable_parameters'], final_trace=state['trace'][-1],
                            result_source='fresh_run_inner_TRAIN_continuation', inner_held_scored=False,
                            independent_roles_read=False, new_forecaster=False)
                        receipt = PUBLIC/'fits'/(key+'.json')
                        base.once(receipt, doc)
                        refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=base.sha(receipt)))
                        beat(state='fit_complete', fits=len(refs), total=54)
                api.assert_matched(states)
                if phase == 'pilot':
                    path = PRIVATE/'pilot/direct_isolated.pt.gz'
                    replay = api.fit(fit, arm='isolated_risk', path=path, stop_at=100,
                                     resume=resume and path.exists(), **common)
                    for key in replay:
                        if key != 'seconds':
                            api.core.exact(replay[key], states[-1][key])
                    for source, args in fit.items():
                        x, env = args[0][:256], args[1][:256]
                        a = api.previous.predict(anchor, source, x, env)
                        b = api.predict(states[-1], source, x, env)
                        api.np.testing.assert_array_equal(a[:, [0, 2, 3]], b[:, [0, 2, 3]])
                    elapsed = time.monotonic()-started
                    estimate = elapsed*20*18*1.5
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    base.once(finish, dict(seconds=elapsed, peak_RSS_bytes=rss, estimated_full_seconds=estimate,
                        engineering_pass=estimate < 43200 and rss < 36*2**30, actual_updates=400,
                        exact_resume=True, matched_sampling=True, protected_outputs_exact=True,
                        parameter_counts={s['arm']: dict(total=s['total_parameters'], trainable=s['trainable_parameters']) for s in states},
                        registration_sha256=base.sha(REG), inner_held_scored=False,
                        independent_roles_read=False, storage=storage(cfg)))
                    beat(state='pilot_complete', estimated_full_seconds=estimate)
        if phase == 'train':
            if len(refs) != 54 or len({r['path'] for r in refs}) != 54:
                raise ValueError('All54 fixed-final models required')
            base.once(finish, dict(fits=refs, registration_sha256=base.sha(REG), seconds=time.monotonic()-started,
                fresh_fits=54, optimizer_steps=108000, matched_sampling=True, inner_held_scored=False,
                outer_development_read=False, independent_roles_read=False, deployment_changed=False))
            beat(state='all54_frozen_before_readout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'pilot', 'train'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'register':
        register()
        print('Registered54 fixed warm-start continuation fits')
    else:
        execute(args.action, args.resume)
