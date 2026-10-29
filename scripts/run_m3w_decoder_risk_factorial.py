"""Registered 2x2 TRAIN-only decoder/objective experiment with exact control."""
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

from scripts import run_m3w_shared_train_risk as prior
from scripts import evaluate_m3w_shared_train_risk as previous_eval

ROOT, base = prior.ROOT, prior.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_decoder_risk_factorial_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_decoder_risk_factorial_v1'
CONFIG = ROOT/'configs/m3w_european_decoder_risk_factorial_v1.json'
REG = PUBLIC/'registration.json'


def register():
    parent, previous_cfg, ancestor = prior.admission()
    cfg = json.loads(CONFIG.read_text())
    if any(cfg[key] != previous_cfg[key] for key in ('head_seeds', 'head_training', 'objective')):
        raise ValueError('Unchanged seed, objective constants and sampling budget required')
    files = [Path(__file__).resolve(), CONFIG, PUBLIC/'protocol.md',
        ROOT/'src/world_model/m3w_decoder_risk_factorial.py', ROOT/'tests/test_m3w_decoder_risk_factorial.py',
        ROOT/'scripts/evaluate_m3w_decoder_risk_factorial.py', ROOT/'tests/test_m3w_decoder_risk_readout.py']
    value = dict(bindings={str(p.relative_to(ROOT)): base.sha(p) for p in files},
        previous_registration_sha256=base.sha(prior.REG),
        previous_training_freeze_sha256=base.sha(prior.PUBLIC/'training_freeze.json'),
        previous_held_complete_sha256=base.sha(prior.PUBLIC/'readout/complete.json'),
        previous_fit_census_complete_sha256=base.sha(prior.PUBLIC/'fit_census/complete.json'),
        contexts=parent['contexts'], seeds=cfg['head_seeds'], arms=cfg['arms'], fits=72,
        cache_verified_local_controls=72, exact_previous_shared_risk_controls_required=18,
        outer_development_read=False, independent_roles_read=False, deployment_changed=False)
    base.once(REG, value)
    return value, cfg, ancestor


def admission():
    reg, cfg, ancestor = register()
    for rel in [*reg['bindings'], str(REG.relative_to(ROOT))]:
        result = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=ROOT, capture_output=True)
        if result.returncode or result.stdout != (ROOT/rel).read_bytes():
            raise ValueError('Commit all fixed factorial code and protocol before execution')
    return reg, cfg, ancestor


def storage(cfg):
    used = sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))
    free = shutil.disk_usage(ROOT).free
    if used > cfg['checkpoint_cap_bytes'] or free < cfg['disk_reserve_bytes']+cfg['checkpoint_cap_bytes']:
        raise OSError('Preserve10GiB reserve and256MiB checkpoint cap')
    return dict(checkpoint_bytes=used, free_bytes=free)


def execute(phase, resume=False):
    from src.world_model import m3w_decoder_risk_factorial as api
    reg, cfg, ancestor = admission()
    references = previous_eval.frozen_models()
    api.torch.set_num_threads(cfg['cpu_threads'])
    api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    finish = PUBLIC/('pilot.json' if phase == 'pilot' else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed phase immutable')
    if phase == 'train' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real fit/resume/exact-control pilot required')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    started, refs = time.monotonic(), []
    def beat(**kw):
        row = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), phase=phase, **kw)
        temp = PRIVATE/'heartbeat.tmp'
        temp.write_text(json.dumps(row, allow_nan=False)+'\n')
        os.replace(temp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume checkpoints, do not lower experiment scope')
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        storage(cfg)
        contexts = sorted(reg['contexts'])[:1] if phase == 'pilot' else sorted(reg['contexts'])
        seeds = reg['seeds'][:1] if phase == 'pilot' else reg['seeds']
        for context in contexts:
            fit, _ = prior.packets(reg['contexts'][context], ancestor, beat)
            for seed in seeds:
                states = []
                identity = dict(context=context, seed=seed, partitions={i['packet']['identity']['source']: i['partition']
                                for i in reg['contexts'][context]})
                common = dict(settings=cfg['head_training'], objective_config=cfg['objective'], seed=seed,
                    identity=identity, experiment_sha256=base.sha(REG), heartbeat=beat, guard=lambda: storage(cfg))
                for arm in cfg['arms']:
                    key = context+'_head'+str(seed)+'_'+arm
                    path = PRIVATE/('pilot' if phase == 'pilot' else 'heads')/(key+'.pt.gz')
                    kwargs = dict(**common, arm=arm, path=path, resume=resume and path.exists(),
                                  stop_at=100 if phase == 'pilot' else None)
                    if phase == 'pilot' and arm == 'source_risk' and not path.exists():
                        api.fit(fit, **{**kwargs, 'stop_at': 50})
                        kwargs['resume'] = True
                    state = api.fit(fit, **kwargs)
                    states.append(state)
                    if phase == 'train':
                        ref = references[context, seed, 'shared_source', None]
                        if arm == 'shared_risk':
                            api.assert_previous_control(state, api.old.core.read_checkpoint(ROOT/ref['checkpoint']['path']))
                        doc = dict(identity=identity, arm=arm, step=state['step'],
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=base.sha(path), bytes=path.stat().st_size),
                            registration_sha256=base.sha(REG), parameter_count=state['parameter_count'],
                            primary_parameter_count=state['primary_parameter_count'],
                            previous_control_exact=True if arm == 'shared_risk' else None,
                            previous_control_sha256=ref['checkpoint']['sha256'] if arm == 'shared_risk' else None,
                            final_trace=state['trace'][-1], result_source='fresh_run_inner_TRAIN_fit',
                            inner_held_scored=False, independent_roles_read=False)
                        receipt = PUBLIC/'fits'/(key+'.json')
                        base.once(receipt, doc)
                        refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=base.sha(receipt)))
                        beat(state='fit_complete', fits=len(refs), total=72)
                api.assert_matched(states)
                if phase == 'pilot':
                    direct = PRIVATE/'pilot/direct_source_risk.pt.gz'
                    replay = api.fit(fit, arm='source_risk', path=direct, stop_at=100,
                        resume=resume and direct.exists(), **common)
                    for key in replay:
                        if key != 'seconds':
                            api.old.core.exact(replay[key], states[2][key])
                    path = PRIVATE/'pilot/previous_control.pt.gz'
                    original = api.prior.fit_shared(fit, mode='shared_source', path=path, stop_at=100,
                        resume=resume and path.exists(), **common)
                    api.assert_previous_control(states[0], original)
                    elapsed = time.monotonic()-started
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    estimate = elapsed*20*18*1.5
                    base.once(finish, dict(seconds=elapsed, peak_RSS_bytes=rss, estimated_full_seconds=estimate,
                        engineering_pass=estimate < 43200 and rss < 36*2**30, actual_updates=600,
                        exact_previous_control=True, exact_resume=True, matched_sampling=True,
                        parameter_counts={s['arm']: s['parameter_count'] for s in states},
                        registration_sha256=base.sha(REG), inner_held_scored=False,
                        independent_roles_read=False, storage=storage(cfg)))
                    beat(state='pilot_complete', estimated_full_seconds=estimate)
        if phase == 'train':
            if len(refs) != 72 or len({r['path'] for r in refs}) != 72:
                raise ValueError('Complete72-fit factorial required')
            base.once(finish, dict(fits=refs, registration_sha256=base.sha(REG), seconds=time.monotonic()-started,
                fresh_fits=72, optimizer_steps=144000, exact_previous_controls=18,
                matched_sampling=True, inner_held_scored=False, outer_development_read=False,
                independent_roles_read=False, deployment_changed=False))
            beat(state='all72_frozen_before_fit_and_held_readout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'pilot', 'train'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'register':
        register()
        print('Registered72 fixed factorial fits')
    else:
        execute(args.action, args.resume)
