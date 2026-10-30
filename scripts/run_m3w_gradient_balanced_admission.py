"""Train18 norm-balanced heads; reuse36 frozen matched controls honestly."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_admission_cost as parent
from scripts import run_m3w_admission_monitor as monitor
from src.world_model import m3w_gradient_balanced_admission as api

NAME = 'european_gradient_balanced_admission_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
CONFIG = ROOT / 'configs/m3w_gradient_balanced_admission_v1.json'
REG = PUBLIC / 'registration.json'
FILES = (Path(__file__), CONFIG, ROOT / 'src/world_model/m3w_gradient_balanced_admission.py',
    ROOT / 'tests/test_m3w_gradient_balanced_admission.py', PUBLIC / 'protocol.md')
sha, once, write = parent.sha, parent.once, parent.write


def admission(register=False):
    cfg = json.loads(CONFIG.read_text())
    pc, values, store, banks, _ = parent.admission()
    evidence = json.loads((monitor.PUBLIC / 'summary.json').read_text())
    finished = json.loads((monitor.PUBLIC / 'complete.json').read_text())
    if (cfg['training'] != pc['training'] or cfg['seeds'] != pc['seeds'] or cfg['rule'] != api.RULE
            or cfg['risk_budget'] != .02 or not evidence['all19_original_scientific_states_exact']
            or finished['summary_sha256'] != sha(monitor.PUBLIC / 'summary.json')
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                                   'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified diagnosis and fixed one-factor design required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        monitor_complete_sha256=sha(monitor.PUBLIC / 'complete.json'),
        monitor_summary_sha256=sha(monitor.PUBLIC / 'summary.json'),
        parent_training_freeze_sha256=sha(parent.PUBLIC / 'training_freeze.json'),
        parent_verification_sha256=sha(parent.PUBLIC / 'verification.json'),
        contexts=json.loads(parent.REG.read_text())['contexts'], fresh_fits=18,
        cached_verified_controls=36, fresh_primary_updates=36000,
        full_refit_updates=2000, real_pilot_updates=400,
        primary='balanced_score_minus_admitted_score', rule=api.RULE,
        decoder_and_2percent_rule_unchanged=True, fitting_roles_only=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Registered norm-parity implementation changed')
            parent.parent.frozen.engine.committed(ROOT / rel)
        parent.parent.frozen.engine.committed(REG)
    return cfg, values, store, banks, parent.models()


def runtime():
    parent.parent.torch.set_num_threads(4)
    parent.parent.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume same balance checkpoints')))
    def guard():
        if time.monotonic()-start > 43200 or shutil.disk_usage(ROOT).free < 10*2**30:
            raise RuntimeError('12h/10GiB guard; resume fixed roster')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > 128*2**20:
            raise OSError('128MiB checkpoint cap')
    def beat(**event):
        guard()
        v = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE / 'heartbeat.json', v)
        with (PRIVATE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(v) + '\n')
        print(json.dumps(v), flush=True)
    beat(state='runtime_ready', threads=4, workers=0, device='cpu')
    return start, beat, guard


def checkpoint(context, seed, phase='train'):
    return PRIVATE / phase / f'{context}_seed{seed}_balanced_score.pt.gz'


def fit(data, cfg, context, seed, path, beat, guard, **kwargs):
    return api.fit(data, variant='new_neural', settings=cfg['training'], seed=seed,
        identity=dict(context=context, seed=seed, registration_sha256=sha(REG), rule=api.RULE),
        path=path, heartbeat=lambda **kw: beat(context=context, seed=seed, **kw), guard=guard, **kwargs)


def pilot(resume=False):
    cfg, values, store, banks, _ = admission()
    if (PUBLIC / 'pilot.json').exists():
        raise ValueError('Completed pilot immutable')
    start, beat, guard = runtime()
    context, items = sorted(values[2]['contexts'].items())[0]
    _, _, _, data = parent.parent.frozen.prepared_context(context, items, values, store, banks, beat)
    path = checkpoint(context, 17, 'pilot')
    if not path.exists():
        fit(data, cfg, context, 17, path, beat, guard, stop_at=100)
    state = fit(data, cfg, context, 17, path, beat, guard, resume=True, stop_at=200)
    direct = fit(data, cfg, context, 17, checkpoint(context, 17, 'pilot_direct'), beat, guard,
        resume=resume, stop_at=200)
    parent.parent.frozen.api.assert_scientific_equal(state, direct)
    stats = state['gradient_balance']
    if (stats['zero_scale_steps'] == 200 or state['unknown_rows_with_loss']
            or not state['unknown_rows_drawn'] or state['parameters'] != 30405):
        raise ValueError('Actual balancing mechanism, unchanged size and known-only loss required')
    once(PUBLIC / 'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_updates=400, exact_recovery=True, balancing_activated=True,
        zero_scale_steps=stats['zero_scale_steps'], capped_steps=stats['capped_steps'],
        final_balance_stats=stats['rows'][-1], unknown_rows_drawn=state['unknown_rows_drawn'],
        unknown_rows_with_loss=state['unknown_rows_with_loss'],
        projected_primary_optimizer_seconds=state['seconds']*10*18,
        projection_excludes_preparation_refit_readout=True, engineering_pass=True))
    beat(state='real400update_balancing_pilot_complete')


def train(resume=False):
    cfg, values, store, banks, old = admission()
    if (PUBLIC / 'training_freeze.json').exists():
        raise ValueError('Completed18head freeze immutable')
    if not json.loads((PUBLIC / 'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real activated balancing pilot required')
    start, beat, guard = runtime()
    refs, cached = [], []
    first = sorted(values[2]['contexts'])[0]
    for context, items in sorted(values[2]['contexts'].items()):
        op, npackets, original, data = parent.parent.frozen.prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            path = checkpoint(context, seed)
            receipt = PUBLIC / 'fits' / f'{context}_seed{seed}.json'
            if receipt.exists():
                if not resume:
                    raise ValueError('Explicit partial resume required')
                row = json.loads(receipt.read_text())
                if row['registration_sha256'] != sha(REG) or row['checkpoint']['sha256'] != sha(path):
                    raise ValueError('Completed norm-parity receipt changed')
                state = parent.parent.frozen.api.completion.base.read_checkpoint(path)
            else:
                state = fit(data, cfg, context, seed, path, beat, guard, resume=resume and path.exists())
                if context == first and seed == 17:
                    rp = checkpoint(context, seed, 'full_refit')
                    refit = fit(data, cfg, context, seed, rp, beat, guard, resume=resume and rp.exists())
                    parent.parent.frozen.api.assert_scientific_equal(state, refit)
                    once(PUBLIC / 'first_full_refit.json', dict(scientific_state_exact=True, updates=2000,
                        checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(rp)))
                row = dict(context=context, seed=seed, objective_arm='balanced_score', step=state['step'],
                    seconds=state['seconds'], parameters=state['parameters'],
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                    draw_hash=state['draw_hash'], unknown_rows_drawn=state['unknown_rows_drawn'],
                    unknown_rows_with_loss=state['unknown_rows_with_loss'], gradient_balance=state['gradient_balance'],
                    first_trace=state['trace'][0], final_trace=state['trace'][-1],
                    result_source='fresh_native_arm64_Torch_gradient_norm_balanced_training',
                    registration_sha256=sha(REG), held_scored=False, independent_roles_opened=False)
                once(receipt, row)
            if state['step'] != 2000 or state['unknown_rows_with_loss'] or state['parameters'] != 30405:
                raise ValueError('Complete same-size2k training required')
            for arm in ('upper_score', 'admitted_score'):
                reference = old[context, seed, arm]
                prior = parent.parent.frozen.api.completion.base.read_checkpoint(ROOT / reference['checkpoint']['path'])
                parent.parent.frozen.api.assert_common_design(prior, state)
                cached.append(dict(context=context, seed=seed, objective_arm=arm,
                    checkpoint=reference['checkpoint'], result_source='cached_verified_frozen_matched_control',
                    common_design_exact=True))
            refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
            beat(state='new_endpoint_complete', context=context, seed=seed, completed=len(refs), total=18)
        del op, npackets, original, data
    once(PUBLIC / 'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fresh_fits=refs, count=18, fresh_primary_optimizer_updates=36000,
        full_refit_updates=2000, pilot_updates=400, cached_verified_controls=cached,
        registration_sha256=sha(REG), first_full_refit_sha256=sha(PUBLIC / 'first_full_refit.json'),
        held_scored=False, independent_roles_opened=False, deployment_changed=False, submission_ready=False))
    beat(state='all18new_heads_frozen_before_scores')


def models():
    freeze = json.loads((PUBLIC / 'training_freeze.json').read_text())
    rows = {}
    for ref in freeze['fresh_fits']:
        path = ROOT / ref['path']
        if sha(path) != ref['sha256']:
            raise ValueError('New endpoint receipt changed')
        row = json.loads(path.read_text())
        rows[row['context'], row['seed'], 'balanced_score'] = row
    for row in freeze['cached_verified_controls']:
        rows[row['context'], row['seed'], row['objective_arm']] = row
    if len(rows) != 54 or len(freeze['fresh_fits']) != 18 or len(freeze['cached_verified_controls']) != 36:
        raise ValueError('All18new36cached matched endpoints required')
    for row in rows.values():
        if sha(ROOT / row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Model checkpoint changed')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            admission(register=True)
        elif args.phase == 'pilot':
            pilot(args.resume)
        else:
            train(args.resume)
