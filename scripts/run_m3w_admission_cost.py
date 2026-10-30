"""Fixed54-head online-admission experiment, checkpointed and recoverable."""
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
from scripts import run_m3w_budget_score_cost as parent
from scripts import run_m3w_selected_action_attribution as diagnosis
from src.world_model import m3w_admission_cost as api

NAME = 'european_admission_cost_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
CONFIG = ROOT / 'configs/m3w_admission_cost_v1.json'
REG = PUBLIC / 'registration.json'
sha, once, write = parent.sha, parent.once, parent.write
FILES = (Path(__file__), CONFIG, ROOT / 'src/world_model/m3w_admission_cost.py',
    ROOT / 'tests/test_m3w_admission_cost.py', PUBLIC / 'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text())
    pc, values, _, store, banks = parent.admission()
    check = json.loads((diagnosis.PUBLIC / 'verification.json').read_text())
    if (cfg['training'] != pc['training'] or cfg['seeds'] != pc['seeds']
            or cfg['objectives'] != list(api.OBJECTIVES) or cfg['risk_budget'] != .02
            or cfg['admission_penalty_weight'] != 1. or not check['all216groups_exact']
            or check['summary_sha256'] != sha(diagnosis.PUBLIC / 'summary.json')
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                                   'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed one-factor design and verified prior diagnosis required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_training_freeze_sha256=sha(parent.PUBLIC / 'training_freeze.json'),
        diagnosis_summary_sha256=sha(diagnosis.PUBLIC / 'summary.json'),
        diagnosis_verification_sha256=sha(diagnosis.PUBLIC / 'verification.json'),
        contexts=json.loads(parent.REG.read_text())['contexts'],
        fits=54, primary_optimizer_updates=108000, steps_per_head=2000,
        exact_original_and_upper_controls=36, full_refit_updates=2000, pilot_updates=160,
        variant='new_neural', parameters=30405, causal_fields=462,
        objective_factor='unit_online_selected_false_admission_on_upper_score',
        inference_decoder_unchanged=True, risk_budget=.02, fitting_roles_only=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Registered online admission changed: ' + rel)
            parent.frozen.engine.committed(ROOT / rel)
        parent.frozen.engine.committed(REG)
    return cfg, values, store, banks, parent.models()


def runtime(cfg):
    parent.torch.set_num_threads(4)
    parent.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume same checkpoints')))
    def guard():
        if time.monotonic() - start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume fixed experiment')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Keep10GiB free reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Private checkpoint cap exceeded')
    def beat(**event):
        guard()
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE / 'heartbeat.json', value)
        with (PRIVATE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), threads=4, workers=0, device='cpu')
    return start, beat, guard


def checkpoint(context, seed, objective, phase='train'):
    return PRIVATE / phase / f'{context}_seed{seed}_{objective}.pt.gz'


def fit(data, cfg, context, seed, objective, path, beat, guard, *, resume=False, stop_at=None):
    return api.fit(data, objective_arm=objective, variant='new_neural', settings=cfg['training'], seed=seed,
        identity=dict(registration_sha256=sha(REG), context=context, objective_arm=objective),
        path=path, heartbeat=lambda **kw: beat(context=context, objective_arm=objective, **kw),
        guard=guard, resume=resume, stop_at=stop_at)


def pilot(resume=False):
    cfg, values, store, banks, _ = admission()
    if (PUBLIC / 'pilot.json').exists():
        raise ValueError('Completed pilot immutable')
    start, beat, guard = runtime(cfg)
    context, items = sorted(values[2]['contexts'].items())[0]
    _, _, _, data = parent.frozen.prepared_context(context, items, values, store, banks, beat)
    states, cells = [], []
    for obj in cfg['objectives']:
        path = checkpoint(context, 17, obj, 'pilot')
        if not path.exists():
            fit(data, cfg, context, 17, obj, path, beat, guard, stop_at=10)
        state = fit(data, cfg, context, 17, obj, path, beat, guard, resume=True, stop_at=20)
        direct = fit(data, cfg, context, 17, obj, checkpoint(context, 17, obj, 'pilot_direct'),
            beat, guard, resume=resume, stop_at=20)
        parent.frozen.api.assert_scientific_equal(state, direct)
        if obj != 'admitted_score':
            reference = parent.api.fit(data, objective_arm=obj, variant='new_neural', settings=cfg['training'],
                seed=17, identity=dict(pilot_control=True),
                path=checkpoint(context, 17, obj, 'pilot_control'), heartbeat=beat, guard=guard,
                resume=resume, stop_at=20)
            parent.frozen.api.assert_scientific_equal(state, reference)
        if not state['unknown_rows_drawn'] or state['unknown_rows_with_loss'] or state['parameters'] != 30405:
            raise ValueError('Real unchanged-size training and known-only supervision required')
        states.append(state)
        cells.append(dict(objective=obj, seconds=state['seconds'], updates=20, exact_recovery=True,
            exact_parent_control=obj != 'admitted_score', unknown_rows_drawn=state['unknown_rows_drawn'],
            unknown_rows_with_loss=state['unknown_rows_with_loss']))
    for other in states[1:]:
        parent.frozen.api.assert_common_design(states[0], other)
    once(PUBLIC / 'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        real_optimizer_updates=160, cells=cells, engineering_pass=True,
        projected_primary_optimizer_seconds=max(r['seconds'] for r in cells) * 100 * 54,
        projection_excludes_preparation_refit_and_readout=True))
    beat(state='real160update_pilot_complete')


def train(resume=False):
    cfg, values, store, banks, legacy = admission()
    if (PUBLIC / 'training_freeze.json').exists():
        raise ValueError('Completed54head freeze immutable')
    if not json.loads((PUBLIC / 'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Measured real pilot required')
    start, beat, guard = runtime(cfg)
    refs, controls = [], 0
    first = sorted(values[2]['contexts'])[0]
    for context, items in sorted(values[2]['contexts'].items()):
        op, npackets, original, data = parent.frozen.prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            states = []
            for obj in cfg['objectives']:
                path = checkpoint(context, seed, obj)
                receipt = PUBLIC / 'fits' / f'{context}_seed{seed}_{obj}.json'
                if receipt.exists():
                    if not resume:
                        raise ValueError('Explicit partial resume required')
                    row = json.loads(receipt.read_text())
                    if row['registration_sha256'] != sha(REG) or row['checkpoint']['sha256'] != sha(path):
                        raise ValueError('Completed receipt changed')
                    state = parent.frozen.api.completion.base.read_checkpoint(path)
                else:
                    state = fit(data, cfg, context, seed, obj, path, beat, guard,
                        resume=resume and path.exists())
                    if obj != 'admitted_score':
                        old = parent.frozen.api.completion.base.read_checkpoint(ROOT / legacy[context, seed, 'new_neural', obj]['checkpoint']['path'])
                        parent.frozen.api.assert_scientific_equal(state, old)
                    if context == first and seed == 17 and obj == 'admitted_score':
                        rp = checkpoint(context, seed, obj, 'full_refit')
                        refit = fit(data, cfg, context, seed, obj, rp, beat, guard, resume=resume and rp.exists())
                        parent.frozen.api.assert_scientific_equal(state, refit)
                        once(PUBLIC / 'first_full_refit.json', dict(scientific_state_exact=True,
                            updates=2000, checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(rp)))
                    row = dict(context=context, seed=seed, objective_arm=obj, step=state['step'],
                        seconds=state['seconds'], parameters=state['parameters'],
                        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                        draw_hash=state['draw_hash'], unknown_rows_drawn=state['unknown_rows_drawn'],
                        unknown_rows_with_loss=state['unknown_rows_with_loss'],
                        exact_parent_control=obj != 'admitted_score',
                        first_trace=state['trace'][0], final_trace=state['trace'][-1],
                        registration_sha256=sha(REG), result_source='fresh_native_arm64_Torch_online_admission_training',
                        held_scored=False, independent_roles_opened=False)
                    once(receipt, row)
                if state['step'] != 2000 or state['parameters'] != 30405 or state['unknown_rows_with_loss']:
                    raise ValueError('Complete fixed training budget required')
                if obj != 'admitted_score':
                    if not row['exact_parent_control']:
                        raise ValueError('Exact parent control required')
                    controls += 1
                if states:
                    parent.frozen.api.assert_common_design(states[0], state)
                states.append(state)
                refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                beat(state='endpoint_complete', context=context, seed=seed, objective_arm=obj,
                    fits=len(refs), total=54, exact_controls=controls)
        del op, npackets, original, data, states
    if len(refs) != 54 or controls != 36:
        raise ValueError('Complete54heads36exact controls required')
    once(PUBLIC / 'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fits=refs, count=54,
        primary_optimizer_updates=108000, full_refit_updates=2000, pilot_updates=160,
        exact_parent_controls=controls, first_full_refit_sha256=sha(PUBLIC / 'first_full_refit.json'),
        registration_sha256=sha(REG), held_scored=False, independent_roles_opened=False,
        deployment_changed=False, submission_ready=False))
    beat(state='all54heads_frozen_no_readout')


def models():
    freeze = json.loads((PUBLIC / 'training_freeze.json').read_text())
    output = {}
    for ref in freeze['fits']:
        path = ROOT / ref['path']
        if sha(path) != ref['sha256']:
            raise ValueError('Head receipt changed')
        row = json.loads(path.read_text())
        if sha(ROOT / row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Private weights changed')
        key = row['context'], row['seed'], row['objective_arm']
        if key in output:
            raise ValueError('Duplicate endpoint')
        output[key] = row
    if len(output) != 54:
        raise ValueError('Complete54head roster required')
    return output


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
