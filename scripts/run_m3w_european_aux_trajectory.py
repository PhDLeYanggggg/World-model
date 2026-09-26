"""Reconstruct registered training without changing its numerical trajectory."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
from scripts import run_m3w_european_aux_gradient as previous
from src.world_model import m3w_aux_trajectory as method
import torch

parent = previous.parent
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_aux_trajectory_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_aux_trajectory_v1'
CONFIG = 'configs/m3w_european_aux_trajectory_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_aux_trajectory.py', 'tests/test_m3w_aux_trajectory.py',
    'scripts/run_m3w_european_aux_trajectory.py', 'scripts/report_m3w_european_aux_trajectory.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    parent.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    _, identity = previous.registration()
    check = json.loads((previous.PUBLIC/'verification.json').read_text())
    assert check['all_passed']
    for p, h in check['artifacts'].items(): assert digest(previous.PUBLIC/p) == h
    for p, h in check['source_bindings'].items(): assert digest(ROOT/p) == h
    assert cfg['arms'] == list(parent.method.ARMS)
    assert cfg['checkpoints'] == [200, 600, 1000, 1400, 2000]
    assert cfg['views'] == 144 and cfg['reconstructed_heads'] == 432 and cfg['snapshots'] == 2160
    assert cfg['training_updates'] == 864000 and cfg['seeds'] == [17, 29, 43]
    assert not any(cfg[k] for k in ('new_model_variant', 'new_policy_evaluation', 'checkpoint_selection',
        'held_readout', 'selection_access', 'reserved_calibration_access', 'confirmation_access',
        'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    doc = dict(parent_registration=artifact(parent.PUBLIC/'registration_lock.json'),
        parent_verification=artifact(parent.PUBLIC/'verification.json'),
        gradient_verification=artifact(previous.PUBLIC/'verification.json'),
        create_queue_receipt=artifact(PRIVATE/'create_queue.json'),
        bindings={p: digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, doc)
    else:
        assert json.loads(path.read_text()) == doc
        parent.risk.base.previous.require_committed(path)
    return cfg, identity


def validate_receipt(path, identity):
    doc = json.loads(path.read_text()); assert doc['identity'] == identity
    for a in doc['snapshots']:
        assert artifact(ROOT/a['path']) == a
        stage = json.loads((ROOT/a['path']).read_text())
        assert stage['identity'] == identity
        assert artifact(ROOT/stage['snapshot']['path']) == stage['snapshot']
    assert artifact(ROOT/doc['final_checkpoint']['path']) == doc['final_checkpoint']
    assert artifact(ROOT/doc['original_checkpoint']['path']) == doc['original_checkpoint']
    return doc


def diagnostics(v, x, y, event, model, state, spec, z, target, samples, identity, snap):
    return dict(identity=identity, step=state['step'], snapshot=artifact(snap),
        severity_spec=spec, fitting_ids_sha256=array_hash(v['ids']),
        gradient_batch_ids_sha256=[array_hash(v['ids'][ids]) for ids in samples],
        localities=method.measure(model, state, z, target, y, v['env'], v['sites'], spec, event),
        gradients=method.gradients(model, state, z, target, v['env'], event, samples),
        result_source='fresh_run_reconstructed_training_fitting_diagnostic',
        outer_outcomes_used=False, inference_features_changed=False)


def experiment(cfg, identity, pilot=False, verify=False):
    parent.check_sources(identity)
    records = {r['tag']: r['input'] for r in json.loads((parent.PUBLIC/'support_report.json').read_text())['rows']}
    settings = json.loads((ROOT/parent.CONFIG).read_text())['head_training']
    refs = []; started = time.monotonic(); train_seconds = 0.
    for v in parent.views(identity):
        if shutil.disk_usage(PRIVATE).free < cfg['minimum_free_gib']*2**30:
            raise OSError('Preserve10GiB free; resume retained states after resolving disk capacity')
        x, y, event = previous.fitting_inputs(v, records[v['tag']])
        easy = parent.labels(v['cv'], v['pr']['positive_easy_cut'])
        seed = int(v['g']['group'].split('_seed')[1].split('_')[0])
        assert seed in cfg['seeds']
        spec = method.severity_spec(y, v['env'], v['pr']['weights'], cfg['severity_quantiles'])
        z, target = method.tensors(x, y, v['pr'])
        samples = [previous.method.sample_rows(v['pr']['known'], v['sites'], seed+18731, r, 256, 256)[0]
                   for r in range(cfg['gradient_batches_per_snapshot'])]
        for arm in cfg['arms']:
            home = PRIVATE/'heads'/v['tag']/arm; complete = home/'complete.json'
            hid = dict(registration=artifact(PUBLIC/'registration_lock.json'), tag=v['tag'],
                group=v['g']['group'], producer=v['g']['producer'], controller=v['g']['controller'],
                pair=v['pair'], excluded_locality=v['outer'], seed=seed, arm=arm, input=records[v['tag']])
            if complete.exists() and not verify:
                receipt = validate_receipt(complete, hid)
            else:
                if verify and not complete.exists(): raise ValueError('Missing completed training')
                stages = []
                for step in cfg['checkpoints']:
                    stage_path = home/f'step{step}.json'; snap = home/f'step{step}.pt'
                    if stage_path.exists() and not verify:
                        stage = json.loads(stage_path.read_text()); assert stage['identity'] == hid and stage['step'] == step
                        assert stage['snapshot'] == artifact(snap)
                    else:
                        if verify:
                            small = torch.load(snap, map_location='cpu', weights_only=False)
                            model, state = parent.method.restore(home)
                            state.update(small); model.load_state_dict(small['model'])
                        else:
                            model, _ = parent.method.fit(x, y, easy, event, v['sites'], v['outer'], v['env'], v['pr'],
                                arm=arm, seed=seed, settings=settings, identity=hid, directory=home,
                                resume=(home/'checkpoint.pt').exists(), stop_at=step,
                                heartbeat=lambda **kw: beat(tag=v['tag'], arm=arm, **kw))
                            state = parent.method.restore(home)[1]
                            assert state['step'] == step
                            method.snapshot(state, snap)
                        before = {k: t.clone() for k, t in model.state_dict().items()}
                        doc = diagnostics(v, x, y, event, model, state, spec, z, target, samples, hid, snap)
                        method.exact_tree(before, model.state_dict())
                        immutable_json(stage_path, doc)
                    stages.append(artifact(stage_path))
                    beat('snapshot_replayed' if verify else 'snapshot_complete', completed_heads=len(refs),
                         tag=v['tag'], arm=arm, step=step, elapsed_seconds=time.monotonic()-started)
                    if pilot:
                        state = parent.method.restore(home)[1]
                        assert state['step'] == step
                        immutable_json(PRIVATE/'pilot.json', dict(snapshots=1, updates=step,
                            train_seconds=state['seconds'], elapsed_seconds=time.monotonic()-started,
                            projected_training_seconds=state['seconds']/step*cfg['training_updates'],
                            projection_excludes_IO_measurement_and_ancestry=True, receipt=stages[-1]))
                        return
                state = parent.method.restore(home)[1]
                source = parent.PRIVATE/'heads'/v['tag']/arm
                prior = json.loads((source/'complete.json').read_text())
                assert prior['identity']['input'] == records[v['tag']]
                assert prior['artifacts']['checkpoint'] == artifact(source/'checkpoint.pt')
                old = parent.method.restore(source)[1]
                trace_match = method.match_final(state, old)
                receipt = dict(identity=hid, snapshots=stages, final_checkpoint=artifact(home/'checkpoint.pt'),
                    original_checkpoint=artifact(source/'checkpoint.pt'), numerical_state_exact=True,
                    trace_match=trace_match, train_seconds=state['seconds'], updates=state['step'])
                immutable_json(complete, receipt)
            train_seconds += receipt['train_seconds']; refs.append(artifact(complete))
            beat('head_replayed' if verify else 'head_complete', completed_heads=len(refs), tag=v['tag'], arm=arm)
    assert len(refs) == cfg['reconstructed_heads']
    immutable_json(PUBLIC/('replay.json' if verify else 'training_freeze.json'), dict(
        registration=artifact(PUBLIC/'registration_lock.json'), receipts=refs,
        heads=len(refs), snapshots=cfg['snapshots'], updates=cfg['training_updates'],
        all_numerical_final_states_exact=True, new_model_variants=0,
        held_readout=False, independent_roles_opened=False, exact_diagnostic_replay=verify))
    if not verify:
        immutable_json(PUBLIC/'compute_receipt.json', dict(train_seconds=train_seconds,
            invocation_seconds=time.monotonic()-started, runtime='native_arm64_torch_cpu',
            torch_version=torch.__version__, threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), num_workers=0, heads=len(refs), updates=cfg['training_updates']))
    beat('replay_complete' if verify else 'training_complete', elapsed_seconds=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', required=True, choices=['register', 'pilot', 'run', 'verify'])
    args = p.parse_args(); PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg, identity = registration(args.phase == 'register')
        if args.phase != 'register': experiment(cfg, identity, args.phase == 'pilot', args.phase == 'verify')


if __name__ == '__main__': main()
