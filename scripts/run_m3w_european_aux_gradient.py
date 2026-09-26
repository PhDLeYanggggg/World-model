"""Registered fitting-only gradient measurements and isolated virtual AdamW steps."""
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
from scripts import run_m3w_european_strong_cap_auxiliary as parent
from src.world_model import m3w_aux_gradient as method
from src.world_model.m3w_native_gain_harm import standardized
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_aux_gradient_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_aux_gradient_v1'
CONFIG = 'configs/m3w_european_aux_gradient_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_aux_gradient.py', 'tests/test_m3w_aux_gradient.py',
         'scripts/run_m3w_european_aux_gradient.py', 'scripts/report_m3w_european_aux_gradient.py',
         str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **kwargs):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    parent.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    _, previous = parent.registration()
    verified = json.loads((parent.PUBLIC/'verification.json').read_text())
    assert verified['all_passed']
    for p, h in verified['artifacts'].items(): assert digest(parent.PUBLIC/p) == h
    for p, h in verified['source_bindings'].items(): assert digest(ROOT/p) == h
    assert cfg['variants'] == list(method.VARIANTS)
    assert cfg['views'] == 144 and cfg['repeats'] == 8 and cfg['checkpoint_step'] == 2000
    assert not any(cfg[k] for k in ('new_full_training', 'new_policy_evaluation', 'held_readout',
        'selection_access', 'reserved_calibration_access', 'confirmation_access', 'deployment_changed',
        'stage5c_executed', 'smc_enabled'))
    identity = dict(parent_registration=artifact(parent.PUBLIC/'registration_lock.json'),
        parent_verification=artifact(parent.PUBLIC/'verification.json'),
        parent_freeze=artifact(parent.PUBLIC/'prediction_freeze.json'),
        create_queue_receipt=artifact(PRIVATE/'create_queue.json'),
        bindings={p: digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity
        parent.risk.base.previous.require_committed(path)
    return cfg, previous


def fitting_inputs(v, record):
    bank, y, _ = parent.risk.banks(v)
    inner, _, _, producer = parent.risk.nested.method.assemble(
        bank, v['raw'], v['cv'], v['sites'], v['outer'], 'oof')
    event = parent.cap.method.event_target(y, inner, v['env'])
    easy = parent.labels(v['cv'], v['pr']['positive_easy_cut'])
    x = v['x']
    for key, value in (('training_ids_sha256', v['ids']), ('training_features_sha256', x),
        ('cost_target_sha256', y), ('easy_sha256', easy), ('cap_target_sha256', event),
        ('row_producer_sha256', producer), ('inner_score_sha256', inner)):
        assert record[key] == array_hash(value)
    parent.method.validate(x, y, easy, event, v['sites'], v['outer'], v['env'], v['pr'])
    return x, y, event


def single_view(v, record, cfg):
    x, y, event = fitting_inputs(v, record)
    seed = int(v['g']['group'].split('_seed')[1].split('_')[0])
    z = torch.from_numpy(standardized(x, v['pr']))
    target = torch.from_numpy(np.nan_to_num(y/v['pr']['cost_scale'], nan=0.).astype(np.float32))
    env = torch.tensor(v['env']/v['pr']['cost_scale'], dtype=torch.float32)
    true = torch.tensor(event, dtype=torch.float32)
    shuffle = torch.tensor(parent.method.auxiliary_target(event, v['sites'], 'shuffled_aux', seed), dtype=torch.float32)
    samples = [method.sample_rows(v['pr']['known'], v['sites'], seed+cfg['sampling_seed_offset'], r,
        cfg['batch_size'], cfg['probe_rows_per_locality']) for r in range(cfg['repeats'])]
    data = dict(tag=v['tag'], group=v['g']['group'], producer=v['g']['producer'], controller=v['g']['controller'],
        seed=seed, pair=v['pair'], excluded_locality=v['outer'], input=record, states=[], initial=None,
        result_source='fresh_run_fitting_only_virtual_adamw', inherited_checkpoints='cached_verified',
        outer_outcomes_read=False, new_fully_trained_models=0)
    reference = None
    for arm in cfg['checkpoint_arms']:
        directory = parent.PRIVATE/'heads'/v['tag']/arm
        receipt = json.loads((directory/'complete.json').read_text())
        assert receipt['identity']['input'] == record and receipt['identity']['seed'] == seed
        for a in receipt['artifacts'].values(): assert artifact(ROOT/a['path']) == a
        model, state = parent.method.restore(directory)
        assert state['step'] == cfg['checkpoint_step']
        for key in ('mean', 'std', 'known', 'weights'):
            np.testing.assert_array_equal(state['preprocess'][key], v['pr'][key])
        assert state['preprocess']['cost_scale'] == v['pr']['cost_scale']
        if reference is None: reference = state
        else:
            parent.match_state(state, reference)
            for k in state['initial_model']: assert torch.equal(state['initial_model'][k], reference['initial_model'][k])
        shared = [i for i, (name, _) in enumerate(model.named_parameters()) if name.startswith(cfg['shared_parameter_prefix'])]
        full = list(range(len(list(model.parameters()))))
        assert len(shared) == 2
        scales = torch.tensor(state['loss_scales'], dtype=torch.float32)
        if data['initial'] is None:
            initial = parent.method.original.AuxiliaryCostHead(x.shape[1], state['settings']['width'])
            initial.load_state_dict(state['initial_model'])
            batch = samples[0][0]
            g, info = method.gradients(initial, z[batch], env[batch], target[batch], true[batch], scales)
            data['initial'] = dict(shared=method.geometry(method.flat(g[0], shared), method.flat(g[2], shared)),
                                   full=method.geometry(method.flat(g[0], full), method.flat(g[2], full)), losses=info)
        rows = []
        for repeat, (batch, probes) in enumerate(samples):
            task = {}; gradients = {}
            for name, labels in [('true', true), ('shuffled', shuffle)]:
                g, losses = method.gradients(model, z[batch], env[batch], target[batch], labels[batch], scales)
                gradients[name] = g
                task[name] = dict(losses=losses)
                for metric, index in [('cost4', 0), ('easy_harm_positive', 1)]:
                    for subset, indices in [('shared', shared), ('full', full)]:
                        task[name][metric+'_'+subset] = method.geometry(method.flat(g[index], indices), method.flat(g[2], indices))
            for a, b in zip(gradients['true'][0], gradients['shuffled'][0]): assert torch.equal(a, b)
            joined = np.concatenate(list(probes.values()))
            before = method.probe_losses(model, z[joined], env[joined], target[joined], scales)
            local_before = {site: method.probe_losses(model, z[ids], env[ids], target[ids], scales)
                            for site, ids in probes.items()}
            comparisons = {}
            for variant in cfg['variants']:
                name = 'shuffled' if 'shuffled' in variant else 'true'
                g = gradients[name]
                updated, norm = method.virtual_step(model, state['optimizer'], state['settings'], g[0], g[2], shared, variant)
                comparisons[variant] = dict(gradient_norm_before_clip=norm,
                    joined=method.probe_losses(updated, z[joined], env[joined], target[joined], scales),
                    localities={site: method.probe_losses(updated, z[ids], env[ids], target[ids], scales)
                                for site, ids in probes.items()})
            rows.append(dict(repeat=repeat, update_ids_sha256=array_hash(v['ids'][batch]),
                probe_ids_sha256={site: array_hash(v['ids'][ids]) for site, ids in probes.items()},
                update_unknown_rows=int((~v['pr']['known'][batch]).sum()),
                row_overlap_count=int(np.intersect1d(batch, joined).size),
                gradients=task, before=before, local_before=local_before, after=comparisons))
        for k in state['model']: assert torch.equal(model.state_dict()[k], state['model'][k])
        data['states'].append(dict(arm=arm, checkpoint=artifact(directory/'checkpoint.pt'), repeats=rows))
    return data


def experiment(cfg, previous, pilot=False, verify=False):
    parent.check_sources(previous)
    records = {r['tag']: r['input'] for r in json.loads((parent.PUBLIC/'support_report.json').read_text())['rows']}
    refs = []; started = time.monotonic()
    for v in parent.views(previous):
        if shutil.disk_usage(PRIVATE).free < cfg['minimum_free_gib']*2**30:
            raise OSError('Preserve 10GiB free; resume from atomic per-view receipts')
        path = PRIVATE/'views'/(v['tag']+'.json')
        if path.exists() and not verify:
            old = json.loads(path.read_text())
            assert old['registration'] == artifact(PUBLIC/'registration_lock.json')
            assert old['input'] == records[v['tag']]
            for s in old['states']: assert artifact(ROOT/s['checkpoint']['path']) == s['checkpoint']
        else:
            if verify and not path.exists(): raise ValueError('Missing completed diagnostic')
            doc = single_view(v, records[v['tag']], cfg)
            doc['registration'] = artifact(PUBLIC/'registration_lock.json')
            immutable_json(path, doc)
        refs.append(artifact(path))
        beat('view_replayed' if verify else 'view_complete', views=len(refs), tag=v['tag'],
             elapsed_seconds=time.monotonic()-started)
        if pilot:
            immutable_json(PRIVATE/'pilot.json', dict(views=1, virtual_updates=120,
                elapsed_seconds=time.monotonic()-started, excludes_ancestry_preflight=True, receipt=refs[-1]))
            return
    assert len(refs) == cfg['views']
    immutable_json(PUBLIC/('replay.json' if verify else 'diagnostic_freeze.json'), dict(
        registration=artifact(PUBLIC/'registration_lock.json'), receipts=refs,
        views=len(refs), frozen_checkpoint_states=432, final_batch_diagnostics=3456,
        initial_checks=144, virtual_adamw_updates=17280, new_full_training_runs=0,
        held_readout=False, exact_replay=verify))
    beat('replay_complete' if verify else 'diagnostic_complete', elapsed_seconds=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', required=True, choices=['register', 'pilot', 'run', 'verify'])
    args = p.parse_args(); PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg, previous = registration(args.phase == 'register')
        if args.phase != 'register': experiment(cfg, previous, args.phase == 'pilot', args.phase == 'verify')


if __name__ == '__main__': main()
