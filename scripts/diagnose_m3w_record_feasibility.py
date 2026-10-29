"""Exact frozen-state diagnosis on fitting recordings only, never held scoring."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_multiaction_support_scale as wrapper
from src.evaluation.m3w_record_gradient_diagnosis import gradient_components, reference_consistency

NAME = 'european_record_feasibility_audit_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
REG = PUBLIC/'registration.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify')); args = parser.parse_args()
    wrapper.configure(); run = wrapper.run
    cfg, pc, reg, _, source, jobs, identity, parents = run.admission(); learned = run.models()
    files = (Path(__file__), ROOT/'src/evaluation/m3w_record_gradient_diagnosis.py',
        ROOT/'tests/test_m3w_record_gradient_diagnosis.py', PUBLIC/'protocol.md',
        wrapper.PUBLIC/'training_freeze.json', wrapper.PUBLIC/'summary.json', wrapper.PUBLIC/'verification.json')
    run.once(REG, dict(bindings={str(p.relative_to(ROOT)):run.sha(p) for p in files},
        contexts=sorted(reg['contexts']), seeds=cfg['seeds'], banks=['physical_only', 'physical_neural'],
        frozen_states=36, optimizer_updates=0, scope='original_fitting_recordings_only',
        inner_held_opened=False, independent_roles_opened=False, stage5c_executed=False, smc_enabled=False))
    if args.phase == 'register':
        return
    for rel, digest in json.loads(REG.read_text())['bindings'].items():
        if run.sha(ROOT/rel) != digest:
            raise ValueError('Frozen diagnostic input changed')
        run.parent.committed(ROOT/rel)
    run.parent.committed(REG)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def beat(**event):
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        run.json_write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows = []
        for context, items in sorted(reg['contexts'].items()):
            packets = run.context_packets(context, items, source, jobs, identity, beat)['fit']
            for seed in cfg['seeds']:
                for arm in ('physical_only', 'physical_neural'):
                    data = run.prepare(context, seed, arm, packets, parents[context, seed, arm])
                    ref = learned[context, seed, arm]['checkpoint']
                    state = run.api.read_checkpoint(ROOT/ref['path'])
                    row = dict(context=context, seed=seed, arm=arm, checkpoint_sha256=ref['sha256'],
                        gradient=gradient_components(state, data), sources=[])
                    cost = run.api.read_checkpoint(ROOT/parents[context, seed, arm]['checkpoint']['path'])
                    for name, packet in sorted(packets.items()):
                        features, anchor, _, _, prediction = run.causal_inputs(packet, cost, arm)
                        choice, _ = wrapper.predict(state, features, anchor, packet['allowed'], arm=arm)
                        row['sources'].append(dict(source=name,
                            reference_consistency=reference_consistency(prediction, packet['y'], choice),
                            compressed_feature_clip_fraction=np.mean(np.abs(features) > 8, axis=(0, 1)).tolist()))
                    rows.append(row); beat(state='frozen_state_diagnosed', context=context, seed=seed, arm=arm,
                        states=len(rows), total=36)
                    del data
            del packets
        result = dict(rows=rows, registration_sha256=run.sha(REG),
            result_source='fresh_frozen_fitting_state_calculation_cached_verified_weights',
            no_optimizer_update=True, no_held_score=True, deployment_changed=False,
            stage5c_executed=False, smc_enabled=False)
        path = PUBLIC/'diagnosis.json'
        if args.phase == 'verify':
            if result != json.loads(path.read_text()):
                raise ValueError('Exact frozen-state diagnosis differs')
            run.once(PUBLIC/'verification.json', dict(exact_replay=True, states=36, diagnosis_sha256=run.sha(path)))
        else:
            run.once(path, result)
            run.once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-started,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, states=36,
                diagnosis_sha256=run.sha(path), registration_sha256=run.sha(REG)))
        run.json_write(PRIVATE/('verification_runtime.json' if args.phase == 'verify' else 'runtime.json'),
            dict(pid=os.getpid(), seconds=time.monotonic()-started))
        beat(state='diagnosis_verified' if args.phase == 'verify' else 'diagnosis_complete')


if __name__ == '__main__':
    main()
