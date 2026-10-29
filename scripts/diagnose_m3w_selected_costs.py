"""Frozen fitting-record cost calibration; no new training or held readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_static_scene_cost as scene
from scripts import report_m3w_feature_cost_evidence as aggregates
from src.evaluation.m3w_selected_cost_diagnosis import audit_record
from src.evaluation.m3w_multiaction_cost_readout import compact_score
import numpy as np

parent = scene.parent
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selected_cost_audit_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selected_cost_audit_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_selected_cost_diagnosis.py',
         ROOT/'tests/test_m3w_selected_cost_diagnosis.py', PUBLIC/'protocol.md')


def admission(register=False):
    values = scene.admission(); models = scene.models()
    for folder in (parent.PUBLIC, scene.PUBLIC):
        check = json.loads((folder/'verification.json').read_text())
        if not check['exact_replay'] or check['summary_sha256'] != parent.sha(folder/'summary.json'):
            raise ValueError('Verified frozen original and scene readouts required')
    parent.once(REG, dict(bindings={str(p.relative_to(ROOT)):parent.sha(p) for p in FILES},
        original_freeze_sha256=parent.sha(parent.PUBLIC/'training_freeze.json'),
        scene_freeze_sha256=parent.sha(scene.PUBLIC/'training_freeze.json'),
        scene_verification_sha256=parent.sha(scene.PUBLIC/'verification.json'),
        states=108, families=['no_scene', 'scene'], scope='original_fitting_recordings_only',
        optimizer_updates=0, independent_roles_opened=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if parent.sha(ROOT/rel) != digest:
                raise ValueError('Frozen diagnosis code changed')
            parent.committed(ROOT/rel)
        parent.committed(REG)
    return values, models


def main(phase):
    (cfg, pc, reg, source, jobs, identity, original, packed, polygons), models = admission(phase=='register')
    if phase == 'register':
        return
    scene.torch.set_num_threads(4); scene.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    def beat(**event):
        if scene.shutil.disk_usage(ROOT).free < 10*2**30:
            raise OSError('Preserve10GiB reserve')
        if time.monotonic()-started > 43200:
            raise TimeoutError('12h healthy-runtime boundary')
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        parent.json_write(PRIVATE/'heartbeat.json', value); print(json.dumps(value), flush=True)
    expected = dict(no_scene=aggregates.aggregate_groups(parent.NAME), scene=aggregates.aggregate_groups(scene.NAME))
    records, states = [], 0
    for context, items in sorted(reg['contexts'].items()):
        packets = scene.scene_packets(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=False),
            source, packed, polygons)['fit']
        for seed in cfg['seeds']:
            for arm, actions in parent.api.ARMS.items():
                for family, endpoints in (('no_scene', original), ('scene', models)):
                    state = parent.api.read_checkpoint(ROOT/endpoints[context, seed, arm]['checkpoint']['path'])
                    for name, packet in sorted(packets.items()):
                        if family == 'scene':
                            p = scene.api.predict(state, packet['x'], packet['scene'], packet['envelope'], packet['scales'])
                        else:
                            p = parent.api.predict(state, packet['x'], packet['envelope'], packet['scales'])
                        choice, _ = parent.api.select_actions(p, packet['allowed'], arm=arm)
                        old = expected[family][f'{context}_{name}_seed{seed}_fit.json']
                        if parent.fingerprint(choice) != old['action_sha256'][arm]:
                            raise ValueError('Frozen primary choice changed')
                        for recording in sorted(set(packet['recordings'])):
                            at = packet['recordings'] == recording
                            safety = compact_score(packet['y'][at], packet['envelope'][at], choice[at])
                            if safety != old['result']['recordings'][str(recording)][arm]:
                                raise ValueError('Frozen per-record score differs')
                            value = audit_record(p[at], packet['y'][at], packet['allowed'][at], choice[at], packet['scales'][at], actions)
                            records.append(dict(family=family, context=context, source=name, seed=seed, arm=arm,
                                recording=str(recording), diagnosis=value, original_score=safety,
                                checkpoint_sha256=endpoints[context, seed, arm]['checkpoint']['sha256']))
                    states += 1; beat(state='state_complete', states=states, total=108, family=family, context=context, seed=seed, arm=arm)
        del packets
    if states != 108 or len(records) != 2664:
        raise ValueError('Complete108state/2664record-view diagnosis required')
    result = dict(result_source='fresh_frozen_TRAIN_prediction_diagnosis', states=states,
        record_views=len(records), records=records, optimizer_updates=0, labels_used_for_inference=False,
        independent_roles_opened=False, deployment_changed=False, registration_sha256=parent.sha(REG))
    if phase == 'verify':
        if result != json.loads((PUBLIC/'diagnosis.json').read_text()):
            raise ValueError('Exact selected-cost diagnosis replay differs')
        parent.once(PUBLIC/'verification.json', dict(exact_replay=True, states=108,
            diagnosis_sha256=parent.sha(PUBLIC/'diagnosis.json')))
    else:
        parent.once(PUBLIC/'diagnosis.json', result)
    parent.json_write(PRIVATE/(phase+'_runtime.json'), dict(pid=os.getpid(), seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='diagnosis_verified' if phase=='verify' else 'diagnosis_complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'run', 'verify'))
    args = parser.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); main(args.phase)
