"""Screen frozen development-selected policies on scene-held-out calibration data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/m3w_independent_experiment.draft.json')
    parser.add_argument('--workspace-root', type=Path, default=ROOT)
    parser.add_argument('--artifacts', type=Path, nargs='+')
    parser.add_argument('--policy-ids', nargs='+')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    if args.threads < 1:
        raise SystemExit('Positive explicit compute threads required')
    from src.evaluation.m3w_experiment_contract import (
        ExperimentContract, file_digest, claim_calibration, finish_calibration, _claim_identity,
    )
    try:
        contract = ExperimentContract(json.loads(args.protocol.read_text()), args.workspace_root,
                                      [json.loads(p.read_text()) for p in args.artifacts or []])
    except ValueError as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc), 'calibration_evaluated': False}))
        return 2
    if args.preflight_only:
        print(json.dumps({'status': 'contract_checked_no_calibration', 'calibration_evaluated': False}))
        return 0
    if not args.policy_ids or args.output_dir is None:
        raise SystemExit('Supply frozen policy IDs in development-priority order and an output directory')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    from src.evaluation.m3w_risk_calibration import validate_rules, load_frozen_policy, evaluate_calibration
    from src.evaluation.m3w_development_evaluation import content_digest
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rules, recordings = validate_rules(contract)
    if args.policy_ids != rules['policy_priority']:
        raise SystemExit('Policy order differs from approved calibration priority')
    for name in args.policy_ids:
        load_frozen_policy(contract, name, device=args.device)
    output = args.output_dir.resolve()
    if not output.is_relative_to(contract.root):
        raise SystemExit('Output must remain inside the workspace')
    if output.name in contract.artifacts:
        raise SystemExit('Calibrator artifact ID collides with an existing producer')
    identity = {'protocol_sha256': contract.digest, 'policy_order': args.policy_ids, 'artifacts': contract.artifacts,
                'code_sha256': {p: file_digest(ROOT / p) for p in
                    ('src/evaluation/m3w_risk_calibration.py', 'src/evaluation/m3w_experiment_contract.py',
                     'src/evaluation/m3w_development_evaluation.py', 'src/world_model/m3w_supervised_intervention.py',
                     'src/world_model/m3w_joint_intervention.py', 'src/data_unification/m3w_causal_recordings.py',
                     'scripts/calibrate_m3w_intervention.py')}, 'device': args.device, 'threads': args.threads}
    identity_path = output / 'run_identity.json'
    if args.resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise SystemExit('Frozen calibration order/protocol/implementation changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(identity_path, identity)
    cached = {}
    for name in args.policy_ids:
        for recording in recordings:
            key = (name, recording)
            stem = content_digest(key)
            receipt = output / (stem + '.receipt.json')
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                cache = output / (stem + '.rows.json')
                if (saved['run_sha256'] != content_digest(identity) or saved['key'] != list(key)
                        or file_digest(cache) != saved['cache_sha256']):
                    raise SystemExit('Calibration cache identity changed')
                cached[key] = json.loads(cache.read_text())
    claim_path = contract._path(contract.protocol['calibration_receipt'])
    completion = output / 'completion.json'

    def calibrator_artifact():
        return {'id': output.name, 'kind': 'calibrator',
                'path': str((output / 'calibration_report.json').relative_to(contract.root)),
                'sha256': file_digest(output / 'calibration_report.json'), 'protocol_sha256': contract.digest,
                'lineage_complete': True, 'fit_recordings': [], 'selection_recordings': [],
                'calibration_recordings': recordings, 'parents': args.policy_ids}

    def verify_finished_claim():
        claim = json.loads(claim_path.read_text())
        expected = _claim_identity(contract, args.policy_ids, 'calibration')
        if (any(claim.get(k) != v for k, v in expected.items()) or claim['state'] != 'completed'
                or Path(claim['result_path']).resolve() != output / 'calibration_report.json'
                or claim['result_sha256'] != file_digest(output / 'calibration_report.json')
                or json.loads((output / 'artifact.json').read_text()) != calibrator_artifact()):
            raise SystemExit('Completed calibration receipt/lineage disagrees with result')

    if completion.exists():
        complete = json.loads(completion.read_text())
        if complete['run_sha256'] != content_digest(identity) or len(cached) != len(args.policy_ids) * len(recordings):
            raise SystemExit('Completed calibration cache/identity is incomplete')
        for relative, digest in complete['artifacts'].items():
            item = (output / relative).resolve()
            if not item.is_relative_to(output) or file_digest(item) != digest:
                raise SystemExit('Completed calibration result changed')
        verify_finished_claim()
        for recording in recordings:
            # Verify the bytes without reopening a completed calibration for labels.
            RecordingWindows(contract._path(contract.protocol['records'][recording]['cache_path']))
        print(json.dumps({'status': 'cached_verified', 'new_calibration_evaluation': False, 'confirmation_evaluated': False}))
        return 0
    # A crash after writing the result/finishing the receipt can resume without re-evaluation.
    if args.resume and claim_path.is_file() and json.loads(claim_path.read_text())['state'] == 'completed':
        verify_finished_claim()
        if len(cached) != len(args.policy_ids) * len(recordings):
            raise SystemExit('Finished calibration has incomplete recording receipts')
        atomic_json(completion, {'run_sha256': content_digest(identity),
                    'artifacts': {name: file_digest(output / name) for name in ('calibration_report.json', 'artifact.json')}})
        print(json.dumps({'status': 'cached_verified_completion_recovered', 'new_calibration_evaluation': False}))
        return 0
    claim_calibration(claim_path, contract, args.policy_ids, resume=args.resume and claim_path.exists())
    began = time.monotonic()

    def save_recording(key, rows):
        stem = content_digest(key)
        cache = output / (stem + '.rows.json')
        atomic_json(cache, rows)
        atomic_json(output / (stem + '.receipt.json'),
                    {'key': list(key), 'run_sha256': content_digest(identity), 'cache_sha256': file_digest(cache)})

    def progress(key, frame, count):
        with (output / 'heartbeat.jsonl').open('a') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'policy_recording': key, 'frame': frame,
                                     'agent_queries': count, 'elapsed_seconds': time.monotonic() - began}) + '\n')

    report, _ = evaluate_calibration(contract, args.policy_ids, claim_path, device=args.device,
                                     cached_rows=cached, on_recording=save_recording, progress=progress)
    report.update(result_source='fresh_run', reused_recording_evaluations=len(cached),
                  elapsed_seconds=time.monotonic() - began, scope=contract.protocol['scope'],
                  runtime={'device': args.device, 'architecture': platform.machine(), 'torch_version': str(torch.__version__),
                           'compute_threads': args.threads, 'interop_threads': 1, 'dataloader_workers': 0})
    result = output / 'calibration_report.json'
    atomic_json(result, report)
    atomic_json(output / 'artifact.json', calibrator_artifact())
    finish_calibration(claim_path, contract, args.policy_ids, result)
    atomic_json(completion, {'run_sha256': content_digest(identity),
                            'artifacts': {name: file_digest(output / name) for name in ('calibration_report.json', 'artifact.json')}})
    print(json.dumps({k: report[k] for k in ('selected_policy_id', 'use_unchanged_baseline',
                  'physical_scene_count_declared', 'independence_verified', 'deployment_approved')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
