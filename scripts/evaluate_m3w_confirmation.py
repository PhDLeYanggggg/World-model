"""One frozen final evaluation family; results cannot rank or refit models."""
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
        ExperimentContract, file_digest, claim_confirmation, finish_confirmation, _claim_identity,
    )
    try:
        contract = ExperimentContract(json.loads(args.protocol.read_text()), args.workspace_root,
                                      [json.loads(p.read_text()) for p in args.artifacts or []])
    except ValueError as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc), 'confirmation_evaluated': False}))
        return 2
    if args.preflight_only:
        print(json.dumps({'status': 'contract_checked_no_confirmation', 'confirmation_evaluated': False}))
        return 0
    if args.output_dir is None:
        raise SystemExit('Supply a new final-evaluation output directory')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    from src.evaluation.m3w_confirmation_evaluation import load_family, evaluate_confirmation, implementation_identity
    from src.evaluation.m3w_development_evaluation import content_digest
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rules, recordings, ids, _, _ = load_family(contract, device=args.device)
    output = args.output_dir.resolve()
    if not output.is_relative_to(contract.root):
        raise SystemExit('Output must remain inside the workspace')
    identity = {'protocol_sha256': contract.digest, 'artifacts': contract.artifacts,
                'comparison_artifact_ids': ids, 'implementation_sha256': implementation_identity(),
                'cost_report_sha256': {c['risk_report_path']: file_digest(contract._path(c['risk_report_path']))
                                       for c in rules['comparisons']},
                'output_dir': str(output), 'device': args.device, 'threads': args.threads}
    identity_path = output / 'run_identity.json'
    if args.resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise SystemExit('Final evaluation identity changed or missing')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(identity_path, identity)
    claim_path = contract._path(contract.protocol['confirmation_receipt'])
    run_hash = content_digest(identity)
    if claim_path.exists():
        claim = json.loads(claim_path.read_text())
        if (not args.resume or claim.get('execution_sha256') != run_hash
                or claim.get('implementation_sha256') != implementation_identity()):
            raise SystemExit('Final claim already exists or execution identity changed; no new evaluation allowed')
    else:
        claim = claim_confirmation(claim_path, contract, ids)
        claim.update(execution_sha256=run_hash, implementation_sha256=implementation_identity())
        atomic_json(claim_path, claim)
    cached = {}
    for candidate in rules['comparisons']:
        for recording in recordings:
            key = (candidate['id'], recording)
            stem = content_digest(key)
            receipt = output / (stem + '.receipt.json')
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                cache = output / (stem + '.rows.json')
                if saved['run_sha256'] != run_hash or saved['key'] != list(key) or file_digest(cache) != saved['cache_sha256']:
                    raise SystemExit('Final evaluation cache identity changed')
                cached[key] = json.loads(cache.read_text())
    completion = output / 'completion.json'
    result = output / 'confirmation_report.json'
    claim = json.loads(claim_path.read_text())
    if claim['state'] == 'completed':
        expected = _claim_identity(contract, ids, 'confirmation')
        if (any(claim.get(k) != v for k, v in expected.items())
                or Path(claim['result_path']).resolve() != result or claim['result_sha256'] != file_digest(result)
                or len(cached) != len(rules['comparisons']) * len(recordings)):
            raise SystemExit('Completed final claim/cache/result mismatch')
        stamp = {'run_sha256': run_hash, 'result_sha256': file_digest(result)}
        if completion.exists() and json.loads(completion.read_text()) != stamp:
            raise SystemExit('Final completion marker changed')
        for recording in recordings:
            RecordingWindows(contract._path(contract.protocol['records'][recording]['cache_path']))
        recovered = not completion.exists()
        if recovered:
            atomic_json(completion, stamp)
        print(json.dumps({'status': 'cached_verified_completion_recovered' if recovered else 'cached_verified',
                          'new_confirmation_evaluation': False, 'deployment_approved': False}))
        return 0
    if completion.exists():
        raise SystemExit('Completion marker exists without completed confirmation claim')
    claim_confirmation(claim_path, contract, ids, resume=True)
    began = time.monotonic()

    def save_recording(key, rows):
        stem = content_digest(key)
        cache = output / (stem + '.rows.json')
        atomic_json(cache, rows)
        atomic_json(output / (stem + '.receipt.json'),
                    {'key': list(key), 'run_sha256': run_hash, 'cache_sha256': file_digest(cache)})

    def progress(key, frame, count):
        with (output / 'heartbeat.jsonl').open('a') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'candidate_recording': key, 'frame': frame,
                                     'agent_queries': count, 'elapsed_seconds': time.monotonic() - began}) + '\n')

    report, _ = evaluate_confirmation(contract, claim_path, device=args.device, cached_rows=cached,
                                     on_recording=save_recording, progress=progress)
    report.update(elapsed_seconds=time.monotonic()-began, reused_recording_evaluations=len(cached),
                  runtime={'architecture': platform.machine(), 'device': args.device, 'torch_version': str(torch.__version__),
                           'compute_threads': args.threads, 'interop_threads': 1, 'dataloader_workers': 0})
    atomic_json(result, report)
    finish_confirmation(claim_path, contract, ids, result)
    atomic_json(completion, {'run_sha256': run_hash, 'result_sha256': file_digest(result)})
    print(json.dumps({'status': 'final_family_evaluated', 'families': list(report['families']),
                      'scope': report['scope'], 'model_selection_performed': False, 'deployment_approved': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
