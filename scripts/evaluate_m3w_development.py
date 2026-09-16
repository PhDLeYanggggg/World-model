"""Evaluate a frozen family on development recordings only; no test access."""
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
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    if args.threads < 1:
        raise SystemExit('Positive thread count required')
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    try:
        contract = ExperimentContract(json.loads(args.protocol.read_text()), args.workspace_root,
                                      [json.loads(p.read_text()) for p in args.artifacts or []])
    except ValueError as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc), 'development_evaluated': False}))
        return 2
    if args.preflight_only:
        print(json.dumps({'status': 'contract_checked_no_evaluation', 'development_evaluated': False}))
        return 0
    if args.plan is None or args.output_dir is None:
        raise SystemExit('Supply an explicit development family and output directory')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    from src.evaluation.m3w_development_evaluation import validate_plan, evaluate_development, content_digest
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    plan = json.loads(args.plan.read_text())
    _, recordings = validate_plan(contract, plan)
    output = args.output_dir.resolve()
    if not output.is_relative_to(contract.root):
        raise SystemExit('Output must remain inside the experiment workspace')
    identity = {'protocol_sha256': contract.digest, 'plan': plan, 'artifacts': contract.artifacts,
                'code_sha256': {p: file_digest(ROOT / p) for p in
                    ('src/evaluation/m3w_development_evaluation.py', 'src/world_model/m3w_supervised_intervention.py',
                     'src/world_model/m3w_cost_sensitive_deferral.py', 'src/world_model/m3w_oof_identity.py',
                     'src/world_model/m3w_joint_intervention.py', 'scripts/evaluate_m3w_development.py')},
                'device': args.device, 'threads': args.threads}
    identity_path = output / 'run_identity.json'
    if args.resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise SystemExit('Development family/implementation identity changed or missing')
    else:
        output.mkdir(parents=True, exist_ok=False)
        # Freeze all candidates before any development labels are accessed.
        atomic_json(identity_path, identity)
    cached = {}
    for c in plan['candidates']:
        for recording in recordings:
            key = (c['id'], recording)
            stem = content_digest(key)
            receipt = output / (stem + '.receipt.json')
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                cache = output / (stem + '.rows.json')
                if saved['run_sha256'] != content_digest(identity) or saved['key'] != list(key) or file_digest(cache) != saved['cache_sha256']:
                    raise SystemExit('Development cache identity changed')
                cached[key] = json.loads(cache.read_text())
    complete = output / 'completion.json'
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt['run_sha256'] != content_digest(identity) or len(cached) != len(plan['candidates']) * len(recordings):
            raise SystemExit('Completed evaluation identity/cache incomplete')
        for path, digest in receipt['artifacts'].items():
            if file_digest(output / path) != digest:
                raise SystemExit('Completed development result changed')
        for recording in recordings:
            contract.open_recording(recording, purpose='development')
        print(json.dumps({'status': 'cached_verified', 'new_development_evaluation': False, 'confirmation_evaluated': False}))
        return 0
    began = time.monotonic()

    def save_recording(key, rows):
        stem = content_digest(key)
        cache = output / (stem + '.rows.json')
        atomic_json(cache, rows)
        atomic_json(output / (stem + '.receipt.json'),
                    {'key': list(key), 'run_sha256': content_digest(identity), 'cache_sha256': file_digest(cache)})

    def progress(key, frame, count):
        with (output / 'heartbeat.jsonl').open('a') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'candidate_recording': key, 'frame': frame,
                                     'agent_queries': count, 'elapsed_seconds': time.monotonic() - began}) + '\n')

    report, _ = evaluate_development(contract, plan, device=args.device, on_recording=save_recording,
                                     cached_rows=cached, progress=progress)
    report.update(result_source='fresh_run', reused_recording_evaluations=len(cached),
                  elapsed_seconds=time.monotonic() - began, scope='development_only_not_confirmatory',
                  runtime={'device': args.device, 'architecture': platform.machine(), 'torch_version': str(torch.__version__),
                           'compute_threads': args.threads, 'interop_threads': 1, 'dataloader_workers': 0},
                  stage5c_executed=False, smc_enabled=False)
    atomic_json(output / 'development_report.json', report)
    selected = report['selection']['selected']
    candidate = next((c for c in plan['candidates'] if c['id'] == selected['candidate_id']), None)
    atomic_json(output / 'selected_policy.json', {'selection': selected, 'candidate': candidate,
                'baseline_name': plan['candidates'][0]['baseline'],
                'protocol_sha256': contract.digest, 'development_report_sha256': file_digest(output / 'development_report.json'),
                'deployment_approved': False, 'calibrated_risk': False})
    artifact = {'id': output.name, 'kind': 'policy', 'path': str((output / 'selected_policy.json').relative_to(contract.root)),
                'sha256': file_digest(output / 'selected_policy.json'), 'protocol_sha256': contract.digest,
                'lineage_complete': True, 'fit_recordings': [], 'selection_recordings': recordings,
                'calibration_recordings': [], 'parents': [candidate['forecaster_id'], candidate['risk_head_id']] if candidate else []}
    atomic_json(output / 'artifact.json', artifact)
    atomic_json(complete, {'run_sha256': content_digest(identity), 'artifacts': {name: file_digest(output / name)
                           for name in ('development_report.json', 'selected_policy.json', 'artifact.json')}})
    print(json.dumps({'selection': report['selection'], 'scope': report['scope'], 'reused_recordings': len(cached)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
