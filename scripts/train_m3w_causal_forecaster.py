"""Fit a clean fixed predictor; no automatic protocol approval or test access."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/m3w_independent_experiment.draft.json')
    parser.add_argument('--workspace-root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/m3w_intervention_backend.json')
    parser.add_argument('--fit-recordings', nargs='+')
    parser.add_argument('--baseline')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import; use arm64 .venv-pytorch')
    if args.threads < 1:
        raise SystemExit('Positive explicit compute thread count required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    try:
        contract = ExperimentContract(json.loads(args.protocol.read_text()), args.workspace_root)
    except ValueError as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc), 'torch_training_started': False}))
        return 2
    if args.preflight_only:
        print(json.dumps({'status': 'contract_checked_no_training', 'protocol_sha256': contract.digest,
                          'independence_proven': False}))
        return 0
    if not args.fit_recordings or args.baseline is None or args.seed is None or args.output_dir is None:
        raise SystemExit('Supply fit-recordings, baseline, protocol-listed seed, and output-dir explicitly')
    import torch
    from src.world_model.m3w_supervised_intervention import ContractForecastDataset, train_forecaster
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.zeros(1, device=args.device)
    config = json.loads(args.config.read_text())
    dataset = ContractForecastDataset(contract, args.fit_recordings, purpose='fit', baseline_name=args.baseline)
    result = train_forecaster(dataset, architecture=config['architecture'],
                              settings={**config['fit_settings'], 'seed': args.seed}, output_dir=args.output_dir,
                              device=args.device, resume=args.resume, stop_after=args.stop_after)
    if result['training_complete']:
        path = Path(result['checkpoint'])
        artifact = {'id': args.output_dir.name, 'kind': 'forecaster',
                    'path': str(path.relative_to(contract.root)), 'sha256': file_digest(path),
                    'protocol_sha256': contract.digest, 'lineage_complete': True,
                    'fit_recordings': args.fit_recordings, 'selection_recordings': [],
                    'calibration_recordings': [], 'parents': []}
        (path.parent / 'artifact.json').write_text(json.dumps(artifact, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'losses'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
