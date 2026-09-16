"""Train a neural benefit/harm head from the ridge control's verified OOF cache.

No new split, task, risk budget, calibration or final-set selection is supplied.
"""
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
    parser.add_argument('--artifacts', type=Path, nargs='+')
    parser.add_argument('--oof-cache-dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    if args.threads < 1:
        raise SystemExit('Positive thread count required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    from src.evaluation.m3w_experiment_contract import ExperimentContract
    try:
        contract = ExperimentContract(json.loads(args.protocol.read_text()), args.workspace_root,
                                      [json.loads(p.read_text()) for p in args.artifacts or []])
    except ValueError as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc), 'neural_cost_training_started': False}))
        return 2
    if args.preflight_only:
        print(json.dumps({'status': 'contract_checked_no_training', 'neural_cost_training_started': False}))
        return 0
    if args.seed is None or args.oof_cache_dir is None or args.output_dir is None:
        raise SystemExit('Supply explicit seed, verified OOF cache and output directory')
    import torch
    from src.world_model.m3w_neural_gain_harm import read_verified_oof_cache, train_neural_gain_harm
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    groups = read_verified_oof_cache(contract, args.oof_cache_dir)
    report = train_neural_gain_harm(contract, groups, seed=args.seed, output_dir=args.output_dir,
                                   device=args.device, resume=args.resume, stop_after=args.stop_after)
    if report['training_complete']:
        output = args.output_dir.resolve()
        artifact = {'id': output.name, 'kind': 'risk_head', 'family': 'neural_gain_harm',
                    'path': str(Path(report['checkpoint']).relative_to(contract.root)),
                    'sha256': report['checkpoint_sha256'], 'protocol_sha256': contract.digest,
                    'lineage_complete': True, 'fit_recordings': report['fit_recordings'],
                    'selection_recordings': [], 'calibration_recordings': [], 'parents': report['parents']}
        temporary = output / 'artifact.tmp'
        temporary.write_text(json.dumps(artifact, indent=2) + '\n')
        os.replace(temporary, output / 'artifact.json')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
