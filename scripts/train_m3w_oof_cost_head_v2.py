"""Compatibility entrypoint: preserve the frozen trainer, strengthen completion.

Incomplete/new runs use the original training implementation. A completed v2 run
is checked without Torch or fitting. Legacy completions without a trusted v2
receipt are refused, not silently blessed with a receipt made from current bytes.
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
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    # Training options and their validation remain owned by the frozen entrypoint.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/m3w_independent_experiment.draft.json')
    parser.add_argument('--workspace-root', type=Path, default=ROOT)
    parser.add_argument('--artifacts', type=Path, nargs='+')
    parser.add_argument('--fold-models', type=Path)
    parser.add_argument('--baseline')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--stop-after-folds', type=int)
    parser.add_argument('--help', '-h', action='store_true')
    args = parser.parse_args()
    if args.help:
        print(__doc__)
        parser.print_help()
        return 0
    if min(args.batch_size, args.threads) < 1:
        raise SystemExit('Positive thread and batch counts required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    from scripts import train_m3w_oof_cost_head as legacy
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.evaluation.m3w_cost_completion import verify_completed_ridge

    def context():
        contract = ExperimentContract(json.loads(args.protocol.read_text()), args.workspace_root,
            [json.loads(p.read_text()) for p in args.artifacts or []])
        mapping = json.loads(args.fold_models.read_text())
        sources = dict(code_sha256=ROOT / 'src/world_model/m3w_supervised_intervention.py',
            oof_identity_source_sha256=ROOT / 'src/world_model/m3w_oof_identity.py',
            script_sha256=ROOT / 'scripts/train_m3w_oof_cost_head.py')
        expected = dict(protocol_sha256=contract.digest, fold_models=mapping, baseline=args.baseline,
            alpha=args.alpha, device=args.device, batch_size=args.batch_size,
            **{k: file_digest(v) for k, v in sources.items()},
            predictor_sha256={p: contract.artifacts[p]['sha256'] for p in mapping.values()})
        validators = {p: file_digest(ROOT / p) for p in (
            'scripts/train_m3w_oof_cost_head_v2.py', 'src/evaluation/m3w_cost_completion.py',
            'src/evaluation/m3w_experiment_contract.py', 'src/world_model/m3w_oof_identity.py')}
        return contract, sources, expected, validators

    out = args.output_dir.resolve() if args.output_dir is not None else None
    completed = out is not None and (out / 'fit_report.json').is_file()
    if args.resume and completed and not args.preflight_only:
        contract, sources, expected, validators = context()
        receipt_path = out / 'completion_v2.json'
        if not receipt_path.is_file():
            raise SystemExit('Missing trusted completion_v2 receipt; use a separately anchored legacy verification')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('schema_version') != 2 or receipt.get('validation_sources') != validators or
                receipt.get('protocol_sha256') != contract.digest or
                receipt.get('artifact_manifest_digest') != contract._artifact_digest() or
                receipt.get('run_identity') != expected):
            raise SystemExit('Completed v2 validation identity changed')
        result = verify_completed_ridge(contract, out, source_paths=sources, expected_run_identity=expected,
                                         expected_files=receipt['file_bindings'])
        print(json.dumps(dict(status='complete_all_dependencies_verified', training_rows=result['training_rows'],
            files_verified=len(result['file_bindings']), new_training=False, new_inference=False,
            torch_imported='torch' in sys.modules, test_evaluated=False)), flush=True)
        return 0

    status = legacy.main()
    if status == 0 and not args.preflight_only and out is not None and (out / 'fit_report.json').is_file():
        contract, sources, expected, validators = context()
        result = verify_completed_ridge(contract, out, source_paths=sources, expected_run_identity=expected)
        receipt = dict(schema_version=2, validation_sources=validators, protocol_sha256=contract.digest,
            artifact_manifest_digest=contract._artifact_digest(), run_identity=expected,
            file_bindings=result['file_bindings'], origin='verified_immediately_after_training_completion')
        receipt_path = out / 'completion_v2.json'
        if receipt_path.exists():
            raise SystemExit('Refusing to replace existing completion_v2 receipt')
        temporary = receipt_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, receipt_path)
        print(json.dumps(dict(status='training_complete_and_dependencies_bound', training_rows=result['training_rows'])))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
