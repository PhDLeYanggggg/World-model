"""Train a cost-aware linear control from verified leave-fold-out forecasts.

No automatic fold assignment, threshold search, calibration, or test evaluation.
"""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-folds', type=int)
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    if min(args.threads, args.batch_size) < 1:
        raise SystemExit('Positive thread and batch counts required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    artifacts = [json.loads(p.read_text()) for p in args.artifacts or []]
    try:
        contract = ExperimentContract(json.loads(args.protocol.read_text()), args.workspace_root, artifacts)
    except ValueError as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc), 'cost_head_training_started': False}))
        return 2
    if args.preflight_only:
        print(json.dumps({'status': 'contract_checked_no_training', 'independence_proven': False}))
        return 0
    if args.fold_models is None or args.baseline is None or args.alpha is None or args.output_dir is None:
        raise SystemExit('Supply explicit fold-models mapping, baseline, alpha, and output-dir')
    mapping = json.loads(args.fold_models.read_text())
    folds = contract.protocol['fit_folds']
    if set(mapping) != {str(f) for f in folds.values()}:
        raise SystemExit('Every fit fold needs an explicit held-fold predictor')
    output = args.output_dir.resolve()
    if not output.is_relative_to(contract.root):
        raise SystemExit('Output must remain inside the experiment workspace')
    run_identity = {'protocol_sha256': contract.digest, 'fold_models': mapping, 'baseline': args.baseline,
                    'alpha': args.alpha, 'device': args.device, 'batch_size': args.batch_size,
                    'code_sha256': file_digest(ROOT / 'src/world_model/m3w_supervised_intervention.py'),
                    'oof_identity_source_sha256': file_digest(ROOT / 'src/world_model/m3w_oof_identity.py'),
                    'script_sha256': file_digest(Path(__file__)),
                    'predictor_sha256': {p: contract.artifacts[p]['sha256'] for p in mapping.values()}}
    identity_path = output / 'run_identity.json'
    if args.resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != run_identity:
            raise SystemExit('Cost-head resume identity changed or missing')
    else:
        output.mkdir(parents=True, exist_ok=False)
        identity_path.write_text(json.dumps(run_identity, indent=2) + '\n')
    import numpy as np
    import torch
    from src.world_model.m3w_supervised_intervention import (
        ContractForecastDataset, fit_linear_gain_harm, load_verified_forecaster, make_oof_cost_rows,
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    groups, reused, began = [], 0, time.monotonic()
    completed_path = output / 'fit_report.json'
    if args.resume and completed_path.exists():
        report = json.loads(completed_path.read_text())
        if file_digest(output / 'linear_cost_head.npz') != report['checkpoint_sha256']:
            raise SystemExit('Completed cost head bytes changed')
        for fold, producer in mapping.items():
            recordings = sorted(r for r, f in folds.items() if str(f) == fold)
            contract.assert_prediction_use(producer, recordings, purpose='oof_risk_training')
            for name in recordings:
                contract.open_recording(name, purpose='fit')
        print(json.dumps({'status': 'already_complete_cached_verified', 'test_evaluated': False}))
        return 0
    for fold, producer in sorted(mapping.items()):
        recordings = sorted(r for r, f in folds.items() if str(f) == fold)
        # Verify the whole held-out fold before opening any label-bearing reader.
        contract.assert_prediction_use(producer, recordings, purpose='oof_risk_training')
        dataset = ContractForecastDataset(contract, recordings, purpose='fit', baseline_name=args.baseline)
        cache, receipt = output / f'fold_{fold}.npz', output / f'fold_{fold}.json'
        if args.resume and receipt.exists():
            metadata = json.loads(receipt.read_text())
            if file_digest(cache) != metadata['cache_sha256'] or metadata['run_identity'] != run_identity:
                raise SystemExit('OOF fold cache content or identity changed')
            with np.load(cache, allow_pickle=False) as arrays:
                group = {**metadata['group_metadata'], 'features': arrays['features'], 'targets': arrays['targets'],
                         'identities': json.loads(str(arrays['identities_json']))}
            reused += 1
            source = 'cached_verified'
        else:
            model = load_verified_forecaster(contract, producer, device=args.device)

            def progress(rows, total):
                item = {'pid': os.getpid(), 'fold': fold, 'processed_rows': rows, 'total_rows': total,
                        'elapsed_seconds': time.monotonic() - began}
                with (output / 'heartbeat.jsonl').open('a') as stream:
                    stream.write(json.dumps(item) + '\n')

            group = make_oof_cost_rows(contract, producer, dataset, model,
                                      batch_size=args.batch_size, device=args.device, progress=progress)
            temporary = cache.with_suffix('.tmp.npz')
            np.savez(temporary, features=group['features'], targets=group['targets'],
                     identities_json=np.array(json.dumps(group['identities'])))
            os.replace(temporary, cache)
            metadata = {'run_identity': run_identity, 'cache_sha256': file_digest(cache),
                        'group_metadata': {k: v for k, v in group.items() if k not in {'features', 'targets', 'identities'}}}
            temporary_receipt = receipt.with_suffix('.tmp')
            temporary_receipt.write_text(json.dumps(metadata, indent=2) + '\n')
            os.replace(temporary_receipt, receipt)
            source = 'fresh_run'
        groups.append(group)
        print(json.dumps({'fold': fold, 'rows': len(group['targets']), 'result_source': source}), flush=True)
        if args.stop_after_folds == len(groups):
            print(json.dumps({'status': 'interrupted_after_complete_fold', 'cost_head_trained': False}))
            return 0
    head = fit_linear_gain_harm(contract, groups, alpha=args.alpha)
    from src.world_model.m3w_oof_identity import oof_feature_identity
    checkpoint = output / 'linear_cost_head.npz'
    temporary = checkpoint.with_suffix('.tmp.npz')
    np.savez(temporary, **{k: head[k] for k in ('mean', 'scale', 'coef', 'intercept')})
    os.replace(temporary, checkpoint)
    report = {k: v for k, v in head.items() if k not in {'mean', 'scale', 'coef', 'intercept'}}
    report.update(result_source='fresh_run', scope='fit_only_OOF_cost_regression_not_policy_evaluation',
                  oof_feature_identity=oof_feature_identity(groups),
                  training_rows=sum(len(g['targets']) for g in groups),
                  oof_folds_reused_cached_verified=reused, oof_total_folds=len(groups),
                  feature_dimension=len(head['mean']), calibrated_policy=False, test_evaluated=False,
                  checkpoint_sha256=file_digest(checkpoint), code_sha256=file_digest(ROOT / 'src/world_model/m3w_supervised_intervention.py'))
    artifact = {'id': output.name, 'kind': 'risk_head', 'path': str(checkpoint.relative_to(contract.root)),
                'sha256': report['checkpoint_sha256'], 'protocol_sha256': contract.digest, 'lineage_complete': True,
                'fit_recordings': head['fit_recordings'], 'selection_recordings': [], 'calibration_recordings': [],
                'parents': head['parents']}
    for name, content in (('artifact.json', artifact), ('fit_report.json', report)):
        temporary = output / (name + '.tmp')
        temporary.write_text(json.dumps(content, indent=2) + '\n')
        os.replace(temporary, output / name)
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
