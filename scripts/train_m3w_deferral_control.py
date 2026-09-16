"""Fit the literature-derived deferral control on verified OOF predictions only."""
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


def atomic_json(path, content):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/m3w_independent_experiment.draft.json')
    parser.add_argument('--workspace-root', type=Path, default=ROOT)
    parser.add_argument('--artifacts', type=Path, nargs='+')
    parser.add_argument('--fold-models', type=Path)
    parser.add_argument('--baseline')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch-size', type=int, default=128, help='OOF extraction batch, not training batch')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-folds', type=int)
    parser.add_argument('--stop-after', type=int, help='Interrupt after this completed optimizer step')
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
        print(json.dumps({'status': 'refused', 'reason': str(exc), 'deferral_training_started': False}))
        return 2
    if args.preflight_only:
        print(json.dumps({'status': 'contract_checked_no_training', 'independence_proven': False,
                          'comparator_spec_present': 'cost_sensitive_deferral' in contract.protocol.get('comparators', {})}))
        return 0
    if args.fold_models is None or args.baseline is None or args.seed is None or args.output_dir is None:
        raise SystemExit('Supply explicit fold-models mapping, baseline, seed, output-dir')
    import numpy as np
    import torch
    from src.world_model.m3w_cost_sensitive_deferral import comparator_spec, make_oof_deferral_rows, train_deferral
    from src.world_model.m3w_supervised_intervention import ContractForecastDataset, load_verified_forecaster
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    comparator_spec(contract)
    mapping, folds = json.loads(args.fold_models.read_text()), contract.protocol['fit_folds']
    if set(mapping) != {str(f) for f in folds.values()}:
        raise SystemExit('Every fit fold needs an explicit held-fold predictor')
    if args.seed not in contract.protocol['seeds']:
        raise SystemExit('Protocol-listed seed required')
    output = args.output_dir.resolve()
    if not output.is_relative_to(contract.root):
        raise SystemExit('Output must remain inside experiment workspace')
    identity = {'protocol_sha256': contract.digest, 'fold_models': mapping, 'baseline': args.baseline,
                'seed': args.seed, 'device': args.device, 'batch_size': args.batch_size,
                'source_sha256': file_digest(ROOT / 'src/world_model/m3w_cost_sensitive_deferral.py'),
                'oof_identity_source_sha256': file_digest(ROOT / 'src/world_model/m3w_oof_identity.py'),
                'feature_backend_sha256': file_digest(ROOT / 'src/world_model/m3w_supervised_intervention.py'),
                'script_sha256': file_digest(Path(__file__)),
                'predictor_sha256': {p: contract.artifacts[p]['sha256'] for p in mapping.values()}}
    identity_path = output / 'run_identity.json'
    if args.resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise SystemExit('Deferral resume identity changed or missing')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(identity_path, identity)
    groups, reused, began = [], 0, time.monotonic()
    for fold, producer in sorted(mapping.items()):
        recordings = sorted(r for r, f in folds.items() if str(f) == fold)
        contract.assert_prediction_use(producer, recordings, purpose='oof_risk_training')
        dataset = ContractForecastDataset(contract, recordings, purpose='fit', baseline_name=args.baseline)
        cache, receipt = output / f'fold_{fold}.npz', output / f'fold_{fold}.json'
        if args.resume and receipt.exists():
            metadata = json.loads(receipt.read_text())
            if file_digest(cache) != metadata['cache_sha256'] or metadata['run_identity'] != identity:
                raise SystemExit('OOF deferral cache changed')
            with np.load(cache, allow_pickle=False) as arrays:
                group = {**metadata['group_metadata'], 'features': arrays['features'], 'targets': arrays['targets'],
                         'identities': json.loads(str(arrays['identities_json']))}
            reused += 1
        else:
            model = load_verified_forecaster(contract, producer, device=args.device)

            def progress(rows, total):
                with (output / 'heartbeat.jsonl').open('a') as stream:
                    stream.write(json.dumps({'pid': os.getpid(), 'fold': fold, 'rows': rows, 'total_rows': total,
                                             'elapsed_seconds': time.monotonic() - began}) + '\n')

            group = make_oof_deferral_rows(contract, producer, dataset, model, batch_size=args.batch_size,
                                          device=args.device, progress=progress)
            temporary = cache.with_suffix('.tmp.npz')
            np.savez(temporary, features=group['features'], targets=group['targets'],
                     identities_json=np.array(json.dumps(group['identities'])))
            os.replace(temporary, cache)
            atomic_json(receipt, {'run_identity': identity, 'cache_sha256': file_digest(cache),
                                 'group_metadata': {k: v for k, v in group.items()
                                                    if k not in {'features', 'targets', 'identities'}}})
        groups.append(group)
        if args.stop_after_folds == len(groups):
            print(json.dumps({'status': 'interrupted_after_complete_fold', 'deferral_training_started': False}))
            return 0
    head_dir = output / 'head'
    if args.resume and (head_dir / 'fit_report.json').exists():
        previous = json.loads((head_dir / 'fit_report.json').read_text())
        if previous['training_complete'] and file_digest(head_dir / 'latest.pt') != previous['checkpoint_sha256']:
            raise SystemExit('Completed deferral checkpoint changed')
    report = train_deferral(contract, groups, seed=args.seed, output_dir=head_dir, device=args.device,
                            resume=args.resume and (head_dir / 'latest.pt').exists(), stop_after=args.stop_after)
    report['oof_folds_reused_cached_verified'] = reused
    atomic_json(output / 'fit_report.json', report)
    if report['training_complete']:
        atomic_json(output / 'artifact.json', {
            'id': output.name, 'kind': 'policy', 'family': 'cost_sensitive_deferral',
            'path': str(Path(report['checkpoint']).relative_to(contract.root)),
            'sha256': report['checkpoint_sha256'], 'protocol_sha256': contract.digest, 'lineage_complete': True,
            'parents': report['parents'], 'fit_recordings': report['fit_recordings'],
            'selection_recordings': [], 'calibration_recordings': [],
        })
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
