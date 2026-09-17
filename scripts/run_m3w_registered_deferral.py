"""Fit and evaluate the registered v7 diagnostic deferral grid. Never select a winner."""
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
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-fit', action='store_true', help='Resumable first real fit smoke check')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '4'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import numpy as np
    import torch
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.evaluation.m3w_development_evaluation import (
        content_digest, validate_plan, _load_cost_head, scene_requests, score_scene,
    )
    from src.evaluation.m3w_registered_deferral import (
        atomic_json, registered_spec, fit_registered, load_registered, infer_scene, matched_rows, summarize_registered,
    )
    from src.world_model.m3w_cost_sensitive_deferral import make_oof_deferral_rows, validate_groups
    from src.world_model.m3w_supervised_intervention import ContractForecastDataset, load_verified_forecaster
    from src.world_model.m3w_oof_identity import oof_feature_identity
    from scripts.evaluate_m3w_forecast_supplement import verify_completed_evaluation
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    registration_path, output = args.registration.resolve(), args.output.resolve()
    if not registration_path.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError('Workspace-local registration/output required')
    registration = json.loads(registration_path.read_text())
    protocol = json.loads((ROOT / registration['parent_protocol']).read_text())
    reference = {'path': str(registration_path.relative_to(ROOT)), 'sha256': file_digest(registration_path)}
    identity = {'registration': reference, 'device': 'cpu', 'threads': 4, 'interop_threads': 1,
                'torch_version': str(torch.__version__)}
    if args.resume:
        if json.loads((output / 'run_identity.json').read_text()) != identity:
            raise ValueError('Supplement run identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / 'run_identity.json', identity)
    began, results, receipts = time.monotonic(), {}, {}

    def heartbeat(state, **fields):
        atomic_json(output / 'heartbeat.json', {'pid': os.getpid(), 'state': state,
                    'elapsed_seconds_this_invocation': time.monotonic() - began, **fields})

    for family, directory in registration['families'].items():
        study = ROOT / directory
        results[family] = {}
        for seed in protocol['seeds']:
            key, target = f'{family}_seed{seed}', output / f'{family}_seed{seed}'
            target.mkdir(exist_ok=True)
            primary = study / f'seed{seed}_development'
            verify_completed_evaluation(primary, file_digest)
            old_identity = json.loads((primary / 'run_identity.json').read_text())
            # Require all original source bytes, not merely a matching metrics JSON.
            for path, digest in old_identity['code_sha256'].items():
                if file_digest(ROOT / path) != digest:
                    raise ValueError('Primary source changed: ' + path)
            artifacts = list(old_identity['artifacts'].values())
            contract = ExperimentContract(protocol, ROOT, artifacts)
            plan = json.loads((study / f'seed{seed}_development_plan.json').read_text())
            if (old_identity['plan'] != plan or old_identity['protocol_sha256'] != contract.digest
                    or old_identity['device'] != 'cpu' or old_identity['threads'] != 4):
                raise ValueError('Primary family/device mismatch')
            rules, recordings = validate_plan(contract, plan)
            if len({c['forecaster_id'] for c in plan['candidates']}) != 1:
                raise ValueError('One frozen forecaster per seed required')
            for candidate in plan['candidates']:
                _load_cost_head(contract, candidate)
            for variant in registration['variants']:
                registered_spec(contract, {**reference, 'variant': variant})
            mapping = json.loads((study / f'seed{seed}_fold_models.json').read_text())
            baseline = plan['candidates'][0]['baseline']
            if set(mapping) != {str(f) for f in protocol['fit_folds'].values()}:
                raise ValueError('Incomplete OOF producer mapping')
            binding = {'run': identity, 'primary': old_identity, 'mapping': mapping}
            binding_sha = content_digest(binding)
            groups = []
            for fold, producer in sorted(mapping.items()):
                cache, receipt = target / f'fold{fold}.npz', target / f'fold{fold}.json'
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved['binding_sha256'] != binding_sha or saved['cache_sha256'] != file_digest(cache):
                        raise ValueError('OOF cached data changed')
                    with np.load(cache, allow_pickle=False) as arrays:
                        group = {**saved['metadata'], 'features': arrays['features'], 'targets': arrays['targets'],
                                 'identities': json.loads(str(arrays['identities']))}
                else:
                    held = sorted(r for r, f in protocol['fit_folds'].items() if str(f) == fold)
                    dataset = ContractForecastDataset(contract, held, purpose='fit', baseline_name=baseline)
                    predictor = load_verified_forecaster(contract, producer, device='cpu')
                    group = make_oof_deferral_rows(contract, producer, dataset, predictor, batch_size=128, device='cpu',
                        progress=lambda n, total: heartbeat('extracting_OOF', family=family, seed=seed, fold=fold, rows=n, total=total))
                    tmp = cache.with_suffix('.tmp.npz')
                    np.savez(tmp, features=group['features'], targets=group['targets'], identities=np.array(json.dumps(group['identities'])))
                    os.replace(tmp, cache)
                    atomic_json(receipt, {'binding_sha256': binding_sha, 'cache_sha256': file_digest(cache),
                        'metadata': {k: v for k, v in group.items() if k not in ('features', 'targets', 'identities')}})
                groups.append(group)
            validate_groups(contract, groups)
            fingerprint = oof_feature_identity(groups)
            for candidate in plan['candidates']:
                risk_report = json.loads((ROOT / candidate['risk_report_path']).read_text())
                if risk_report['oof_feature_identity'] != fingerprint:
                    raise ValueError('Deferral and original cost-head OOF feature rows differ')
            heads, fit_reports = {}, {}
            for variant in registration['variants']:
                head_output = target / variant
                heartbeat('training_deferral', family=family, seed=seed, variant=variant)
                report = fit_registered(contract, groups, reference={**reference, 'variant': variant}, seed=seed,
                                        output=head_output, resume=(head_output / 'latest.pt').exists())
                fit_reports[variant] = report
                heads['deferral_' + variant] = load_registered(contract, report)
                print(json.dumps({'family': family, 'seed': seed, 'variant': variant, 'fit_steps': report['steps'],
                                  'OOF_rows': report['training_rows'], 'features_exactly_matched': True}), flush=True)
                if args.stop_after_fit:
                    heartbeat('paused_after_completed_fit', family=family, seed=seed, variant=variant)
                    return 0
            full = load_verified_forecaster(contract, plan['candidates'][0]['forecaster_id'], device='cpu')
            state = torch.load(ROOT / contract.artifacts[plan['candidates'][0]['forecaster_id']]['path'], weights_only=True)
            if state['identity']['settings']['seed'] != seed:
                raise ValueError('Final forecaster seed mismatch')
            eval_binding = content_digest({'binding': binding, 'heads': {
                v: r['checkpoint_sha256'] for v, r in fit_reports.items()}})
            per_candidate = {c['id']: [] for c in plan['candidates']}
            scoring_sources = {}
            for recording in recordings:
                cache, receipt = target / f'{recording}.rows.json', target / f'{recording}.receipt.json'
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved != {'binding_sha256': eval_binding, 'cache_sha256': file_digest(cache)}:
                        raise ValueError('Deferral scoring cache changed')
                    fresh = json.loads(cache.read_text())
                    scoring_source = 'cached_verified'
                else:
                    reader, _ = contract.open_recording(recording, purpose='development')
                    fresh = []
                    for number, scene in enumerate(scene_requests(reader, protocol['task'], stride=rules['query_stride'])):
                        # Every head is fitted/verified, and all decisions precede label retrieval.
                        decision = infer_scene(scene, full, heads, baseline=baseline, geometry=rules['geometry_by_recording'][recording])
                        labels = reader.get_scene_labels(scene)
                        fresh.extend(score_scene(scene, decision, labels, label_policy=rules['label_policy']))
                        if number % 50 == 0:
                            heartbeat('scoring_development', family=family, seed=seed, recording=recording, query=number)
                    atomic_json(cache, fresh)
                    atomic_json(receipt, {'binding_sha256': eval_binding, 'cache_sha256': file_digest(cache)})
                    scoring_source = 'fresh_run'
                scoring_sources[recording] = scoring_source
                for candidate in plan['candidates']:
                    old_key = [candidate['id'], recording]
                    old_cache = primary / (content_digest(old_key) + '.rows.json')
                    old_receipt = json.loads((primary / (content_digest(old_key) + '.receipt.json')).read_text())
                    if old_receipt != {'key': old_key, 'run_sha256': content_digest(old_identity), 'cache_sha256': file_digest(old_cache)}:
                        raise ValueError('Primary ordinary-control rows changed')
                    per_candidate[candidate['id']].extend(matched_rows(json.loads(old_cache.read_text()), fresh))
                    receipts[str(old_cache.relative_to(ROOT))] = file_digest(old_cache)
                print(json.dumps({'family': family, 'seed': seed, 'recording': recording,
                                  'scoring_source': scoring_source, 'queries': len(fresh), 'parent_scoring_exact_match': True}), flush=True)
            summaries = {name: summarize_registered(contract, rows) for name, rows in per_candidate.items()}
            result = {'fits': {v: {k: value for k, value in r.items() if k != 'losses'} for v, r in fit_reports.items()},
                      'summaries': summaries, 'original_controls_source': 'cached_verified', 'OOF_features_exactly_matched': True,
                      'scoring_sources': scoring_sources,
                      'fresh_forecast_scoring_exactly_matches_parent': True}
            atomic_json(target / 'comparison.json', result)
            results[family][str(seed)] = result
            heartbeat('seed_complete', family=family, seed=seed)
    report = {'result_source': 'fresh_run_or_receipt_verified_resume_per_fit', 'registration': reference,
              'results': results, 'primary_row_sha256': receipts, 'scope': 'development_diagnostic_only',
              'selected_variant': None, 'independent_confirmation': False, 'stage5c_executed': False, 'smc_enabled': False}
    atomic_json(output / 'comparison.json', report)
    atomic_json(output / 'completion.json', {'identity_sha256': content_digest(identity), 'report_sha256': file_digest(output / 'comparison.json')})
    heartbeat('all_registered_fits_and_evaluations_complete')
    print(json.dumps({'status': 'complete', 'families': len(results), 'seeds_per_family': len(protocol['seeds']),
                      'heads': len(results) * len(protocol['seeds']) * len(registration['variants'])}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
