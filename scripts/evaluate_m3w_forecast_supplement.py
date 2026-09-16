"""Exact raw50 prefixes and count-matched controls for a completed v6 study."""
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


def verify_completed_evaluation(directory, digest):
    receipt = json.loads((directory / 'completion.json').read_text())
    required = {'development_report.json', 'selected_policy.json', 'artifact.json'}
    if set(receipt['artifacts']) != required:
        raise ValueError('Primary development completion manifest is incomplete')
    for name, expected in receipt['artifacts'].items():
        if digest(directory / name) != expected:
            raise ValueError('Primary development result changed')
    import hashlib
    identity = json.loads((directory / 'run_identity.json').read_text())
    actual = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if receipt['run_sha256'] != actual:
        raise ValueError('Primary completion/run identity mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    if args.threads < 1:
        raise SystemExit('Positive thread count required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import torch
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.evaluation.m3w_development_evaluation import (
        content_digest, validate_plan, load_verified_forecaster, _load_cost_head,
        scene_requests, decide_scene, score_scene,
    )
    from src.evaluation.m3w_forecast_supplement import (
        attach_count_controls, score_raw_prefix, summarize_supplement,
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    protocol = json.loads(args.protocol.read_text())
    study, output = args.study_dir.resolve(), args.output_dir.resolve()
    if not study.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise SystemExit('Use workspace-local study and cache paths')
    heartbeat = json.loads((study / 'runner_heartbeat.json').read_text())
    if heartbeat.get('state') != 'requested_seeds_complete' or heartbeat['seeds'] != protocol['seeds']:
        raise SystemExit('Complete every registered primary seed before supplementary scoring')
    families = []
    for seed in protocol['seeds']:
        artifacts = [json.loads((study / f'seed{seed}_{name}' / 'artifact.json').read_text())
                     for name in ('full', 'hold0', 'hold1', 'hold2', 'ridge', 'neural_cost')]
        contract = ExperimentContract(protocol, ROOT, artifacts)
        plan = json.loads((study / f'seed{seed}_development_plan.json').read_text())
        rules, recordings = validate_plan(contract, plan)
        primary = study / f'seed{seed}_development'
        verify_completed_evaluation(primary, file_digest)
        primary_identity = json.loads((primary / 'run_identity.json').read_text())
        if primary_identity['device'] != args.device or primary_identity['plan'] != plan:
            raise ValueError('Keep the registered primary device and candidate family')
        if primary_identity['protocol_sha256'] != contract.digest:
            raise ValueError('Primary protocol identity mismatch')
        for candidate in plan['candidates']:
            model = load_verified_forecaster(contract, candidate['forecaster_id'], device='cpu')
            if model._fitted_baseline_name != candidate['baseline']:
                raise ValueError('Baseline identity mismatch')
            _load_cost_head(contract, candidate)
            del model
        families.append((seed, contract, plan, rules, recordings))
    decision_path = ROOT / 'outputs/publication_readiness_2026_09/forecast_supplement_v6_decision.md'
    identity = {'parent_protocol_sha256': contract.digest, 'family_artifacts': [c.artifacts for _, c, _, _, _ in families],
        'plans': [p for _, _, p, _, _ in families], 'device': args.device, 'threads': args.threads,
        'decision_sha256': file_digest(decision_path), 'raw_horizon': 50,
        'code_sha256': {p: file_digest(ROOT / p) for p in (
            'src/evaluation/m3w_forecast_supplement.py', 'scripts/evaluate_m3w_forecast_supplement.py')},
        'eligible_for_selection': False}
    identity_path = output / 'run_identity.json'
    if args.resume:
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Supplement identity changed; no cache reuse permitted')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(identity_path, identity)
    began = time.monotonic()
    results, reused = {}, 0
    for seed, contract, plan, rules, recordings in families:
        results[str(seed)] = {}
        for candidate in plan['candidates']:
            model = load_verified_forecaster(contract, candidate['forecaster_id'], device=args.device)
            head = _load_cost_head(contract, candidate, device=args.device)
            all_full, all_prefix, all_counts = [], [], []
            for recording in recordings:
                reader, _ = contract.open_recording(recording, purpose='development')
                key = [seed, candidate['id'], recording]
                stem = content_digest(key)
                cache, receipt = output / (stem + '.json'), output / (stem + '.receipt.json')
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved != {'key': key, 'run_sha256': content_digest(identity), 'cache_sha256': file_digest(cache)}:
                        raise ValueError('Supplement cache changed')
                    part = json.loads(cache.read_text())
                    reused += 1
                else:
                    part = {'full': [], 'raw50': [], 'counts': []}
                    for number, scene in enumerate(scene_requests(reader, protocol['task'], stride=rules['query_stride'])):
                        decision = attach_count_controls(decide_scene(scene, model, head, baseline=candidate['baseline'],
                            policy=rules['policies'][candidate['policy_id']], geometry=rules['geometry_by_recording'][recording],
                            device=args.device, solver_seconds=rules['solver_seconds'], include_matched_coverage=True))
                        # Decisions, including the exact-count control, precede all label access.
                        labels = reader.get_scene_labels(scene)
                        part['full'].extend(score_scene(scene, decision, labels, label_policy=rules['label_policy']))
                        prefix = score_raw_prefix(scene, decision, labels, raw_horizon=50)
                        if prefix is not None:
                            part['raw50'].extend(prefix)
                        part['counts'].append({**{k: scene[k] for k in ('recording_id', 'frame_id', 'horizon_raw')},
                            **{k: decision['coverage_match'][k] for k in (
                                'status', 'matched', 'nonzero_matched', 'reference_count', 'joint_count', 'agents')},
                            'raw50_on_native_grid': prefix is not None})
                        if number % 50 == 0:
                            atomic_json(output / 'heartbeat.json', {'pid': os.getpid(), 'key': key, 'query': number,
                                'elapsed_seconds': time.monotonic()-began, 'status': 'running_not_complete'})
                    atomic_json(cache, part)
                    atomic_json(receipt, {'key': key, 'run_sha256': content_digest(identity), 'cache_sha256': file_digest(cache)})
                all_full.extend(part['full'])
                all_prefix.extend(part['raw50'])
                all_counts.extend(part['counts'])
            results[str(seed)][candidate['id']] = summarize_supplement(all_full, all_prefix, all_counts, contract)
            print(json.dumps({'seed': seed, 'candidate': candidate['id'], 'agent_queries': len(all_full),
                              'status': 'supplement_candidate_complete_not_selected'}), flush=True)
            del model, head
    report = {'result_source': 'cached_verified' if reused == sum(len(p['candidates'])*len(r) for _, _, p, _, r in families) else 'fresh_run',
        'reused_recording_evaluations': reused, 'scope': 'development_diagnostic_only', 'identity': identity,
        'seeds': results, 'elapsed_seconds_this_invocation': time.monotonic()-began,
        'stage5c_executed': False, 'smc_enabled': False}
    atomic_json(output / 'supplement_report.json', report)
    atomic_json(output / 'completion.json', {'run_sha256': content_digest(identity), 'report_sha256': file_digest(output / 'supplement_report.json')})
    print(json.dumps({'status': 'supplement_complete', 'seeds': list(results), 'independent_confirmation': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
