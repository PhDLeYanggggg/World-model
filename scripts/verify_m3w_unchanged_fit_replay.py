"""Verify unchanged fitting across the explicit development-source amendment."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.world_model.m3w_supervised_intervention import load_verified_forecaster, parameter_digest


def compare_training_states(a, b):
    identities = [{k: v for k, v in s['identity'].items() if k != 'protocol_sha256'} for s in (a, b)]
    if identities[0] != identities[1] or any(s['step'] != s['identity']['settings']['steps'] for s in (a, b)):
        raise ValueError('Only complete fits with unchanged training identities may be compared')
    keys = set(a['model'])
    if keys != set(b['model']):
        raise ValueError('Parameter/state schema changed')
    return {'steps': a['step'], 'loss_sequence_exactly_equal': a['losses'] == b['losses'],
        'model_state_exactly_equal': all(torch.equal(a['model'][k], b['model'][k]) for k in keys),
        'max_model_state_absolute_difference': max(float((a['model'][k] - b['model'][k]).abs().max()) for k in keys),
        'sampler_order_exactly_equal': torch.equal(a['order'], b['order']),
        'sampler_cursor_equal': a['cursor'] == b['cursor'],
        'sampler_rng_exactly_equal': torch.equal(a['sampler_rng'], b['sampler_rng'])}


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    states, provenance = [], []
    configs = ('m3w_8to12_public_predictors_v4.json', 'm3w_8to12_continuous_context_v5.json')
    for version, config in zip(('v4', 'v5'), configs):
        artifact_path = ROOT / f'data/stage_cvpr2027_experiments/8to12_eqmotion_{version}/seed17_full/artifact.json'
        artifact = json.loads(artifact_path.read_text())
        contract = ExperimentContract(json.loads((ROOT / 'configs' / config).read_text()), ROOT, [artifact])
        model = load_verified_forecaster(contract, artifact['id'], device='cpu')
        path = ROOT / artifact['path']
        states.append(torch.load(path, map_location='cpu', weights_only=True))
        provenance.append({'version': version, 'protocol_sha256': contract.digest,
            'checkpoint_sha256': file_digest(path), 'parameter_sha256': parameter_digest(model)})
    result = {'result_source': 'fresh_run_exact_replay_comparison_of_verified_completed_real_fits',
        **compare_training_states(*states), 'provenance': provenance,
        'v4_resumed_after_step': 100, 'v5_uninterrupted_fit': True,
        'changed_component': 'development_source_only_not_fit_data_or_training',
        'accuracy_evidence': False, 'independent_confirmation': False,
        'scope': 'one_real_seed_same_hardware_not_general_cross_device_reproducibility',
        'code_sha256': file_digest(Path(__file__))}
    output = ROOT / 'outputs/publication_readiness_2026_09/8to12_public_predictors_v5'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'unchanged_fit_replay.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
