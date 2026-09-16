"""Register a new paired development protocol without rewriting prior studies."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, protocol_digest


def main():
    destination = ROOT / 'configs/m3w_8to12_public_predictors_v3.json'
    if destination.exists():
        raise SystemExit('Versioned protocol exists; do not overwrite')
    parent = 'configs/m3w_8to12_robust_v2.json'
    protocol = json.loads((ROOT / parent).read_text())
    configs = ['configs/m3w_eqmotion_8to12_v3.json', 'configs/m3w_transformer_8to12_v3.json']
    a, b = [json.loads((ROOT / name).read_text()) for name in configs]
    if a['fit_settings'] != b['fit_settings']:
        raise SystemExit('Matched training budgets/loss required')
    decision = 'outputs/publication_readiness_2026_09/public_predictor_v3_decision.md'
    protocol['experimental_change'] = {
        'parent_protocol_sha256': protocol_digest(protocol),
        'factor': 'paired_public_core_and_local_predictors_matched_budget_and_past_neighbor_support',
        'decision': decision, 'development_adaptive_not_confirmatory': True,
        'not_single_factor_relative_to_v2': True, 'configs': configs}
    for stage in ('calibration', 'confirmation'):
        protocol[f'{stage}_receipt'] = f'data/stage_cvpr2027_experiments/8to12_public_v3/{stage}_forbidden.json'
    bindings = set(protocol['bindings']) | set(configs) | {
        parent, decision, 'configs/m3w_eqmotion_source.json',
        'src/world_model/m3w_eqmotion_adapter.py',
        'scripts/prepare_m3w_8to12_public_predictors.py',
        'scripts/run_m3w_8to12_development.py', 'scripts/train_m3w_causal_forecaster.py'}
    protocol['bindings'] = {name: file_digest(ROOT / name) for name in sorted(bindings)}
    protocol['approval'] = {'approved_by': 'research_agent_under_explicit_user_delegation_2026-09-16',
        'decision_reference': decision, 'protocol_sha256': protocol_digest(protocol)}
    contract = ExperimentContract(protocol, ROOT)
    with destination.open('x') as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'protocol': str(destination), 'sha256': contract.digest,
                      'independent_confirmation': False}))


if __name__ == '__main__':
    main()
