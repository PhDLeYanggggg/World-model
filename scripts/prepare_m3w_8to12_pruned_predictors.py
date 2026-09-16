"""Register equivalent selected-head execution, preserving the incomplete pilot."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, protocol_digest


def main():
    destination = ROOT / 'configs/m3w_8to12_public_predictors_v4.json'
    if destination.exists():
        raise SystemExit('Versioned protocol exists; do not overwrite')
    parent = 'configs/m3w_8to12_public_predictors_v3.json'
    p = json.loads((ROOT / parent).read_text())
    configs = ['configs/m3w_eqmotion_8to12_pruned_v4.json', 'configs/m3w_transformer_8to12_v3.json']
    eq, local = [json.loads((ROOT / name).read_text()) for name in configs]
    original = json.loads((ROOT / 'configs/m3w_eqmotion_8to12_v3.json').read_text())
    if ({k: v for k, v in eq['architecture'].items() if k != 'prune_unused_heads'} != original['architecture']
            or not eq['architecture']['prune_unused_heads']
            or eq['fit_settings'] != original['fit_settings'] or eq['fit_settings'] != local['fit_settings']):
        raise SystemExit('Execution-only optimization must preserve architecture and matched budget')
    decision = 'outputs/publication_readiness_2026_09/public_predictor_v4_execution_decision.md'
    p['experimental_change'] = {'parent_protocol_sha256': protocol_digest(p),
        'factor': 'skip_unused_author_output_heads_selected_arithmetic_unchanged',
        'decision': decision, 'configs': configs, 'development_adaptive_not_confirmatory': True}
    for stage in ('calibration', 'confirmation'):
        p[f'{stage}_receipt'] = f'data/stage_cvpr2027_experiments/8to12_public_v4/{stage}_forbidden.json'
    bindings = set(p['bindings']) | set(configs) | {parent, decision,
        'scripts/prepare_m3w_8to12_pruned_predictors.py', 'scripts/run_m3w_public_predictor_pair.py'}
    p['bindings'] = {name: file_digest(ROOT / name) for name in sorted(bindings)}
    p['approval'] = {'approved_by': 'research_agent_under_explicit_user_delegation_2026-09-16',
        'decision_reference': decision, 'protocol_sha256': protocol_digest(p)}
    contract = ExperimentContract(p, ROOT)
    with destination.open('x') as stream:
        json.dump(p, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'protocol': str(destination), 'sha256': contract.digest,
                      'independent_confirmation': False}))


if __name__ == '__main__':
    main()
