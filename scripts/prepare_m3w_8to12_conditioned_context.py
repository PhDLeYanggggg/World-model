"""Bind a new input-conditioning ablation, leaving source caches and labels alone."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, protocol_digest


def main():
    destination = ROOT / 'configs/m3w_8to12_conditioned_context_v6.json'
    if destination.exists():
        raise SystemExit('Versioned protocol exists; do not overwrite')
    parent = 'configs/m3w_8to12_continuous_context_v5.json'
    p = json.loads((ROOT / parent).read_text())
    names = ['eqmotion', 'transformer']
    configs = [f'configs/m3w_{name}_8to12_conditioned_v6.json' for name in names]
    old_configs = ['configs/m3w_eqmotion_8to12_pruned_v4.json', 'configs/m3w_transformer_8to12_v3.json']
    for new, old in zip(configs, old_configs):
        a, b = (json.loads((ROOT / name).read_text()) for name in (new, old))
        if (a['fit_settings'] != b['fit_settings']
                or a['architecture']['input_conditioning'] != 'observed_joint_max_norm'
                or {k: v for k, v in a['architecture'].items() if k != 'input_conditioning'} != b['architecture']):
            raise ValueError('Only input conditioning may change')
    decision = 'outputs/publication_readiness_2026_09/conditioned_context_v6_decision.md'
    p['experimental_change'] = {'parent_protocol_sha256': protocol_digest(p),
        'factor': 'past_only_joint_input_scaling_output_restored_before_unchanged_loss',
        'decision': decision, 'configs': configs, 'fit_or_development_rows_changed': False,
        'metric_or_targets_changed': False, 'development_adaptive_not_confirmatory': True}
    for stage in ('calibration', 'confirmation'):
        p[f'{stage}_receipt'] = f'data/stage_cvpr2027_experiments/conditioned_context_v6/{stage}_forbidden.json'
    bindings = set(p['bindings']) | set(configs) | {parent, decision,
        'scripts/prepare_m3w_8to12_conditioned_context.py', 'scripts/run_m3w_conditioned_predictor_pair.py',
        'src/world_model/m3w_context_conditioning.py'}
    p['bindings'] = {name: file_digest(ROOT / name) for name in sorted(bindings)}
    p['approval'] = {'approved_by': 'research_agent_under_explicit_user_delegation_2026-09-16',
        'decision_reference': decision, 'protocol_sha256': protocol_digest(p)}
    contract = ExperimentContract(p, ROOT)
    with destination.open('x') as stream:
        json.dump(p, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'protocol': str(destination), 'sha256': contract.digest,
        'frozen_error_scale_unchanged': True, 'independent_confirmation': False}))


if __name__ == '__main__':
    main()
