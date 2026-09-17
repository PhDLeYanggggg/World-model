"""Freeze an output-only paired ablation while preserving the v6 data/evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, protocol_digest


def main():
    destination = ROOT/'configs/m3w_8to12_residual_parameterization_v7.json'
    if destination.exists():
        raise SystemExit('Versioned protocol exists; do not overwrite')
    parent = 'configs/m3w_8to12_conditioned_context_v6.json'
    p = json.loads((ROOT/parent).read_text())
    if protocol_digest(p) != '53be3aafbda47ddf8d60891e779896f6685fcd59222a1fe4ea867e689f8914de':
        raise ValueError('Unexpected parent protocol')
    reference = json.loads((ROOT/'configs/m3w_transformer_8to12_conditioned_v6.json').read_text())
    configs = [f'configs/m3w_transformer_8to12_{mode}_v7.json' for mode in ('residual_skip', 'motion_bounded')]
    for path, mode in zip(configs, ('baseline_skip', 'motion_bounded')):
        config = json.loads((ROOT/path).read_text())
        if (config['fit_settings'] != reference['fit_settings']
                or config['architecture']['output_parameterization'] != mode
                or {k: v for k, v in config['architecture'].items() if k != 'output_parameterization'} != reference['architecture']):
            raise ValueError('Only output parameterization may change')
    decision = 'outputs/publication_readiness_2026_09/residual_parameterization_v7_decision.md'
    p['experimental_change'] = {'parent_protocol_sha256': protocol_digest(p),
        'factor': 'CV_initialized_residual_with_or_without_observed_motion_amplitude_bound',
        'decision': decision, 'configs': configs, 'fit_or_development_rows_changed': False,
        'metric_or_targets_changed': False, 'development_adaptive_not_confirmatory': True}
    for stage in ('calibration', 'confirmation'):
        p[f'{stage}_receipt'] = f'data/stage_cvpr2027_experiments/residual_parameterization_v7/{stage}_forbidden.json'
    bindings = set(p['bindings']) | set(configs) | {parent, decision,
        'scripts/prepare_m3w_8to12_residual_parameterization.py', 'scripts/run_m3w_residual_parameterization_pair.py',
        'src/world_model/m3w_baseline_relative_forecaster.py'}
    p['bindings'] = {name: file_digest(ROOT/name) for name in sorted(bindings)}
    p['approval'] = {'approved_by': 'research_agent_under_explicit_user_delegation_2026-09-17',
        'decision_reference': decision, 'protocol_sha256': protocol_digest(p)}
    contract = ExperimentContract(p, ROOT)
    with destination.open('x') as stream:
        json.dump(p, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'protocol': str(destination), 'sha256': contract.digest,
        'frozen_error_scale_and_rows_unchanged': True, 'independent_confirmation': False}))


if __name__ == '__main__':
    main()
