"""Register the declared one-factor loss ablation; preserve the original study."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, protocol_digest


def main():
    source = ROOT / 'configs/m3w_8to12_development_v1.json'
    destination = ROOT / 'configs/m3w_8to12_robust_v2.json'
    if destination.exists():
        raise SystemExit('Versioned ablation exists; do not overwrite it')
    p = json.loads(source.read_text())
    old = json.loads((ROOT / 'configs/m3w_intervention_backend.json').read_text())
    newpath = 'configs/m3w_intervention_robust_backend.json'
    new = json.loads((ROOT / newpath).read_text())
    if (new['architecture'] != old['architecture'] or
            {k: v for k, v in new['fit_settings'].items() if k != 'objective'} != old['fit_settings'] or
            new['fit_settings']['objective'] != 'smooth_l1'):
        raise SystemExit('The declared single-factor ablation changed additional training settings')
    decision = 'outputs/publication_readiness_2026_09/robust_loss_ablation_decision.md'
    p['experimental_change'] = {'parent_protocol_sha256': protocol_digest(p),
        'factor': 'forecaster_coordinate_loss', 'from': 'mse', 'to': 'smooth_l1_beta_1',
        'decision': decision, 'development_adaptive_not_confirmatory': True}
    for stage in ('calibration', 'confirmation'):
        p[f'{stage}_receipt'] = f'data/stage_cvpr2027_experiments/8to12_robust_v2/{stage}_forbidden.json'
    bindings = set(p['bindings']) | {decision, newpath, 'configs/m3w_8to12_development_v1.json',
                                   'scripts/prepare_m3w_8to12_robust_ablation.py'}
    p['bindings'] = {name: file_digest(ROOT / name) for name in sorted(bindings)}
    p['approval'] = {'approved_by': 'research_agent_under_explicit_user_delegation_2026-09-16',
                     'decision_reference': decision, 'protocol_sha256': protocol_digest(p)}
    contract = ExperimentContract(p, ROOT)
    with destination.open('x') as stream:
        json.dump(p, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'protocol': str(destination), 'sha256': contract.digest,
                      'changed_factor': p['experimental_change'], 'independent_confirmation': False}))


if __name__ == '__main__':
    main()
