"""Constructed-cost optimization check, not a real forecasting experiment."""
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
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/deferral_control')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '4'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import torch
    from src.evaluation.m3w_experiment_contract import file_digest
    from src.world_model.m3w_cost_sensitive_deferral import DeferralHead, deferral_surrogate, deferral_decision
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(928)
    began = time.monotonic()
    costs = torch.tensor([[.02, .01]] * 90 + [[.02, 1.]] * 10, device=args.device)
    x = torch.zeros(100, 1, device=args.device)
    head = DeferralHead([0.], [1.], width=0).to(args.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=.03)
    losses = []
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = deferral_surrogate(head(x), costs)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        probabilities = head(x).softmax(1)[0].cpu().tolist()
        selected = deferral_decision(head, x, support=torch.ones(100, dtype=torch.bool, device=args.device))
    if selected['use_candidate'].any():
        raise SystemExit('Cost-sensitive optimization failed the rare-harm fixture')
    report = {'result_source': 'fresh_run', 'scope': 'synthetic_constructed_cost_training_only',
              'rows': 100, 'independent_real_scenes': 0, 'feature_dimension': 1, 'optimizer_steps': len(losses),
              'seed': 928, 'learning_rate': .03, 'device': args.device, 'architecture': platform.machine(),
              'threads': 4, 'interop_threads': 1, 'dataloader_workers': 0, 'torch_version': str(torch.__version__),
              'majority_oracle_label': 'candidate', 'candidate_win_fraction': .9,
              'baseline_expected_constructed_cost': float(costs[:, 0].mean().cpu()),
              'candidate_expected_constructed_cost': float(costs[:, 1].mean().cpu()),
              'cost_sensitive_choice': 'baseline', 'softmax_by_action': probabilities,
              'softmax_is_calibrated_safety_probability': False,
              'per_row_normalized_weight_choice': int((costs.flip(1) / costs.sum(1, keepdim=True)).mean(0).argmax().cpu()),
              'loss_first': losses[0], 'loss_last': losses[-1], 'losses': losses,
              'elapsed_seconds': time.monotonic() - began,
              'real_forecasting_training': False, 'real_predictive_gain': False,
              'test_evaluated': False, 'formal_protocol_changed': False,
              'stage5c_executed': False, 'smc_enabled': False,
              'source_sha256': {p: file_digest(ROOT / p) for p in (
                  'scripts/check_m3w_deferral_control.py', 'src/world_model/m3w_cost_sensitive_deferral.py',
                  'src/world_model/m3w_supervised_intervention.py')}}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / 'constructed_cost_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'losses', 'source_sha256'}}))


if __name__ == '__main__':
    main()
