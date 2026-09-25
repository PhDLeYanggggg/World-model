"""A constructed estimand counterexample, not a real-data model result."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.world_model.m3w_ranked_hurdle import ranking_loss as old_loss
from src.world_model.m3w_supported_rank_pairs import ranking_loss


def counterexample():
    # A has two equally probable futures; B has one. All four pairs are in one locality.
    qa = Fraction(1, 4)
    qb = Fraction(3, 103)
    positive_share_weight = (Fraction(1, 2)-qb)/2
    negative_share_weight = qb/2
    target = torch.tensor([[1., 1.], [1., .03], [100., 0.], [1., .03]], dtype=torch.float64)
    true = torch.tensor([[50.5, .5], [1., .03], [50.5, .5], [1., .03]], dtype=torch.float64)
    biased = true.clone(); biased[[1, 3], 1] = .005
    sites = np.array(['same'] * 4)
    a, _ = ranking_loss(true, target, sites, epsilon=1e-6)
    b, _ = ranking_loss(biased, target, sites, epsilon=1e-6)
    old, _ = old_loss(true, target, sites, epsilon=1e-6)
    # Independent conditional draws give E[H_A B_B-H_B B_A]. This is only an algebraic target check.
    positive_cross_weight = Fraction(97, 100)/2
    negative_cross_weight = Fraction(3, 1)/2
    return dict(
        result_source='fresh_constructed_algebraic_diagnostic_not_real_data',
        conditional_risk_H_over_B=dict(A=float(Fraction(1, 101)), B=.03),
        expected_realized_share=dict(A=float(qa), B=float(qb)),
        share_pair_weight_positive=float(positive_share_weight),
        share_pair_weight_negative=float(negative_share_weight),
        observed_share_pair_optimal_score_A_minus_B=math.log(float(positive_share_weight/negative_share_weight)),
        cross_moment_pair_optimal_score_A_minus_B=math.log(float(positive_cross_weight/negative_cross_weight)),
        expected_cross_moment_difference=float(positive_cross_weight-negative_cross_weight),
        existing_rank_loss_true_moments=float(a), existing_rank_loss_biased_moments=float(b),
        original_supported_loss_equal=bool(torch.equal(old, a)),
        biased_prediction=dict(B_reference=1., B_harm=.005),
        finite_log_floor=1e-6,
        interpretation='realized_share_order_can_oppose_ratio_of_conditional_expected_costs',
        proves_empirical_cause=False, new_model_training=False, new_deployment=False,
        caveat='cross_moment_sign_identity_requires_independent_conditional_draws; batch_ratio_normalization_and_dependence_remain')


def main():
    p = ROOT/'outputs/publication_readiness_2026_09/european_supported_pairs_v1'
    result = counterexample()
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p.mkdir(parents=True, exist_ok=True)
    (p/'estimand_counterexample.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
