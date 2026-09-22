import numpy as np
import pytest
import torch
from scripts.verify_m3w_fraction_square import empirical_square, fractions_from_logits, manual_choices, native_scores
from src.world_model.m3w_bounded_cost_head import build, predict
from src.world_model.m3w_fraction_square_head import fraction, loss_value
from src.evaluation.m3w_risk_ranking import fixed_count


def test_separate_weighted_loss_equals_empirical_batch_objective():
    rng = np.random.default_rng(76); logits = torch.tensor(rng.normal(size=(30, 2)), dtype=torch.float64)
    q = rng.dirichlet([1, 1, 1], 30)[:, :2]; d = rng.uniform(.1, 3, 30); w = rng.uniform(.1, 2, 30)
    np.testing.assert_allclose(fractions_from_logits(logits.numpy()), fraction(logits).numpy(), rtol=1e-12, atol=1e-14)
    risk = empirical_square(fractions_from_logits(logits.numpy()), q, d*w)
    objective = loss_value(logits, torch.tensor(q), torch.tensor(d), torch.tensor(w))
    assert risk == pytest.approx(float(objective)/(d*w).mean())


def test_separate_rank_and_strict_choices():
    rng = np.random.default_rng(761); p = rng.uniform(0, 10, (300, 2))
    ids = rng.permutation(300); e = rng.uniform(size=300) > .1
    q = manual_choices(p, e, ids, 17)
    for rule in ('ratio', 'gain'): np.testing.assert_array_equal(q['square_'+rule], fixed_count(p, e, ids, 17, rule))
    np.testing.assert_array_equal(q['square_strict'], e & (p[:, 0] > p[:, 1]) & (p[:, 1] <= .1*p[:, 0]))


def test_assembled_float32_forward_exactly_matches_checkpoint_predictor():
    rng = np.random.default_rng(20); x = rng.normal(size=(4130, 6)).astype(np.float32)
    d = rng.uniform(0, 1000, len(x)); d[:4] = 0
    model = build(6, 11, 20)
    with torch.no_grad():
        model.network[-1].weight.copy_(torch.randn_like(model.network[-1].weight))
        model.network[-1].bias.copy_(torch.randn_like(model.network[-1].bias))
    pr = dict(mean=np.ones(6), std=np.full(6, 2.), constant=np.zeros(6, bool), cost_scale=7.7)
    cp = dict(model=model.state_dict(), preprocess=pr)
    np.testing.assert_array_equal(native_scores(cp, x, d), predict(model, x, d, pr, 'bounded_native'))
