import numpy as np
import pytest
import torch

from test_m3w_native_forecast import population
from src.training.m3w_frozen_source_refit import full_source_design
from src.world_model.m3w_native_forecast import fold_design, masked_objective, draw_batch, fit_trial
from src.world_model.m3w_supervised_intervention import build_forecaster


def test_full_refit_is_explicitly_fit_only_not_an_outer_test():
    data = population()
    result = full_source_design(data, ["a", "b", "c"])
    np.testing.assert_array_equal(result["train_ids"], np.arange(12))
    assert len(result["held_ids"]) == 0
    assert result["evaluation_scope"] == "none_all_rows_are_fitting"
    assert sum(len(group) for group in result["groups"]) == 12


def test_same_site_normalization_as_existing_complement_implementation():
    data = population()
    result = full_source_design(data, ["a", "b", "c"])
    for site in ["a", "b", "c"]:
        old = fold_design(data, site, "native_coordinate")
        ids = old["train_ids"]
        np.testing.assert_array_equal(result["factors"][ids], old["factors"][ids])
        for source, normalizer in old["normalizers"].items():
            assert normalizer == result["normalizers"][source]


def test_unsupported_rows_do_not_disappear_or_change_supported_scene_objective():
    data = population()
    result = full_source_design(data, ["a", "b", "c"])
    predictions = torch.from_numpy(data["geometry"][:, 332:356].reshape(-1, 12, 2).copy())
    per_site = []
    for group in result["groups"]:
        loss, _ = masked_objective(predictions[group], torch.from_numpy(data["target"][group]),
            torch.from_numpy(data["valid"][group]), torch.tensor(result["factors"][group]))
        per_site.append(float(loss))
    np.testing.assert_allclose(per_site, 1., rtol=2e-6)
    assert 1 in result["groups"][0]


@pytest.mark.parametrize("change", ["new_site", "duplicate_site", "shape", "empty_labels", "nonfinite", "zero_scale"])
def test_refit_cannot_admit_another_domain_or_bad_supervision(change):
    data, expected = population(), ["a", "b", "c"]
    if change == "new_site":
        expected.append("reserved_external")
    elif change == "duplicate_site":
        expected.append("a")
    elif change == "shape":
        data["target"] = data["target"][:-1]
    elif change == "empty_labels":
        data["valid"][data["sites"] == "b"] = False
    elif change == "nonfinite":
        data["target"][0, 0] = np.nan
    else:
        data["scale"][0] = 0
    with pytest.raises(ValueError):
        full_source_design(data, expected)


def test_sampler_can_match_families_without_shared_model_rng():
    result = full_source_design(population(), ["a", "b", "c"])
    left, right = (torch.Generator().manual_seed(17 + 7919) for _ in range(2))
    a = draw_batch(result["groups"], 64, left)
    torch.rand(1000)
    b = draw_batch(result["groups"], 64, right)
    np.testing.assert_array_equal(a, b)


def test_full_source_real_torch_resume_is_exact(tmp_path):
    torch.set_num_threads(2)
    data = population()
    fold = full_source_design(data, ["a", "b", "c"])
    architecture = dict(width=8, heads=2, layers=1, neighbor_policy="complete_aligned_history",
        input_conditioning="observed_joint_max_norm", output_parameterization="motion_bounded")
    settings = dict(steps=4, batch_size=4, learning_rate=.001, minimum_lr_ratio=.01,
        weight_decay=.0001, gradient_clip=5, checkpoint_every=2, heartbeat_every=1)
    identity = {"scope": "synthetic_full_source_refit_test_not_model_result"}
    def run(path, **kwargs):
        torch.manual_seed(17)
        model = build_forecaster(architecture)
        result = fit_trial(model, data, fold, seed=17, settings=settings, identity=identity,
                           directory=path, heartbeat=lambda **values: None, **kwargs)
        return model, result
    full, _ = run(tmp_path / "full")
    run(tmp_path / "resumed", stop_at=2)
    resumed, result = run(tmp_path / "resumed", resume=True)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    assert result["held_rows_sampled"] == 0 and result["complete"]
    assert result["new_updates"] == 2 and result["total_draws"] == 16
