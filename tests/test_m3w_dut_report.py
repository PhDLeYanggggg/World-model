from types import SimpleNamespace

import numpy as np
import pytest

from scripts.report_m3w_dut_readout import merge, ratio_change, arm_metrics
from src.evaluation.m3w_dut_readout import batch_infer


def test_merge_refuses_different_populations_of_fields():
    assert merge([dict(a=1,b=dict(c=2)), dict(a=3,b=dict(c=4))]) == dict(a=4,b=dict(c=6))
    with pytest.raises(ValueError):
        merge([dict(a=1), dict(b=2)])


def test_zero_baseline_is_not_fabricated_percentage():
    assert ratio_change(0., 0.) == 0.
    assert ratio_change(1., 0.) is None
    assert ratio_change(3., 2.) == 50.


def site(n, ade, baseline):
    part = dict(n=n, ade_sum=n*ade, fde_sum=n*ade, baseline_ade_sum=n*baseline,
                baseline_fde_sum=n*baseline, normalized_ade_sum=n*ade/2, harm_count=0)
    return dict(targets=n, complete=n, queries=n, arms=dict(example=dict(switches=n,
        gain_lower_sum=n*(baseline-ade), gain_upper_sum=n*(baseline-ade), slices=dict(all=part))))


def test_site_equal_bootstrap_never_turns_windows_into_sites():
    x = arm_metrics(dict(a=site(100, 1., 2.), b=site(1, 5., 4.)), "example")
    m = x["subsets"]["all"]
    assert m["native_ADE"] == 3.
    assert m["baseline_ADE"] == 3.
    assert m["improvement_percent"] == 0.
    assert m["paired_two_site_bootstrap_interval"] == [-25., 50.]
    assert m["interval_is_population_certificate"] is False
    assert x["full_population_gain_interval"] == [0., 0.]
    with pytest.raises(ValueError):
        arm_metrics(dict(a=site(100, 1., 2.)), "example")


def test_no_supported_site_does_not_silently_change_aggregation():
    x = arm_metrics(dict(a=site(0, 1., 2.), b=site(1, 5., 4.)), "example")
    assert x["subsets"]["all"]["status"] == "insufficient_support_in_at_least_one_site"


def test_batched_predictions_keep_scene_decisions_separate(monkeypatch):
    from tests.test_m3w_frozen_policy_chain import prefix
    from src.world_model.m3w_frozen_policy_chain import scene_from_prefix, decide
    scenes = [scene_from_prefix(prefix()), scene_from_prefix(prefix(short=True))]
    import src.evaluation.m3w_dut_readout as module
    def predict(model, data, ids, batch_size):
        assert batch_size == 128
        return data["geometry"][ids, 332:356].reshape(-1,12,2)+.01
    def costs(model, x, d, preprocess):
        return np.column_stack((.8*d, .01*d))
    monkeypatch.setattr(module, "forecast", predict)
    monkeypatch.setattr(module, "predict_forest", costs)
    view = SimpleNamespace(predictor=None, head=None, forest=True, preprocess={"cost_scale":1.})
    out = batch_infer(view, scenes)
    for s, actual in zip(scenes, out):
        base = s.geometry[:,332:356].reshape(-1,12,2)
        p = base+.01
        d = module.disagreement(p, base, s.scales).mean(1)
        expected = decide(s, p, costs(None,None,d,None), cost_scale=1.)
        for arm in expected["choices"]:
            np.testing.assert_array_equal(actual["choices"][arm], expected["choices"][arm])
        np.testing.assert_array_equal(actual["candidate"], expected["candidate"])
