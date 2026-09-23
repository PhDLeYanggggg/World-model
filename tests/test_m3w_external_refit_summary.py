import copy

import pytest

from scripts.summarize_m3w_external_refit import summarize


def fixture():
    models = []
    for family in ("transformer", "eqmotion"):
        for seed in (17, 29, 43):
            models.append(dict(family=family, seed=seed, trainable_parameters=10,
                fit=dict(step=4000, complete=True, total_draws=256000,
                    unique_training_rows=10, held_rows_sampled=0, seconds=12.,
                    losses=[dict(step=s, loss=1., gradient_norm=2., learning_rate=.001,
                        supported_batch_rows=64, mean_past_normalized_ADE=20.)
                        for s in [1, *range(50, 4001, 50)]])))
    return dict(scope="fixed_source_predictor_refit_not_external_results", models=models,
        no_model_selection=True, source_training_only=True, external_gain_established=False,
        reserved_source_inference="not_run", independent_calibration="not_run",
        independent_confirmation="not_run", deployment=False)


def test_summary_separates_sampled_training_loss_from_evaluation():
    report = summarize(fixture())
    assert report["models"] == 6 and report["optimizer_updates"] == 24000
    assert report["sampled_loss_records"] == 486
    assert not report["heldout_accuracy_established"]
    assert report["trials"][0]["last_ten_logged_losses_mean"] == 1.
    assert report["trials"][0]["preclip_gradient_above_five_logged_fraction"] == 0.


@pytest.mark.parametrize("change", ["missing", "duplicate", "step", "nan", "unordered", "held", "claim"])
def test_incomplete_or_misclaimed_fit_rejected(change):
    data = copy.deepcopy(fixture())
    fit = data["models"][0]["fit"]
    if change == "missing":
        data["models"].pop()
    elif change == "duplicate":
        data["models"][1] = data["models"][0]
    elif change == "step":
        fit["step"] = 100
    elif change == "nan":
        fit["losses"][0]["loss"] = float("nan")
    elif change == "unordered":
        fit["losses"][0], fit["losses"][1] = fit["losses"][1], fit["losses"][0]
    elif change == "held":
        fit["held_rows_sampled"] = 1
    else:
        data["external_gain_established"] = True
    with pytest.raises(ValueError):
        summarize(data)
