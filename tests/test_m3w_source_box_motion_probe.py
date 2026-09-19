import numpy as np
import pytest

from scripts.probe_m3w_source_box_motion import probability_metrics


def test_metrics_distinguish_accuracy_and_prior():
    y = np.array([0, 0, 1, 1])
    good = probability_metrics(y, np.array([.1,.2,.8,.9]))
    prior = probability_metrics(y, np.full(4,.5))
    assert good['auroc'] == good['auprc'] == 1
    assert prior['brier'] == .25 and good['brier'] < prior['brier']
    assert good['log_loss'] < prior['log_loss']


def test_single_class_is_undefined_not_zero_and_invalid_labels_rejected():
    result = probability_metrics(np.zeros(3), np.full(3,.1))
    assert result['auroc'] is None and result['auprc'] is None
    with pytest.raises(ValueError): probability_metrics(np.array([2]),np.array([.3]))
    with pytest.raises(ValueError): probability_metrics(np.array([1]),np.array([np.nan]))
