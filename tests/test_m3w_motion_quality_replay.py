import json

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier

from scripts.run_m3w_source_motion_quality import positive_probability, probability_metrics
from scripts.verify_m3w_source_motion_quality import preserve_verification


@pytest.mark.parametrize('label', [0, 1])
def test_one_class_prediction_and_worker_restoration(label):
    x = np.arange(40).reshape(20, 2)
    model = ExtraTreesClassifier(n_estimators=4, n_jobs=4, random_state=17)
    model.fit(x, np.full(20, label))
    np.testing.assert_array_equal(positive_probability(model, x), np.full(20, label))
    assert model.n_jobs == 4


def test_prediction_failure_restores_workers():
    class Broken:
        n_jobs = 4

        def predict_proba(self, x):
            assert self.n_jobs == 1
            raise RuntimeError('fixture')

    model = Broken()
    with pytest.raises(RuntimeError, match='fixture'):
        positive_probability(model, np.zeros((1, 2)))
    assert model.n_jobs == 4


def test_probabilities_score_against_training_prior_not_held_prevalence():
    result = probability_metrics(np.array([0, 0, 1]), np.array([0., .5, 1.]), .8)
    assert result['brier'] == pytest.approx(.25 / 3)
    assert result['reference_brier'] == pytest.approx((.64 + .64 + .04) / 3)
    assert sum(item['rows'] for item in result['calibration_curve']) == 3
    with pytest.raises(ValueError):
        probability_metrics(np.array([0]), np.array([1.1]), .8)


def test_repeat_verification_preserves_old_receipt_but_checks_evidence(tmp_path):
    path = tmp_path / 'verification.json'
    old = {'immutable_artifacts': 149, 'completed_resume': {'pid': 1}, 'new_fits_on_resume': 0}
    preserve_verification(path, old)
    before = path.read_bytes()
    preserve_verification(path, {**old, 'completed_resume': {'pid': 2}})
    assert path.read_bytes() == before
    assert json.loads(path.read_text()) == old
    with pytest.raises(ValueError, match='evidence changed'):
        preserve_verification(path, {**old, 'immutable_artifacts': 148})
