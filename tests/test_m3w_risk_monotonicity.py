import numpy as np
import pytest
from src.evaluation.m3w_multiaction_support import score_choices


def test_fewer_switches_need_not_reduce_selected_positive_harm_ratio():
    # Two legal synthetic action costs, with true shared positive easy reference.
    targets = np.array([[[0., .03, 1., 1., .03]], [[.1, 0., 1., 1., 0.]]])
    envelope = np.full((2, 1), .2)
    both = score_choices(targets, envelope, np.array([0, 0]))
    pruned = score_choices(targets, envelope, np.array([0, -1]))
    assert both['known_easy_positive_risk'] == pytest.approx(.015)
    assert pruned['known_easy_positive_risk'] == pytest.approx(.03)
    assert both['known_easy_positive_risk'] <= .02 < pruned['known_easy_positive_risk']
    assert pruned['selected_count'] < both['selected_count']
