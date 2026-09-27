import pytest
from scripts.diagnose_m3w_causal_descriptor_refit import paired_quality


def rows():
    return [dict(group='g', site='s', seed=17, policy=p, metric=dict(
        all_fit_MSE=fit, all_signed_MSE=held, easy_fit_MSE=.1, easy_signed_MSE=.2))
        for p, fit, held in [('control', .3, .8), ('descriptor', .2, .9)]]


def test_paired_sign_and_generalization_gap():
    value = paired_quality(rows())[0]['metric']
    assert value['all_fit_MSE'] == pytest.approx(-.1)
    assert value['all_signed_MSE'] == pytest.approx(.1)
    assert value['all_generalization_gap_difference'] == pytest.approx(.2)
    assert value['easy_generalization_gap_difference'] == 0


def test_missing_or_duplicate_view_rejected():
    with pytest.raises(ValueError, match='roster'):
        paired_quality(rows()[:1])
    with pytest.raises(ValueError, match='Duplicate'):
        paired_quality(rows() + rows()[:1])


def test_unknown_score_is_not_zero():
    value = rows()
    value[1]['metric']['all_signed_MSE'] = None
    result = paired_quality(value)[0]['metric']
    assert result['all_signed_MSE'] is None
    assert result['all_generalization_gap_difference'] is None


def test_mismatched_seed_rejected():
    value = rows()
    value[1]['seed'] = 29
    with pytest.raises(ValueError, match='seeds'):
        paired_quality(value)
