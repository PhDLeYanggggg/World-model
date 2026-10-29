import pytest

from scripts.run_m3w_coupled_local_cost import summarize
from scripts.report_m3w_coupled_local_cost import interval


def test_complete_roster_is_not_replaceable_by_a_partial_success():
    with pytest.raises(ValueError,match='Complete288'):
        summarize([],dict(comparison_seeds=[17,29,43]),{}, {})


def test_undefined_interval_is_not_rendered_as_zero_or_safe():
    assert interval(dict(point=None,CI95=None))=='not_estimable'
    assert interval(dict(point=1.,CI95=None))=='+1.000000 [not_estimable]'
    assert interval(dict(point=1.,CI95=[-1.,2.]))=='+1.000000 [-1.000000, +2.000000]'
