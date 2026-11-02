import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import plot_m3w_independent_risk_calibration as api


def complete_documents():
    curves, taxonomy, comparisons = [], [], []
    for multiplier in [2**i for i in range(13)]:
        curves.append(dict(multiplier=multiplier, policies={a:dict(
            required_record_views=918, supported_record_views=10) for a in api.ARMS}))
        for arm in api.ARMS:
            taxonomy.append(dict(multiplier=multiplier, arm=arm,
                exclusive=dict(supported=10, unsupported_reference=908)))
            if arm != 'independent':
                comparisons.append(dict(multiplier=multiplier, contrast=arm+'_minus_independent'))
    return (dict(curves=curves),
            dict(rows=[dict(multiplier=r['multiplier']) for r in curves], full_model_record_grids=918),
            dict(rows=taxonomy, repeated_pipeline_record_views=918),
            dict(comparisons=comparisons, complete_pipeline_record_views=918))


def test_complete_cohort_with_no_safety_pass_can_be_plotted():
    api.validate(*complete_documents())


@pytest.mark.parametrize('change', ['missing_curve', 'missing_policy', 'bad_gate', 'duplicate_contrast'])
def test_incomplete_or_inconsistent_evidence_rejected(change):
    docs = copy.deepcopy(complete_documents())
    if change == 'missing_curve':
        docs[0]['curves'].pop()
    elif change == 'missing_policy':
        docs[0]['curves'][0]['policies'].pop('joint_free')
    elif change == 'bad_gate':
        docs[0]['curves'][0]['policies']['independent']['supported_record_views'] = 11
    else:
        docs[3]['comparisons'][0] = docs[3]['comparisons'][1]
    with pytest.raises(ValueError):
        api.validate(*docs)


def test_nonestimable_point_remains_missing_not_a_safe_zero():
    fig, ax = plt.subplots()
    api.series(ax, [1, 2], [dict(point=None, CI95=None), dict(point=3, CI95=[1, 4])], 'black', 'test')
    values = ax.lines[0].get_ydata()
    assert np.isnan(values[0]) and values[1] == 3
    plt.close(fig)
