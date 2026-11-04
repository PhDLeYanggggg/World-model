import numpy as np
import pytest
from scripts.report_m3w_moving_harm_refinement import factorization,text_once


def test_factorization_separates_harm_and_reference_error():
    out=factorization([10,40,100,20,10],[8,1,99,18,.04])
    assert out['moments']['H']['ratio']==40
    assert out['moments']['EH']['ratio']==250
    assert out['moments']['R']['ratio']==pytest.approx(100/99)
    for key in ('all','easy'):
        row=out['risk'][key]
        assert row['known_over_predicted_risk']==pytest.approx(row['harm_factor_over_reference_factor'])


def test_zero_support_and_zero_predictions_are_not_safe_zero_ratios():
    out=factorization(np.zeros(5),np.zeros(5))
    assert out['risk']['all']['known_pooled_risk']is None
    assert out['moments']['H']['ratio']is None
    bad=factorization([0,1,10,2,1],[0,0,10,2,0])
    assert bad['moments']['EH']['status']=='zero_prediction_positive_realized_mass'
    assert bad['risk']['easy']['known_over_predicted_risk']is None
    with pytest.raises(ValueError):factorization([-1]*5,[1]*5)


def test_reports_are_plain_markdown_and_immutable(tmp_path):
    path=tmp_path/'readout.md';text='# Findings\n\nNot a JSON-encoded string.\n'
    text_once(path,text);text_once(path,text)
    assert path.read_text()==text
    with pytest.raises(ValueError):text_once(path,text+'changed')
