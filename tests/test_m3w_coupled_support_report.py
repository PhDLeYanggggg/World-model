import pytest
from scripts import report_m3w_coupled_support as report
from tests.test_m3w_coupled_support_verifier import evidence


def test_reporting_preserves_denominators_and_negative_slices(evidence):
    rows,_,_=evidence;data,md=report.describe(rows)
    assert len(data['source'])==4 and len(data['context'])==6 and len(data['seed'])==3
    assert len(data['strata'])==6
    for arm in report.run.api.ARMS:
        for t in ('all','easy'):
            got=data['all'][arm][t]
            assert got['predicted_harm']==pytest.approx(got['predicted_harm_on_events']+got['predicted_harm_on_non_events'])
    assert data['support']['local_supported']==data['all']['origin']['rows']
    assert sum(data['overlap'].values())==data['all']['origin']['rows']
    assert 'not trajectory ADE/FDE' in md and 'NOT the equal-view primary metric' in md
    assert all(v['not_pooled_quantiles'] for v in data['concentration'].values())
