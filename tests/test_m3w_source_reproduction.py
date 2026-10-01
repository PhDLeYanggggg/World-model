import copy
import json

import pytest

from scripts.package_m3w_source_reproduction import build, isolated_verify
from reproducibility.selected_pool_source.verify import verify


@pytest.fixture(scope='module')
def bundle():
    return build()


@pytest.fixture
def evidence(bundle):
    return json.loads(bundle['evidence.json'])


def test_real_evidence_in_isolated_process(bundle):
    result = isolated_verify(bundle)
    assert (result['views'], result['defined'], result['undefined']) == (72, 30, 42)
    assert result['point_value_checks'] == 504
    assert result['contributing_localities'] == 8
    assert result['CI95'][0] < 0 < result['CI95'][1]
    assert not result['torch_imported'] and not result['raw_data_or_checkpoint_replay']


def test_moment_corruption_rejected(evidence):
    evidence['rows'][0]['pools']['kept']['truth'][4] += 1
    with pytest.raises(AssertionError): verify(evidence)


def test_unknown_count_corruption_rejected(evidence):
    evidence['rows'][0]['pools']['kept']['unknown'] = 0
    with pytest.raises(AssertionError): verify(evidence)


def test_empty_pool_cannot_become_zero_risk(evidence):
    empty = next(r for r in evidence['rows'] if r['expected']['kept_risk'] is None)
    empty['expected']['kept_risk'] = 0
    with pytest.raises(AssertionError): verify(evidence)


def test_bad_ci_rejected(evidence):
    evidence['expected_interval']['defined_only_descriptive']['CI95'][0] += .001
    with pytest.raises(AssertionError): verify(evidence)


def test_changed_budget_and_duplicate_view_rejected(evidence):
    bad = copy.deepcopy(evidence); bad['risk_budget'] = .03
    with pytest.raises(AssertionError): verify(bad)
    evidence['rows'][-1] = evidence['rows'][0]
    with pytest.raises(AssertionError): verify(evidence)


def test_silently_dropping_undefined_rows_rejected(evidence):
    evidence['rows'] = [r for r in evidence['rows'] if r['expected']['kept_risk'] is not None]
    with pytest.raises(AssertionError): verify(evidence)
