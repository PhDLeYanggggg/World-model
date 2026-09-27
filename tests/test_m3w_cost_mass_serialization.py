import copy
import json
import numpy as np
import pytest
from scripts import report_m3w_european_cost_mass_native as r


def fixture():
    return {'full': {'cost_only': {weight: {component:
        {r.COUNT: np.int64(60), 'unmodified_MSE': .123, 'missing': None}
        for component in ('H_all', 'H_easy')}
        for weight in ('equal_locality', 'row_weighted')}}}


def test_count_conversion_preserves_every_value_and_original():
    source = fixture(); before = copy.deepcopy(source)
    with pytest.raises(TypeError): json.dumps(source, allow_nan=False)
    result = r.native_counts(source)
    assert result == source == before
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    for row in result['full']['cost_only'].values():
        assert type(row['H_easy'][r.COUNT]) is int
    assert type(source['full']['cost_only']['equal_locality']['H_easy'][r.COUNT]) is np.int64


def test_noninteger_counts_are_not_silently_truncated():
    source = fixture(); source['full']['cost_only']['equal_locality']['H_easy'][r.COUNT] = 1.2
    with pytest.raises(TypeError): r.native_counts(source)
