import copy
import json
import pytest

from scripts.analyze_m3w_record_balanced_failures import record_weights, run


def fixture():
    torch = run.api.torch
    return dict(record_keys=[json.dumps(['source', '04']), json.dumps(['source', '4'])],
        utility_weights=torch.tensor([.25, .75]), constraint_factor=torch.tensor([2., 2./3.]))


def test_literal_recording_ids_remain_distinct():
    value = record_weights(fixture())
    assert set(value) == {('source', '04'), ('source', '4')}
    assert value['source', '04']['direction'] == 'increased'
    assert value['source', '4']['direction'] == 'decreased'


def test_duplicate_numeric_or_missing_recording_key_rejected():
    for keys in ([json.dumps(['source', '4'])]*2,
                 [json.dumps(['source', 4]), json.dumps(['source', '4'])], []):
        state = fixture()
        state['record_keys'] = keys
        with pytest.raises(ValueError):
            record_weights(state)


def test_nonuniform_or_unnormalized_weights_rejected():
    state = fixture()
    for field in ('utility_weights', 'constraint_factor'):
        changed = copy.deepcopy(state)
        changed[field][0] *= 2
        with pytest.raises(ValueError):
            record_weights(changed)
