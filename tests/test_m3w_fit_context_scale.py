from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_m3w_fit_context_scale import audit_fit_only, context_magnitudes, summarize
from src.data_unification.m3w_causal_recordings import FEATURE_NAMES


def example():
    f = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    f[FEATURE_NAMES.index('normalization_scale')] = .001
    return {'history_xy': np.array([[-.5, 0], [-.2, 0], [0, 0]]),
        'history_mask': np.ones(3, bool), 'history_frame_offsets': np.array([-2, -1, 0]),
        'neighbor_xy': np.array([[[500., 0], [500., 0], [500., 0]], [[9999., 0]] * 3]),
        'neighbor_mask': np.array([[True] * 3, [False, True, True]]),
        'neighbor_frame_offsets': np.array([[-2, -1, 0], [0, -1, 0]]), 'causal_features': f}


def test_aligned_neighbors_and_float32_scale_floor():
    row = context_magnitudes(example())
    assert row[1:].tolist() == [.5, 500., 1.]
    assert summarize([row])['floor_and_neighbor_above_100'] == 1


def test_incomplete_or_offset_context_not_used():
    x = example()
    x['neighbor_frame_offsets'][0, 0] = -3
    assert context_magnitudes(x)[2:].tolist() == [0., 0.]


def test_future_or_invalid_values_refused():
    x = example()
    for field, value in [('neighbor_frame_offsets', 1), ('history_xy', np.inf)]:
        changed = deepcopy(x)
        changed[field].flat[0] = value
        with pytest.raises(ValueError, match='past-only'):
            context_magnitudes(changed)


def test_fit_reader_only_no_label_access():
    opened = []

    def open_recording(name, *, purpose):
        opened.append((name, purpose))
        assert name == 'fit' and purpose == 'fit'
        return SimpleNamespace(get_inputs=lambda _: example()), [0, 1]

    contract = SimpleNamespace(protocol={'assignments': {'fit': 'fit', 'dev': 'development', 'test': 'confirmation'}},
                               open_recording=open_recording)
    assert audit_fit_only(contract)['fit']['rows'] == 2
    assert opened == [('fit', 'fit')]
