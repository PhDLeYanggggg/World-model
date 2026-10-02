import io
import numpy as np
import pytest
from scripts.run_m3w_quality_components import read_bytes, summarize
from src.world_model import m3w_quality_component_diagnostic as api


def test_transport_bounds_and_truncation():
    assert read_bytes(io.BytesIO(b'abc'), 3) == b'abc'
    with pytest.raises(EOFError): read_bytes(io.BytesIO(b'abc'), 4)
    with pytest.raises(ValueError): read_bytes(io.BytesIO(), 33*2**20)
    with pytest.raises(ValueError): read_bytes(io.BytesIO(), -1)


def test_summary_uses_localities_not_repeated_heads():
    p = api.variants(np.tile([.5, .01, 3., 3., .01], (4, 1)), np.zeros((4, 5)), np.ones(4))
    y = np.tile([.2, 0., 2., 2., 0.], (4, 1))
    r = api.evaluate(p, y, np.ones(4), np.ones(4, bool), np.ones(4, bool),
                     np.array([0, 0, 1, 1]), np.zeros(4), np.arange(4))
    rows = [dict(source=str(i//3), result=r) for i in range(6)]
    s = summarize(rows, dict(bootstrap_draws=3000, bootstrap_seed=20261002))
    assert s['arms']['quality']['full_utility_difference_percent']['localities'] == 2
    assert s['arms']['quality']['full_utility_difference_percent']['CI95'] == [0., 0.]
    assert s['projection_added_actions'] == 0
    assert not s['policy_selection'] and not s['training']
