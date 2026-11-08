import numpy as np

from scripts.report_m3w_replay_motion import serializable


def test_nested_readout_array_serialization():
    values = {"sources": np.array(["site_a"]), "mixture": [
        {"mean": np.array([.25, .75]), "n": np.int64(81), "support": None}
    ]}
    assert serializable(values) == {
        "sources": ["site_a"], "mixture": [{"mean": [.25, .75], "n": 81, "support": None}]
    }
