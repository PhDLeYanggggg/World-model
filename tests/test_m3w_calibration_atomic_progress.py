import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from src.world_model import m3w_calibration_atomic_progress as api


def test_legacy_two_file_window_is_reproducible(tmp_path):
    import hashlib
    from scripts.build_m3w_european_squares_source import store_array
    matrix = tmp_path/'choices.npy'; progress = tmp_path/'progress.json'
    x = np.full((13, 4, 2), -1, np.int8); x[0] = 0; store_array(matrix, x)
    digest = hashlib.sha256(matrix.read_bytes()).hexdigest()
    progress.write_text(json.dumps(dict(choices_sha256=digest)))
    x[1] = 1; store_array(matrix, x)
    # Simulate interruption after replacing choices but before replacing JSON.
    assert hashlib.sha256(matrix.read_bytes()).hexdigest() != json.loads(progress.read_text())['choices_sha256']


def test_atomic_replace_failure_keeps_previous_valid_prefix(tmp_path):
    import os
    path = tmp_path/'grid.gz'; identity = dict(frozen='same')
    choices = np.full((13, 4, 3), -1, np.int8); choices[0] = 0
    api.commit(path, identity, choices, [dict(multiplier=1)])
    choices[1] = 1
    with patch.object(os, 'replace', side_effect=OSError('interrupted before atomic replace')):
        with pytest.raises(OSError):
            api.commit(path, identity, choices, [dict(multiplier=1), dict(multiplier=2)])
    old = api.load(path, identity, 3)
    assert len(old['allocation']) == 1
    assert (old['choices'][1:] == -1).all()
    api.commit(path, identity, choices, [dict(multiplier=1), dict(multiplier=2)])
    resumed = api.load(path, identity, 3)
    assert len(resumed['allocation']) == 2
    np.testing.assert_array_equal(resumed['choices'], choices)


def test_wrong_identity_or_uncommitted_nonempty_suffix_rejected(tmp_path):
    path = tmp_path/'grid.gz'; choices = np.full((13, 4, 2), -1, np.int8)
    api.commit(path, dict(frozen=1), choices, [])
    with pytest.raises(ValueError):
        api.load(path, dict(frozen=2), 2)
    choices[2] = 0
    with pytest.raises(ValueError):
        api.commit(path, dict(frozen=1), choices, [])
