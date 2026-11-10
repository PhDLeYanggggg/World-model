import json

import numpy as np
import pytest

from reproducibility.opendd_angular_supplement.verify import replay
from scripts.build_m3w_angular_review_supplement import PANELS, archive, export, previous


def test_angular_panels_replay_and_require_same_rotation_count():
    data, expected = {}, {}
    for name, (folder, _, _) in PANELS.items():
        d = json.loads((archive.PUBLIC/folder/'readout.json').read_text())
        data[name], expected[name] = export(name, d)
    draws = np.random.default_rng(817).integers(7, size=(3000, 7)).tolist()
    assert previous.prior_builder.check(replay(data, draws), expected)
    data['calibration'][0][0]['counts']['constant_rotations'] += 1
    with pytest.raises(ValueError, match='Matched rotation counts'):
        replay(data, draws)
