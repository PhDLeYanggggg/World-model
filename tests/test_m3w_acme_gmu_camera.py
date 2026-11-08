import numpy as np
import pytest

from scripts.run_m3w_acme_gmu_camera import associate_query, rollout
from scripts.run_m3w_acme_gmu_reassociate import reassociate


def make_example():
    history = [[5. + i, 20.] for i in range(8)]
    frames = [{"camera_frame": i*12, "rows": [[9, x-2, y-8, x+2, y, .9, 1]]}
              for i, (x, y) in enumerate(history)]
    labels = {"same": {i*4: xy for i, xy in enumerate(history)}}
    return history, frames, labels


def test_both_baselines_use_same_anchor_and_recover_linear_motion():
    history, _, _ = make_example()
    paths = rollout(history)
    assert paths.shape == (2, 12, 2)
    np.testing.assert_allclose(paths[0], paths[1])
    np.testing.assert_allclose(paths[0, -1], [24., 20.])


def test_future_perturbation_cannot_change_admission_or_forecasts():
    _, frames, labels = make_example()
    support, bank = associate_query(frames, labels, 0)
    labels["same"][32] = [999., -999.]
    support_after, bank_after = associate_query(frames, labels, 0)
    assert support["stable_matches"] == 1
    assert support_after == support
    np.testing.assert_array_equal(bank[0][1], bank_after[0][1])


def test_discontinuous_online_history_or_wrong_clock_never_passes():
    _, frames, labels = make_example()
    frames[2]["rows"] = []
    support, bank = associate_query(frames, labels, 0)
    assert support["online_histories"] == support["stable_matches"] == 0
    assert bank == []
    frames[2]["camera_frame"] += 1
    with pytest.raises(ValueError, match="frame mismatch"):
        associate_query(frames, labels, 0)


def test_causal_reassociation_preserves_boxes_and_prefix():
    _, frames, _ = make_example()
    for i, frame in enumerate(frames):
        frame["rows"][0][0] = 100+i
    full = reassociate(frames)
    assert full[:4] == reassociate(frames[:4])
    assert {f["rows"][0][0] for f in full} == {0}
    assert [f["rows"][0][1:] for f in full] == [f["rows"][0][1:] for f in frames]
