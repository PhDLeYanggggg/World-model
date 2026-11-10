import numpy as np
import pytest

from scripts import freeze_m3w_external_recipient_factorial as study
from src.evaluation import m3w_recipient_factorial_readout as evaluator


def test_candidate_pairs_restrict_two_actors_and_preserve_tie_order():
    row = np.repeat(np.arange(4), 3)
    i, j = study.candidate_pairs(row, dict(I=[2, 0], A=[0, 2], J=[3, 1]))
    assert list(zip(i, j)) == sorted(set(zip(i, j)))
    assert {(row[a], row[b]) for a, b in zip(i, j)} == {(0, 2), (1, 3)}
    assert len(i) == 18


def test_matching_preserves_the_original_scoring_block_shapes(monkeypatch):
    monkeypatch.setattr(study.pilot.work, 'CHUNK', 7)
    row = np.repeat(np.arange(6), 3)
    groups = dict(I=[1, 3], J=[4, 5])
    i, j, use = study.original_blocks(row, groups)
    np.testing.assert_array_equal([i[use], j[use]], study.candidate_pairs(row, groups))
    original = list(study.pilot.work.stream.blocks(row, 7))
    for lo in range(0, len(i), 7):
        part = slice(lo, lo+7)
        assert any(np.array_equal(i[part], a) and np.array_equal(j[part], b) for a, b in original)


def test_envelope_is_full_roster_and_handles_recipient_overlap():
    cv = np.zeros((4, 12, 2))
    a, b = cv[[0, 1]].copy(), cv[[1, 2]].copy()
    a[..., 0] = 1
    b[0, :, 0] = 1
    b[1, :, 1] = 1
    left, right = ([0, 1], None, a), ([1, 2], None, b)
    expected_rel = (4+np.sqrt(2))/6
    np.testing.assert_allclose(study.distance_envelope(cv, left, right), [.5, .5, expected_rel, 1, 1])
    np.testing.assert_allclose(study.distance_envelope(cv, right, left), [.5, .5, expected_rel, 1, 1])
    np.testing.assert_array_equal(study.distance_envelope(cv, left, left), np.zeros(5))


def test_readout_separates_matched_utility_from_full_roster_risk():
    cv = np.zeros((5, 12, 2))
    offsets = np.array([0, 2, 5])
    frozen = dict(offsets=offsets, query_index=np.array([1]), radius=np.array([.5]))
    for a in evaluator.CELLS:
        frozen[a+'_rows'] = np.array([[2, 3]])
        frozen[a+'_paths'] = np.broadcast_to([.5, 0.], (1, 2, 12, 2)).copy()
    y = cv+.1
    valid = np.ones((5, 12), dtype=bool)
    out = evaluator.score(cv, cv, y, valid, offsets, frozen, study.CONTRASTS, easy_cut=.2, hard_cut=.3)
    assert out['matched'] == out['unmatched'] == 1
    assert out['arms']['II']['original_risk']['rows'] == 5
    assert out['arms']['II']['original_risk']['selected'] == 2
    assert out['paired']['path_j_i_on_i']['lower_pp'] == 0.
    frozen['IJ_rows'] = np.array([[2, 4]])
    with pytest.raises(ValueError, match='changed its assigned recipients'):
        evaluator.restore(cv, offsets, frozen)
