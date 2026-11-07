import numpy as np
import pytest

from src.evaluation import m3w_dronecrowd_confirmation as m
from src.world_model.m3w_dronecrowd_calibration import GATES


def test_never_reads_calibration_record_or_ambiguous_role():
    manifest = dict(recording_role_index=dict(a='confirmation', b='calibration', c='confirmation'),
        assignments=dict(calibration={'recordings': ['b', 'c']}, confirmation={'recordings': ['a', 'c']}))
    m.require_confirmation(manifest, 'a')
    for key in ('b', 'c', 'd'):
        with pytest.raises(PermissionError):
            m.require_confirmation(manifest, key)


def test_positive_matched_counts_do_not_overrule_required_recording_failures():
    effect = dict(completion_lower_pp=dict(interval95=[.1, .2]))
    names = ('cv', 'ols4', 'matched_independent', 'matched_scene', 'matched_pair_additive')
    summary = dict(risk=dict(selected=dict(supported_views=215, required_views=216, switches=99)),
                   effects={f'selected_vs_{k}': effect for k in names})
    pair = dict(selected_vs_matched_pair_additive=dict(grid_ade_lower=dict(interval95=[.1, .2])))
    assert not m.verdict(summary, pair)['independent_joint_claim_supported']
    summary['risk']['selected']['supported_views'] = 216
    assert m.verdict(summary, pair)['independent_joint_claim_supported']


def test_matched_query_diagnostic_does_not_activate_when_policy_abstains():
    cv = np.zeros((4, 12, 2))
    i = np.array([[0, 1], [2, 3]])
    p = np.ones((2, 2, 12, 2))
    take = np.array([[0, 1], [-1, -1]])
    r = m.matched_forecast(cv, i, p, take)
    np.testing.assert_array_equal(r[:2], 1.)
    np.testing.assert_array_equal(r[2:], 0.)
    with pytest.raises(ValueError):
        m.matched_forecast(cv, i, p, np.array([[0, -1], [-1, -1]]))


def test_only_the_original_qualifying_threshold_is_admissible():
    summary = dict(recordings=40, source_groups=15, views=120,
        risk={k: dict(supported_views=120 if k == 'gate_0' else 119,
                     required_views=120, switches=1) for k in GATES},
        effects={f'{k}_vs_cv': dict(completion_lower_pp=dict(interval95=[.1, .2], point=.15)) for k in GATES})
    doc = dict(summary=summary, calibration_pass=True, confirmation_outcome_access=False,
               independent_confirmation=False, original_rules_unchanged=True,
               selected_arm='gate_0', selected_normalized_gain_margin=0.)
    assert m.selected_gate(doc) == 'gate_0'
    with pytest.raises(ValueError):
        m.selected_gate(dict(doc, selected_normalized_gain_margin=.01))
    with pytest.raises(ValueError):
        m.selected_gate(dict(doc, calibration_pass=False))


def test_readout_keeps_empty_matched_diagnostic_unestimable(tmp_path, monkeypatch):
    from scripts import run_m3w_dronecrowd_confirmation as runner

    class Fixture:
        query_frames = np.array([7])

        def get_inputs(self, frame):
            assert frame == 7
            return dict(history_complete=np.array([True, True]), agent_ids=np.array([1, 2]),
                        positions=np.zeros((2, 8, 2)))

        def get_labels(self, raw):
            return dict(future_positions=np.ones((2, 12, 2)), future_mask=np.ones((2, 12), bool))

    monkeypatch.setattr(runner, 'reader', lambda *args: Fixture())
    path = tmp_path/'forecast.npz'
    cv = np.zeros((2, 12, 2))
    fields = dict(frames=Fixture.query_frames, offsets=np.array([0, 2]), agents=np.array([1, 2]),
                  origin=np.zeros((2, 2)), cv=cv, ols=cv)
    for seed in runner.cal.SEEDS:
        fields[f'index_{seed}'] = np.full((4, 1, 2), -1)
        fields[f'paths_{seed}'] = np.zeros((4, 1, 2, 12, 2))
    np.savez(path, **fields)
    monkeypatch.setattr(runner.run, 'checked', lambda a: path)
    output = runner.recording({}, 'fixture', {'artifact': {}}, {}, {}, {'easy': 2., 'hard': 4.}, 'g')
    for row in output['rows']:
        assert row['comparisons']['selected_vs_cv']['reference'] > 0
        for name in runner.PAIR_CONTRASTS:
            assert row['comparisons'][name]['reference'] == 0
        assert not row['metrics']['selected']['risk']['supported']
    assert not np.asarray([v['counts'] for v in output['pair_moments']]).any()
