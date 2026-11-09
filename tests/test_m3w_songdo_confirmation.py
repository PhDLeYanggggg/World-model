from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from scripts import run_m3w_songdo_confirmation as runner
from src.evaluation import m3w_songdo_protocol as protocol


class Fixture:
    sites = {'a.csv': 'A', 'a2.csv': 'A', 'b.csv': 'B', 'c.csv': 'C', 'd.csv': 'D'}
    lineage = dict(candidate_source_components=[['A', 'B'], ['C'], ['D']])
    binding = {'fixture': 'test'}

    def __init__(self):
        self.calls = []
        self.cutoff_calls = []
        self.cuts_seen = []
        self.fail_once = None

    def cutoff_errors(self, flight):
        self.cutoff_calls.append(flight)
        return [.2, .25, .3] if flight == 'a.csv' else [.25]

    def score(self, flight, spec, thresholds):
        self.calls.append((flight, thresholds is not None))
        self.cuts_seen.append((spec.get('easy_cut'), spec.get('hard_cut')))
        if flight == self.fail_once:
            self.fail_once = None
            raise OSError('test interruption')
        cv = np.zeros((3, 12, 2))
        delta = cv[:2].copy()
        delta[..., 0] = .1
        future = cv.copy()
        future[..., 0] = -.25 if flight in ('b.csv', 'd.csv') else .25
        p = dict(bounds=(0, 3), radius=.1, arms={a: (np.array([0, 1]), delta) for a in protocol.ARMS})
        scores = {a: (2. if flight in ('a.csv', 'c.csv', 'd.csv') else 1.) for a in protocol.ARMS}
        return protocol.score_session(cv, cv, np.array([0, 3]), future, np.ones(cv.shape[:2], bool),
                                      {s: (p, scores) for s in protocol.SEEDS}, spec, thresholds)


def specification():
    return dict(mode='calibrated', approval_reference='synthetic test only', easy_cut=.3, hard_cut=.5,
                site_roles={'A': 'calibration', 'B': 'calibration', 'C': 'confirmation', 'D': 'confirmation'},
                release={'method': 'risk_passing_score_prefix'})


def test_split_linked_sites_is_rejected_before_any_outcome_access(tmp_path):
    data, spec = Fixture(), specification()
    spec['site_roles']['B'] = 'confirmation'
    with pytest.raises(ValueError, match='Linked sites'):
        runner.execute(tmp_path, 'register', spec, data)
    assert not data.calls and not list(tmp_path.iterdir())


def test_calibration_freezes_before_confirmation_and_resumes_once(tmp_path):
    data, spec = Fixture(), specification()
    runner.execute(tmp_path, 'register', spec, data)
    with pytest.raises(FileNotFoundError, match='freeze.json'):
        runner.execute(tmp_path, 'confirm', spec, data)
    assert not data.calls
    calibration = runner.execute(tmp_path, 'calibrate', spec, data)
    assert data.calls == [('a.csv', False), ('a2.csv', False), ('b.csv', False)]
    assert calibration['thresholds']['17']['joint'] == 2.
    assert calibration['calibration_counts']['17']['joint'] == {'eligible': 3, 'released': 1}
    data.fail_once = 'd.csv'
    with pytest.raises(OSError, match='interruption'):
        runner.execute(tmp_path, 'confirm', spec, data)
    assert runner.first.read(tmp_path/'heartbeat.json')['last'] == 'c.csv'
    confirm = runner.execute(tmp_path, 'confirm', spec, data)
    assert data.calls[3:] == [('c.csv', True), ('d.csv', True), ('d.csv', True)]
    assert confirm['flight_seed_views'] == 6
    assert confirm['released']['joint']['edited'] == 6
    assert confirm['released']['joint']['risk_supported'] == 3  # A failing confirmation is not retuned.
    assert runner.execute(tmp_path, 'confirm', spec, data) == confirm
    assert len(data.calls) == 6
    spec['easy_cut'] = .35
    with pytest.raises(ValueError, match='changed after registration'):
        runner.execute(tmp_path, 'confirm', spec, data)


def test_zero_shot_cannot_fit_and_empty_release_is_not_certified(tmp_path):
    data, spec = Fixture(), specification()
    spec.update(mode='zero_shot', site_roles={s: 'confirmation' for s in 'ABCD'},
                release=dict(method='fixed_scores', threshold_source='synthetic fixture',
                             thresholds={str(s): {a: None for a in protocol.ARMS} for s in protocol.SEEDS}))
    runner.execute(tmp_path, 'register', spec, data)
    with pytest.raises(ValueError, match='No flights assigned to calibration'):
        runner.execute(tmp_path, 'calibrate', spec, data)
    assert not data.calls
    confirm = runner.execute(tmp_path, 'confirm', spec, data)
    assert all(has_thresholds for _, has_thresholds in data.calls)
    joint = confirm['released']['joint']
    assert joint['edited'] == joint['risk_supported'] == 0
    assert joint['no_edit_not_certified'] == 15
    assert joint['completion_lower_pp']['vs_cv']['point'] == 0.
    assert confirm['matched_seed_views'] == 15


def test_calibration_estimates_cuts_from_approved_groups_only(tmp_path):
    data, spec = Fixture(), specification()
    spec.pop('easy_cut')
    spec.pop('hard_cut')
    spec['cutoff_rule'] = protocol.CALIBRATION_CUTOFF_RULE
    runner.execute(tmp_path, 'register', spec, data)
    with pytest.raises(FileNotFoundError, match='freeze.json'):
        runner.execute(tmp_path, 'confirm', spec, data)
    assert not data.cutoff_calls
    frozen = runner.execute(tmp_path, 'calibrate', spec, data)
    assert data.cutoff_calls == ['a.csv', 'a2.csv', 'b.csv']
    assert frozen['easy_cut'] == frozen['hard_cut'] == .25
    runner.execute(tmp_path, 'confirm', spec, data)
    assert data.cutoff_calls == ['a.csv', 'a2.csv', 'b.csv']
    assert data.cuts_seen == [(.25, .25)]*5
    cuts = tmp_path/'calibration_cutoffs.json'
    cuts.write_text(cuts.read_text()+'\n')
    with pytest.raises(ValueError, match='Frozen file changed'):
        runner.execute(tmp_path, 'confirm', spec, data)


def test_matched_only_zero_shot_reports_utility_without_any_risk_pass(tmp_path):
    data = Fixture()
    spec = dict(mode='zero_shot', approval_reference='synthetic only',
                site_roles={s: 'confirmation' for s in 'ABCD'}, cutoff_rule='not_used',
                release={'method': 'matched_only'})
    runner.execute(tmp_path, 'register', spec, data)
    with pytest.raises(ValueError, match='No flights assigned to calibration'):
        runner.execute(tmp_path, 'calibrate', spec, data)
    out = runner.execute(tmp_path, 'confirm', spec, data)
    assert out['risk_and_release'] == 'not_run_no_calibration'
    assert 'released' not in out
    assert out['matched_joint_minus']['additive']['lower_pp']['point'] == 0.
    assert out['matched_native_metrics']['cv']['available_ade_native']['point'] == .25
    assert out['matched_vs_cv']['joint']['lower_pp']['status'] == 'estimable'
    assert data.cuts_seen == [(None, None)]*5 and not data.cutoff_calls


def test_complete_cv_cutoff_excludes_zero_and_partial_rows():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[1:, :, 0] = np.array([.1, .3, .8])[:, None]
    valid = np.ones((4, 12), dtype=bool)
    valid[-1, -1] = False
    y[~valid] = np.nan
    x = protocol.positive_complete_cv_ade(cv, y, valid)
    np.testing.assert_allclose(x, [.1, .3])
    r = protocol.fit_native_cutoffs({'cal.csv': x, 'empty.csv': []})
    assert r['easy_cut'] == pytest.approx(.15)
    assert r['hard_cut'] == pytest.approx(.25)
    with pytest.raises(ValueError, match='No complete positive'):
        protocol.fit_native_cutoffs({'empty.csv': []})


def test_selective_archive_parser_checks_original_identity_and_ignores_unselected_xy(tmp_path):
    from tests.test_m3w_songdo_future_join import example

    p, ix, _ = example()
    ix['future_index'][0, 0], ix['future_index'][1, 1] = 2, 1
    csv = ('Vehicle_ID,Drone_ID,Local_Time,Visibility,Local_X,Local_Y\n'
           '19,2,00:00:00.000,1,NOT_A_NUMBER,NOT_A_NUMBER\n'
           '23,2,00:00:00.801,1,126,200\n'
           '19,2,00:00:00.400,1,121,200\n')
    path = Path(tmp_path)/'fixture.zip'
    with ZipFile(path, 'w') as z:
        z.writestr('synthetic.csv', csv)
    with ZipFile(path) as z:
        rows = runner.future_rows(z, 'synthetic.csv', ix)
    np.testing.assert_array_equal(rows['row'], [1, 2])
    y = runner.join.displacements(p, ix, rows)
    np.testing.assert_array_equal(y[ix['future_mask']], [[1., 0.], [2., 0.]])
