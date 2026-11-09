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
        self.fail_once = None

    def score(self, flight, spec, thresholds):
        self.calls.append((flight, thresholds is not None))
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
