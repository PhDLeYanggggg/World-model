import numpy as np
import pytest

from scripts import rehearse_m3w_pneuma_confirmation as exercise


def test_full_synthetic_paths_rehearse_calibration_and_sealed_resume(tmp_path):
    z = exercise.rehearsal(tmp_path/'zero', 'zero_shot')
    c = exercise.rehearsal(tmp_path/'cal', 'calibrated')
    assert z['synthetic_calibration_members'] == 0 and z['synthetic_confirmation_members'] == 6
    assert c['synthetic_calibration_members'] == 4 and c['synthetic_confirmation_members'] == 2
    assert not z['resume_reopens_outcomes'] and not c['resume_reopens_outcomes']


def test_block_before_any_outcome_if_remote_freeze_not_verified(tmp_path):
    p, w = tmp_path/'public', tmp_path/'work'
    p.mkdir()
    w.mkdir()
    dataset = exercise.Synthetic()
    publish = exercise.MockPublication()
    rule = exercise.spec('zero_shot')
    run = exercise.runner.execute
    run('register', rule, dataset, p, w, verify=publish)
    publish.publish(p/'registration.json')
    with pytest.raises(ValueError, match='Missing prospective publication'):
        run('confirm', rule, dataset, p, w, verify=publish)
    assert not dataset.access


def test_incomplete_members_cannot_be_dropped_from_date_average():
    rows = {n: {str(s): dict(value=1. if d == 'date2' else 0.) for s in exercise.readout.SEEDS}
            for n, d in exercise.DATES.items()}
    g = exercise.readout.date_interval(rows, exercise.DATES, lambda r: r['value'])
    assert g['point'] == .25
    rows['mock_b.csv']['17']['value'] = None
    r = exercise.readout.date_interval(rows, exercise.DATES, lambda x: x['value'])
    assert r['point'] is None and r['by_date']['date2'] is None


def test_radius_and_unidentifiable_pair_direction_are_separate():
    cv, ols, m, y, valid, offset, p = exercise.fixture('mock_d.csv')
    r = exercise.readout.score(cv, ols, m, y, valid, offset, p, easy_cut=.16, hard_cut=.28)
    z = r['matched_contrasts']['same_family']
    assert z['relative_lower'] <= z['relative_upper']
    assert np.isclose(r['arms']['joint']['intervention_rate'], .5)
    p['radius'][0] *= 2
    with pytest.raises(ValueError, match='Mismatched edit radius'):
        exercise.readout.score(cv, ols, m, y, valid, offset, p, easy_cut=.16, hard_cut=.28)
