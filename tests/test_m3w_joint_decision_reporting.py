from copy import deepcopy
import pytest
from scripts.report_m3w_moving_joint_decision import verify_roster


def test_reporting_requires_every_frozen_fit_not_only_a_favorable_context():
    jobs = [dict(key='critic'+str(i)) for i in range(3)]
    rows = [dict(job=j['key'], arm=arm, step=2000, parameters=64260,
        both_neural_encoders_updated=True) for j in jobs for arm in ('per_channel', 'joint')]
    verify_roster(rows, jobs)
    with pytest.raises(ValueError):
        verify_roster(rows[:-1], jobs)
    duplicate = deepcopy(rows); duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError):
        verify_roster(duplicate, jobs)
    incomplete = deepcopy(rows); incomplete[2]['step'] = 1999
    with pytest.raises(ValueError):
        verify_roster(incomplete, jobs)
    frozen = deepcopy(rows); frozen[4]['both_neural_encoders_updated'] = False
    with pytest.raises(ValueError):
        verify_roster(frozen, jobs)
