from copy import deepcopy
import pytest
from scripts.report_m3w_moving_harm_floor import verify_roster


def test_reporting_requires_all_three_arms_in_all_contexts():
    jobs = [dict(key='critic'+str(i)) for i in range(3)]
    rows = [dict(job=j['key'], arm=a, step=2000, parameters=64260, both_neural_encoders_updated=True)
        for j in jobs for a in ('joint', 'uniform_harm', 'harm_floor')]
    verify_roster(rows, jobs)
    with pytest.raises(ValueError):
        verify_roster(rows[:-1], jobs)
    duplicate = deepcopy(rows); duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError):
        verify_roster(duplicate, jobs)
    partial = deepcopy(rows); partial[0]['step'] = 1999
    with pytest.raises(ValueError):
        verify_roster(partial, jobs)
