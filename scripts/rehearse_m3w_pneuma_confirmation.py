"""Run both pNEUMA readout contracts on fabricated, role-logged futures."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_pneuma_confirmation as runner

np, readout, protocol, storage = runner.np, runner.readout, runner.protocol, runner.storage
OUT = runner.OUT.parent/'pneuma_confirmation_rehearsal_v1'
DATES = dict(zip(('mock_a.csv', 'mock_b.csv', 'mock_c.csv', 'mock_d.csv', 'mock_e.csv', 'mock_f.csv'),
                 ('date1', 'date2', 'date2', 'date2', 'date3', 'date4'), strict=True))


def spec(mode):
    value = dict(mode=mode, approval_reference='synthetic fixture only',
                 observed_steps=8, prediction_steps=12, stride_ticks=10, risk_budget=.02,
                 comparison='all_past_defined_matched_queries')
    if mode == 'zero_shot':
        return dict(value, date_roles={d: 'confirmation' for d in DATES.values()}, rule=protocol.ZERO_RULE,
                    easy_cut=.16, hard_cut=.28, donor_cut_source='synthetic fixed reference')
    return dict(value, date_roles={d: 'calibration' if d in ('date1', 'date2') else 'confirmation' for d in DATES.values()},
                rule=protocol.RELEASE_RULE, calibration_cut_rule=protocol.CAL_RULE)


def fixture(name):
    phase = list(DATES).index(name)
    n, steps = 8, 12
    cv = np.zeros((n, steps, 2))
    y = cv.copy()
    y[:, :, 0] = (.12+np.arange(n)[:, None]*.022+phase*.002)
    y[:, :, 1] = np.arange(n)[:, None]*.01
    valid = np.ones((n, steps), dtype=bool)
    if phase % 2:
        valid[5, 9:] = False
        y[~valid] = np.nan
    ols, median = cv+.4, cv+.04
    offset = np.array([0, 4, 8])
    proposals = dict(radius=np.full(2, .035), offsets=offset)
    for a, k in zip(readout.ARMS, (0, 1, 2, 3), strict=True):
        rows = np.array([[0, k or 1], [4, 4+(k or 1)]])
        paths = cv[rows].copy()
        paths[..., 0] += .035
        if (phase, k) == (0, 1):
            paths[0, 0, :, 0] *= -1.
        proposals.update({f'{a}_rows': rows, f'{a}_paths': paths, f'{a}_score': np.full(2, .1+.1*phase+k/100.)})
    return cv, ols, median, y, valid, offset, proposals


class Synthetic:
    dates = DATES
    binding = {'fixture': 'explicitly synthetic coordinates, no data loader'}

    def __init__(self):
        self.access = []

    def calibration_errors(self, name):
        self.access.append(('cutoff', name))
        cv, _, _, y, valid, _, _ = fixture(name)
        return runner.positive_complete_cv_ade(cv, y, valid).tolist()

    def score(self, name, easy, hard):
        self.access.append(('score', name))
        row = readout.score(*fixture(name), easy_cut=easy, hard_cut=hard)
        return {str(s): row for s in readout.SEEDS}


class MockPublication:
    def __init__(self):
        self.published = {}

    def publish(self, *paths):
        self.published.update({p.name: storage.sha(p) for p in paths})

    def __call__(self, path):
        if self.published.get(path.name) != storage.sha(path):
            raise ValueError('Missing prospective publication')


def rehearsal(folder, mode):
    p, w = folder/'public', folder/'private'
    p.mkdir(parents=True)
    w.mkdir()
    mock, data = MockPublication(), Synthetic()
    rule = spec(mode)
    r, f = p/'registration.json', p/'freeze.json'
    first = runner.execute('register', rule, data, p, w, verify=mock)
    if data.access or first['outcomes_opened']:
        raise AssertionError('Registration opened data')
    mock.publish(r)
    if mode == 'calibrated':
        runner.execute('calibrate', rule, data, p, w, verify=mock)
        if {data.dates[n] for _, n in data.access} != {'date1', 'date2'}:
            raise AssertionError('Calibration touched held dates')
    mock.publish(f)
    before = len(data.access)
    result = runner.execute('confirm', rule, data, p, w, verify=mock)
    held = {data.dates[n] for _, n in data.access[before:]}
    expected = set(data.dates.values()) if mode == 'zero_shot' else {'date3', 'date4'}
    if held != expected:
        raise AssertionError('Held phase touched a nonconfirmation date')
    cached = len(data.access)
    for member in (w/'confirmation').glob('*.readout.json'):
        member.unlink()
    again = runner.execute('confirm', rule, data, p, w, verify=mock)
    if len(data.access) != cached or result != again:
        raise AssertionError('A resume must reuse immutable score receipts')
    return dict(mode=mode, synthetic_calibration_members=before//2, synthetic_confirmation_members=len(result['members']),
                synthetic_confirmation_dates=sorted(held), resume_reopens_outcomes=False,
                paired_relative_interval=result['matched']['joint_minus']['independent']['relative_lower'],
                risk_budget=result['official_risk_budget'], evidence_scope='fabricated targets only')


def main():
    with TemporaryDirectory(prefix='pneuma-rehearsal-') as directory:
        results = [rehearsal(Path(directory)/mode, mode) for mode in ('zero_shot', 'calibrated')]
    OUT.mkdir(parents=True, exist_ok=True)
    storage.write_json(OUT/'rehearsal.json', dict(result_source='fresh_synthetic_only', runs=results,
        actual_pneuma_future_access=False, formal_roles_assigned=False, actual_forecasts_scored=False, deployment_changed=False))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
