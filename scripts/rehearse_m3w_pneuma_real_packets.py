"""Exercise both Athens contracts on frozen arrays and fabricated targets only."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import rehearse_m3w_pneuma_confirmation as mock

runner, np, storage = mock.runner, mock.np, mock.storage
OUT = runner.OUT.parent/'pneuma_real_packet_rehearsal_v1'


class FrozenWithFabricatedTargets(runner.Dataset):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.binding = dict(self.binding, target_source='fabricated from row index and past history')

    def _labels(self, name, prefix):
        # No real file is passed to this object or opened on this path.
        self.calls.append(name)
        h = prefix['history']
        stride = np.arange(1, 13)
        speed = (h[:, -1]-h[:, -4])/3.
        target = speed[:, None]*stride[None, :, None]
        target[:, :, 0] += (.08+.07*(np.arange(len(h)) % 29))[:, None]*stride[None, :]
        valid = np.ones(target.shape[:2], dtype=bool)
        valid[::7, 8:] = False
        valid[::101] = False
        target[~valid] = np.nan
        return target, valid


def specification(data, mode):
    dates = sorted(set(data.dates.values()))
    roles = ['confirmation']*4 if mode == 'zero_shot' else ['calibration']*2+['confirmation']*2
    spec = dict(mode=mode, approval_reference='rehearsal with fabricated targets only',
        observed_steps=8, prediction_steps=12, stride_ticks=10, risk_budget=.02,
        comparison='all_past_defined_matched_queries', date_roles=dict(zip(dates, roles, strict=True)))
    if mode == 'zero_shot':
        p = runner.REHEARSAL/'donor_cuts.json'
        cuts = runner.pilot.read(p)
        spec.update(rule=runner.protocol.ZERO_RULE, donor_cut_source=storage.sha(p),
                    easy_cut=cuts['easy_cut'], hard_cut=cuts['hard_cut'])
    else:
        spec.update(rule=runner.protocol.RELEASE_RULE, calibration_cut_rule=runner.protocol.CAL_RULE)
    return spec


def exercise(data, mode, root):
    public, private = root/'public', root/'private'
    public.mkdir(parents=True)
    private.mkdir()
    spec = specification(data, mode)
    data.check_cuts(spec)
    published = mock.MockPublication()
    registration, freeze = public/'registration.json', public/'freeze.json'
    count = len(data.calls)
    row = runner.execute('register', spec, data, public, private, verify=published)
    if len(data.calls) != count or row['outcomes_opened']:
        raise ValueError('Rehearsal registration tried to score')
    published.publish(registration)
    if mode == 'calibrated':
        runner.execute('calibrate', spec, data, public, private, verify=published)
        if set(data.calls[count:]) != {n for n, r in row['member_roles'].items() if r == 'calibration'}:
            raise ValueError('Calibration accessed a non-calibration member')
    n_cal = len(set(data.calls[count:]))
    published.publish(freeze)
    before = len(data.calls)
    outcome = runner.execute('confirm', spec, data, public, private, verify=published)
    held = set(data.calls[before:])
    if held != {n for n, r in row['member_roles'].items() if r == 'confirmation'}:
        raise ValueError('Confirmation accessed a non-confirmation member')
    n = len(data.calls)
    again = runner.execute('confirm', spec, data, public, private, verify=published)
    if n != len(data.calls) or outcome != again:
        raise ValueError('Existing summary did not replay')
    report = public/'confirmation.json'
    snapshot = private/'confirmation.snapshot.json'
    report.rename(snapshot)
    rebuilt = runner.execute('confirm', spec, data, public, private, verify=published)
    if n != len(data.calls) or rebuilt != outcome or storage.sha(report) != storage.sha(snapshot):
        raise ValueError('Member receipts did not reconstruct the missing summary')
    return dict(mode=mode, calibration_members=n_cal, confirmation_members=len(held),
                confirmation_dates=len(set(outcome['dates'].values())),
                risk_budget=outcome['official_risk_budget'], cached_summary_replayed=True,
                rebuilt_from_member_receipts_without_target_calls=True)


def run():
    data = FrozenWithFabricatedTargets()
    with TemporaryDirectory(prefix='pneuma-real-packet-rehearsal-') as temp:
        rows = [exercise(data, mode, Path(temp)/mode) for mode in ('zero_shot', 'calibrated')]
    OUT.mkdir(parents=True, exist_ok=True)
    storage.write_json(OUT/'receipt.json', dict(result_source='fresh_frozen_inputs_synthetic_outcomes_rehearsal',
        bindings=dict(data.binding, rehearsal_driver=storage.sha(Path(__file__))), variants=rows,
        real_future_positions_read=False, real_accuracy_scored=False,
        real_roles_assigned=False, deployment_changed=False))
    print(rows, flush=True)


if __name__ == '__main__':
    run()
