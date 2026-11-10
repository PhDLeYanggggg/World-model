"""Rehearse both registered roles on the actual frozen proposals, with fake futures."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import rehearse_m3w_pneuma_real_packets as mock
from scripts import run_m3w_pneuma_confirmation_with_attribution as study

primary, read, storage = study.primary, study.read, study.storage
OUT = study.PROTOCOL.parent


class WithFakeTargets(study.Dataset):
    def __init__(self, spec):
        super().__init__(spec)
        self.calls = []
        self.binding['label_origin'] = 'synthetic_rehearsal_never_real'

    def load_targets(self, name, prefix):
        return mock.FrozenWithFabricatedTargets._labels(self, name, prefix)


def run():
    cases = {}
    dates = primary.Dataset()
    with TemporaryDirectory(prefix='pneuma-factorial-rehearsal-') as d:
        for mode in ('zero_shot', 'calibrated'):
            spec = mock.specification(dates, mode)
            data = WithFakeTargets(spec)
            data.check_cuts(spec)
            reg, w = Path(d)/mode/'public', Path(d)/mode/'work'
            reg.mkdir(parents=True)
            w.mkdir()
            verify = mock.mock.MockPublication()
            r = primary.execute('register', spec, data, reg, w, verify=verify)
            if data.calls:
                raise ValueError('Registration opened targets')
            verify.publish(reg/'registration.json')
            if mode == 'calibrated':
                primary.execute('calibrate', spec, data, reg, w, verify=verify)
                if set(data.calls) != {n for n, role in r['member_roles'].items() if role == 'calibration'}:
                    raise ValueError('Calibration crossed the date split')
            start = len(data.calls)
            verify.publish(reg/'freeze.json')
            c = primary.execute('confirm', spec, data, reg, w, verify=verify)
            if data.calls[start:] != c['members']:
                raise ValueError('Confirmation did not read each held member exactly once')
            part = study.report(c, reg, w)
            for p in (reg/'confirmation.json', reg/'recipient_path_attribution.json'):
                p.rename(p.with_suffix('.original.json'))
            c2 = primary.execute('confirm', spec, data, reg, w, verify=verify)
            part2 = study.report(c2, reg, w)
            if c != c2 or part != part2 or len(data.calls) != start+len(c['members']):
                raise ValueError('Scalars did not reproduce without new target access')
            cases[mode] = dict(calibration_calls=start, held_join_calls=len(c['members']),
                               held_dates=part['source_date_count'], exact_cached_replay=True)
    storage.write_json(OUT/'rehearsal.json', dict(real_future_coordinates_read=False, real_roles_assigned=False,
        fabricated_metrics_not_exported=True, cases=cases, runner=storage.sha(Path(study.__file__)),
        rehearsal=storage.sha(Path(__file__)), bounds=storage.sha(Path(study.bounds.__file__)),
        bank=storage.sha(study.frozen.OUT/'athens_freeze.json'), original_2pct=True))
    print(cases, flush=True)


if __name__ == '__main__':
    run()
