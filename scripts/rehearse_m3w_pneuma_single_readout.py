"""Exercise one-pass Athens scoring and interruption recovery with fake futures."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import rehearse_m3w_pneuma_real_packets as mock
from scripts import run_m3w_pneuma_single_readout as study

primary, storage = study.primary, study.storage


class Interrupted(RuntimeError):
    pass


class Fabricated(study.Dataset):
    def __init__(self, spec):
        super().__init__(spec)
        self.calls = []
        self.fail_before = None
        self.binding['label_origin'] = 'fabricated_never_real'

    def load_targets(self, name, prefix):
        if name == self.fail_before:
            self.fail_before = None
            raise Interrupted('Synthetic failure before a member join')
        return mock.FrozenWithFabricatedTargets._labels(self, name, prefix)


def output_paths(root):
    public, work = root/'public', root/'work'
    public.mkdir(parents=True)
    work.mkdir()
    return public, work


def blocked_before_join(fn, data):
    n = len(data.calls)
    try:
        fn()
    except ValueError as exc:
        if str(exc) != 'Missing prospective publication' or len(data.calls) != n:
            raise
    else:
        raise AssertionError('Unpublished freeze allowed a target join')


def run_case(mode, path):
    spec = mock.specification(primary.Dataset(), mode)
    data = Fabricated(spec)
    public, work = output_paths(path/'one_pass')
    publication = mock.mock.MockPublication()
    data.check_cuts(spec)
    registration = primary.execute('register', spec, data, public, work, verify=publication)
    if data.calls or registration['outcomes_opened']:
        raise AssertionError('Registration joined targets')
    roles = registration['member_roles']
    held = [n for n in sorted(roles) if roles[n] == 'confirmation']
    held_score = lambda: primary.execute('confirm', spec, data, public, work, verify=publication)
    blocked_before_join(held_score, data)
    publication.publish(public/'registration.json')
    if mode == 'calibrated':
        primary.execute('calibrate', spec, data, public, work, verify=publication)
        if set(data.calls) != {n for n, role in roles.items() if role == 'calibration'}:
            raise AssertionError('Calibration touched the confirmation roster')
    blocked_before_join(held_score, data)
    publication.publish(public/'freeze.json')
    before = len(data.calls)
    data.fail_before = held[1]
    try:
        held_score()
    except Interrupted:
        if data.calls[before:] != held[:1] or len(list((work/'confirmation').glob('*.readout.json'))) != 1:
            raise AssertionError('Interrupted run did not preserve its first completed member')
    else:
        raise AssertionError('No synthetic interruption was exercised')

    def all_reports():
        c = held_score()
        a = study.attributed.report(c, public, work)
        return c, a, study.report(c, public, work)

    reports = all_reports()
    if data.calls[before:] != held:
        raise AssertionError('Expected only one join per confirmation member across resume')
    files = [public/name for name in ('confirmation.json', 'recipient_path_attribution.json',
                                     'stepwise_budget_sensitivity.json')]
    hashes = [storage.sha(p) for p in files]
    for p in files:
        p.rename(work/p.name)
    if reports != all_reports() or hashes != [storage.sha(p) for p in files] or data.calls[before:] != held:
        raise AssertionError('Scalars did not reconstruct without a new target join')

    old = mock.FrozenWithFabricatedTargets()
    p, w = output_paths(path/'primary_only')
    v = mock.mock.MockPublication()
    old.check_cuts(spec)
    primary.execute('register', spec, old, p, w, verify=v)
    v.publish(p/'registration.json')
    if mode == 'calibrated':
        primary.execute('calibrate', spec, old, p, w, verify=v)
        if study.read(p/'freeze.json')['thresholds'] != study.read(public/'freeze.json')['thresholds']:
            raise AssertionError('Secondary controls changed the calibrated rule')
    v.publish(p/'freeze.json')
    original = primary.execute('confirm', spec, old, p, w, verify=v)
    if any(original[k] != reports[0][k] for k in ('matched', 'release', 'official_risk_budget')):
        raise AssertionError('Primary scores or release changed')
    return dict(mode=mode, forecast_packets_verified=len(data.pairs), confirmation_members=len(held),
                confirmation_dates=reports[2]['source_date_count'], calibration_joins=before,
                held_joins=len(held), primary_metrics_exact=True, risk_and_release_exact=True,
                unpublished_read_blocked=True, interrupted_member_resume=True, scalar_replay_exact=True)


def run():
    with TemporaryDirectory(prefix='pneuma-one-pass-') as tmp:
        cases = [run_case(mode, Path(tmp)/mode) for mode in ('zero_shot', 'calibrated')]
    receipt = dict(result_source='fresh_runner_rehearsal_on_frozen_forecasts_with_fabricated_targets',
                   cases=cases, runner=storage.sha(Path(study.__file__)), rehearsal=storage.sha(Path(__file__)),
                   protocol=storage.sha(study.PROTOCOL),
                   forecast_stepwise_freeze=storage.sha(study.frozen.OUT/'athens.json'),
                   real_roles_assigned=False, real_future_positions_read=False, real_accuracy_scored=False,
                   original_2pct_unchanged=True, deployment_changed=False)
    storage.write_json(study.PUBLIC/'rehearsal.json', receipt)
    print(cases, flush=True)


if __name__ == '__main__':
    run()
