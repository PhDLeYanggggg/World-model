"""Registered pNEUMA readout. Actual outcomes require a published protocol."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_pneuma_prefix_pilot as pilot
from src.evaluation import m3w_pneuma_future_join as future
from src.evaluation import m3w_pneuma_matched_readout as readout
from src.evaluation import m3w_pneuma_protocol as protocol
from src.evaluation.m3w_songdo_protocol import positive_complete_cv_ade
from src.evaluation import m3w_sparse_pair_bounds as sparse


np, storage = pilot.np, pilot.storage
OUT = pilot.OUT.parent/'pneuma_confirmation_v1'
REHEARSAL = pilot.OUT.parent/'pneuma_confirmation_rehearsal_v1'
WORK = pilot.LOCAL/'confirmation'


def verify_hash(path, digest):
    if storage.sha(path) != digest:
        raise ValueError(f'Frozen file changed: {path.name}')


class Dataset:
    def __init__(self):
        pref, bank, menu = (pilot.OUT/f'{x}_receipt.json' for x in ('prefix', 'forecast', 'matched'))
        inputs = pilot.source.PUBLIC/'samples.json'
        a, b, c, d = (pilot.read(x) for x in (pref, bank, menu, inputs))
        if (b['binding']['prefix_receipt'] != storage.sha(pref) or a['binding']['samples'] != storage.sha(inputs)
                or c['binding']['prefix'] != storage.sha(pref) or c['binding']['forecast'] != storage.sha(bank)
                or not c['equal_radius_verified'] or not c['distinct_recipients_verified']
                or c['predictive_roles_assigned'] or c['forecast_accuracy_scored'] or c['future_coordinates_loaded']
                or set(a['files']) != set(b['files']) or set(a['files']) != set(d['members'])
                or c['query_seed_views'] != 1929 or set(c['seeds']) != set(readout.SEEDS)):
            raise ValueError('Require the complete unscored pilot receipts')
        self.inputs, self.bank, self.menu, self.source = a['files'], b['files'], c['files'], d['members']
        self.dates = {n: r['day'] for n, r in self.source.items()}
        self.binding = {p.name: storage.sha(p) for p in (pref, bank, menu, inputs)}
        for mod in (Path(__file__), Path(protocol.__file__), Path(readout.__file__), Path(future.__file__), Path(sparse.__file__)):
            self.binding[str(mod.resolve().relative_to(pilot.source.ROOT))] = storage.sha(mod)

    def _forecasts(self, name):
        stem = Path(name).stem
        p, q = (pilot.LOCAL/f'{stem}.{s}.npz' for s in ('prefix', 'forecast'))
        verify_hash(p, self.inputs[name]['prefix_sha256'])
        verify_hash(q, self.bank[name]['forecast_sha256'])
        with np.load(p, allow_pickle=False) as a, np.load(q, allow_pickle=False) as b:
            return {k: a[k] for k in a.files}, {k: b[k] for k in ('cv', 'ols4', 'median4')}

    def _labels(self, name, prefix):
        raw = pilot.source.LOCAL/name
        verify_hash(raw, self.source[name]['sha256'])
        with raw.open(encoding='utf-8-sig', newline='') as f:
            return future.join(f, prefix, pilot.STRIDE)

    def check_cuts(self, spec):
        if spec['mode'] == 'zero_shot':
            p = REHEARSAL/'donor_cuts.json'
            verify_hash(p, spec['donor_cut_source'])
            cuts = pilot.read(p)
            if cuts['receiver_labels_opened'] or any(spec[k] != cuts[k] for k in ('easy_cut', 'hard_cut')):
                raise ValueError('Zero-shot cutoffs do not match the prior-only donor receipt')
            self.binding['donor_cuts'] = storage.sha(p)

    def calibration_errors(self, name):
        prefix, prediction = self._forecasts(name)
        y, valid = self._labels(name, prefix)
        return positive_complete_cv_ade(prediction['cv'], y, valid).tolist()

    def score(self, name, easy, hard):
        prefix, forecast = self._forecasts(name)
        items = {}
        for seed in readout.SEEDS:
            key = f'{Path(name).stem}_{seed}'
            p = pilot.LOCAL/'matched'/f'{key}.npz'
            row = self.menu[key]
            r = row['identity']
            if (r['name'] != name or r['seed'] != seed or r['prefix_sha256'] != self.inputs[name]['prefix_sha256']
                    or r['forecast_sha256'] != self.bank[name]['forecast_sha256']
                    or row['queries'] != self.inputs[name]['queries']):
                raise ValueError('Unbound prospective slate')
            verify_hash(p, row['proposals_sha256'])
            with np.load(p, allow_pickle=False) as z:
                items[str(seed)] = {k: z[k] for k in z.files}
        y, valid = self._labels(name, prefix)
        return {s: readout.score(forecast['cv'], forecast['ols4'], forecast['median4'], y, valid, prefix['offsets'],
                                 p, easy_cut=easy, hard_cut=hard) for s, p in items.items()}


def require_published(path):
    root = pilot.source.ROOT
    rel = path.resolve().relative_to(root)
    line = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=root, timeout=20).split()
    if len(line) != 2 or line[1] != b'refs/heads/main':
        raise ValueError('Cannot verify the prospective remote record')
    found = subprocess.check_output(['git', 'show', f'{line[0].decode()}:{rel}'], cwd=root, timeout=10)
    if path.read_bytes() != found:
        raise ValueError('Publish the exact immutable registration/freeze before opening outcomes')


def heartbeat(work, phase, n, total):
    p = work/'heartbeat.pending'
    p.write_text(json.dumps(dict(pid=os.getpid(), phase=phase, done=n, total=total))+'\n')
    p.replace(work/'heartbeat.json')


def execute(command, spec, dataset, public, work, *, verify=require_published):
    roles = protocol.assignments(spec, dataset.dates)
    registration, freeze = (public/f'{x}.json' for x in ('registration', 'freeze'))
    identity = dict(protocol=spec, data_binding=dataset.binding, dates=dataset.dates, member_roles=roles)
    if command == 'register':
        storage.write_json(registration, identity)
        if spec['mode'] == 'zero_shot':
            storage.write_json(freeze, dict(registration=storage.sha(registration), rule=protocol.ZERO_RULE,
                easy_cut=spec['easy_cut'], hard_cut=spec['hard_cut'], thresholds=None, label_access=False))
        return dict(member_roles=roles, outcomes_opened=False)
    if command not in ('calibrate', 'confirm') or not registration.is_file() or pilot.read(registration) != identity:
        raise ValueError('Require the exact prospective registration')
    verify(registration)
    role = 'calibration' if command == 'calibrate' else 'confirmation'
    names = [n for n in sorted(roles) if roles[n] == role]
    if not names:
        raise ValueError('No files in the authorised role')
    reg = storage.sha(registration)
    local = work/role
    local.mkdir(exist_ok=True)
    if role == 'confirmation':
        if not freeze.is_file():
            raise ValueError('Freeze the calibration or donor operating rule before confirmation')
        verify(freeze)
        cut = pilot.read(freeze)
        if cut['registration'] != reg or cut['rule'] != spec['rule'] or not protocol.fixed_cuts(cut):
            raise ValueError('Risk rule does not bind the registered run')
        if spec['mode'] == 'zero_shot' and (cut['thresholds'] is not None or
                any(cut[k] != spec[k] for k in ('easy_cut', 'hard_cut')) or cut['label_access']):
            raise ValueError('Zero-shot cannot use any Athens calibration')
        if spec['mode'] == 'calibrated':
            if not cut['label_access'] or cut['confirmation_outcomes_opened'] or set(cut['calibration_members']) != {
                    n for n, v in roles.items() if v == 'calibration'}:
                raise ValueError('Calibration receipt does not match the assigned dates')
            protocol.check_thresholds(cut['thresholds'])
    elif freeze.is_file():
        existing = pilot.read(freeze)
        if existing['registration'] != reg or existing['rule'] != spec['rule']:
            raise ValueError('Stale calibration freeze')
        return existing
    else:
        data = {}
        for i, n in enumerate(names, 1):
            p = local/f'{Path(n).stem}.errors.json'
            if not p.exists():
                storage.write_json(p, dict(registration=reg, member=n, errors=dataset.calibration_errors(n)))
            item = pilot.read(p)
            if item['registration'] != reg or item['member'] != n:
                raise ValueError('Calibration inputs changed')
            data[n] = item['errors']
            heartbeat(work, 'cutoffs', i, len(names))
        cut = protocol.fit_native_cutoffs(data)
    bind = dict(registration=reg, freeze=storage.sha(freeze) if role == 'confirmation' else None, role=role,
                easy_cut=cut['easy_cut'], hard_cut=cut['hard_cut'])
    report_path = public/'confirmation.json'
    if role == 'confirmation' and report_path.is_file():
        result = pilot.read(report_path)
        if result['binding'] != bind or result['mode'] != spec['mode'] or result['members'] != names:
            raise ValueError('Existing confirmation belongs to a different prospective run')
        return result
    records = {}
    for i, n in enumerate(names, 1):
        path = local/f'{Path(n).stem}.readout.json'
        if not path.exists():
            storage.write_json(path, dict(binding=bind, member=n, score=dataset.score(n, cut['easy_cut'], cut['hard_cut'])))
        item = pilot.read(path)
        if item['binding'] != bind or item['member'] != n:
            raise ValueError('Cached score does not bind the frozen role and rule')
        records[n] = item['score']
        heartbeat(work, role, i, len(names))
    if role == 'calibration':
        result = dict(registration=reg, rule=spec['rule'], easy_cut=cut['easy_cut'], hard_cut=cut['hard_cut'],
                      cutoff_support=cut['support_rows'], calibration_members=names, thresholds=protocol.thresholds(records),
                      label_access=True, confirmation_outcomes_opened=False)
        storage.write_json(freeze, result)
        return result
    dates = {n: dataset.dates[n] for n in names}
    release = (protocol.calibrated_report(records, dates, cut['thresholds']) if spec['mode'] == 'calibrated'
               else 'not_run_zero_shot_fixed_count_only')
    result = dict(binding=bind, mode=spec['mode'], members=names, dates=dates,
                  matched=readout.summarize(records, dates), release=release, official_risk_budget=.02,
                  roles_frozen_before_read=True, new_fit=False, deployment_changed=False)
    storage.write_json(report_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'calibrate', 'confirm'))
    p.add_argument('--protocol', required=True, type=Path)
    a = p.parse_args()
    spec = pilot.read(a.protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = Dataset()
        protocol.assignments(spec, dataset.dates)
        dataset.check_cuts(spec)
        result = execute(a.phase, spec, dataset, OUT, WORK)
    print(json.dumps({k: v for k, v in result.items() if k not in ('matched', 'release')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
