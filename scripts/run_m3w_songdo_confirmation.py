"""Run a registered Songdo evaluation without opening another role's outcomes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_first_day as first
from src.evaluation import (
    m3w_native_metrics, m3w_opendd_location, m3w_selection_forecast_readout,
    m3w_songdo_future_join as join, m3w_songdo_protocol as protocol,
)

pack, audit, np = first.pack, first.audit, first.np
PUBLIC = first.PUBLIC.parent / 'songdo_matched_proposals_v1'
PACKETS = first.PRIVATE.parent / PUBLIC.name
RUN = first.PRIVATE.parent / 'songdo_confirmation_v1'


def future_rows(archive, name, index):
    """Parse XY only for original CSV rows referenced by the frozen label mask."""
    wanted = np.unique(index['future_index'][index['future_mask']])
    if not len(wanted):
        return dict(row=wanted, xy=np.empty((0, 2)), agent=wanted, drone=wanted, ms=wanted, visible=wanted)
    keep = np.zeros(int(wanted[-1])+2, dtype=bool)
    keep[0] = True
    keep[wanted+1] = True
    with archive.open(name) as file:
        d = audit.pd.read_csv(file, usecols=('Vehicle_ID', 'Drone_ID', 'Local_Time', 'Visibility', *pack.XY),
                              skiprows=lambda k: k >= len(keep) or not keep[k],
                              dtype={'Local_Time': 'str', 'Local_X': 'float64', 'Local_Y': 'float64'})
    if len(d) != len(wanted):
        raise ValueError(f'Label rows missing: {name}')
    ms = audit.pd.to_timedelta(d['Local_Time'], errors='raise').to_numpy().astype('timedelta64[ms]').astype(np.int64)
    return dict(row=wanted, xy=d[list(pack.XY)].to_numpy(), agent=d['Vehicle_ID'].to_numpy(),
                drone=d['Drone_ID'].to_numpy(), ms=ms, visible=d['Visibility'].to_numpy())


def checked(path, digest):
    if pack.sha(path) != digest:
        raise ValueError(f'Frozen file changed: {path}')
    return path


class Songdo:
    def __init__(self):
        public = dict(archives=first.PUBLIC / 'archives.json', prefix=first.PUBLIC / 'input_receipt.json',
                      groups=first.PUBLIC / 'lineage.json', bank=first.PUBLIC / 'vehicle_bank.json',
                      proposals=PUBLIC / 'receipt.json')
        self.binding = {k: pack.sha(p) for k, p in public.items()}
        self.archives, source, self.lineage, bank, proposals = (first.read(p) for p in public.values())
        self.binding.update({Path(m.__file__).stem: pack.sha(Path(m.__file__)) for m in (
            protocol, protocol.readout, join, m3w_native_metrics, m3w_opendd_location,
            m3w_selection_forecast_readout, pack, audit)})
        self.binding['runner'] = pack.sha(Path(__file__))
        self.source, self.bank, self.proposals = source['sessions'], bank['sessions'], proposals['sessions']
        self.sites = {name: item['site'] for name, item in self.source.items()}
        if (not len(self.sites) == 200 or set(self.sites) != set(self.proposals) or set(self.sites) != set(self.bank)
                or source['binding']['archive_manifest'] != self.binding['archives']
                or self.lineage['input_receipt_sha256'] != self.binding['prefix']
                or proposals['binding']['prefix_receipt'] != self.binding['prefix']
                or proposals['binding']['candidate_bank'] != self.binding['bank']
                or proposals['binding']['lineage'] != self.binding['groups']
                or bank['binding']['prefix'] != self.binding['prefix']
                or bank['binding']['lineage'] != self.binding['groups']
                or any(r['split_assigned'] for r in (source, self.lineage, bank, proposals))
                or source['future_coordinates_parsed'] or proposals['future_coordinates_loaded']
                or proposals['forecasts_scored'] or not proposals['original_2pct_unchanged']
                or any(self.proposals[n]['site'] != s or self.bank[n]['site'] != s
                       or self.bank[n]['prefix_sha'] != self.source[n]['prefix_sha'] for n, s in self.sites.items())):
            raise ValueError('Expected the unchanged split-neutral first-day inputs')
        self.verified_archives = set()

    def targets(self, name, cache, prefix, item):
        index = checked(cache / f'{Path(name).stem}_label_index.npz', item['label_index_sha'])
        file = f"{audit.DATE}_{item['site']}.zip"
        path = audit.LOCAL / file
        if file not in self.verified_archives:
            checked(path, self.archives['files'][file]['sha256'])
            self.verified_archives.add(file)
        with np.load(index, allow_pickle=False) as ix, audit.zipfile.ZipFile(path) as archive:
            index = {k: ix[k] for k in ('future_index', 'future_mask', 'origin')}
            if int(index['future_mask'].all(1).sum()) != item['complete_agent_futures']:
                raise ValueError('Frozen complete-label coverage changed')
            target = join.displacements(prefix, index, future_rows(archive, name, index),
                                        stride=audit.STRIDE, tick_ms=audit.DT_MS)
        return target, index['future_mask']

    def observed(self, name):
        stem = Path(name).stem
        item = self.source[name]
        cache = pack.CACHE if item['cache_role'] == 'previous' else first.PRIVATE
        prefix = checked(cache / f'{stem}_prefix.npz', item['prefix_sha'])
        bank = checked(first.PRIVATE / 'vehicle_bank' / f'{stem}.npz', self.bank[name]['sha256'])
        return item, cache, prefix, bank

    def cutoff_errors(self, name):
        item, cache, prefix, bank = self.observed(name)
        with np.load(prefix, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b:
            np.testing.assert_array_equal(p['offsets'], b['offsets'])
            target, valid = self.targets(name, cache, p, item)
            return protocol.positive_complete_cv_ade(b['cv'], target, valid).tolist()

    def score(self, name, spec, thresholds):
        item, cache, prefix, bank = self.observed(name)
        stem = Path(name).stem
        proposal = checked(PACKETS / f'{stem}.npz', self.proposals[name]['packet_sha256'])
        with np.load(prefix, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b, \
                np.load(proposal, allow_pickle=False) as z:
            cv, ols, offsets = b['cv'], b['ols4'], b['offsets']
            np.testing.assert_array_equal(offsets, p['offsets'])
            if len(cv) != item['agents'] or len(offsets)-1 != item['queries']:
                raise ValueError('Forecast roster changed')
            options = {}
            for seed in protocol.SEEDS:
                k = f'seed{seed}_'
                has_proposal = k+'bounds' in z
                if has_proposal != (self.proposals[name]['seeds'][str(seed)] is not None):
                    raise ValueError('Frozen matched eligibility changed')
                options[seed] = (dict(bounds=z[k+'bounds'], radius=float(z[k+'radius'].item()),
                    arms={a: (z[k+a+'_rows'], z[k+a+'_paths']) for a in protocol.ARMS}),
                    {a: float(z[k+a+'_rank_score'].item()) for a in protocol.ARMS}) if has_proposal else (None, {})
            target, valid = self.targets(name, cache, p, item)
            return protocol.score_session(cv, ols, offsets, target, valid, options, spec, thresholds)


def heartbeat(run, role, done, total, last):
    tmp = run / 'heartbeat.pending'
    progress = dict(pid=os.getpid(), role=role, completed=done, total=total, last=last)
    tmp.write_text(json.dumps(progress)+'\n')
    tmp.replace(run / 'heartbeat.json')
    print(progress, flush=True)


def fit_calibration_cutoffs(run, files, dataset, registration_sha):
    output = run / 'calibration_cutoffs.json'
    identity = dict(registration_sha256=registration_sha, rule=protocol.CALIBRATION_CUTOFF_RULE,
                    calibration_flights=sorted(files))
    if not output.exists():
        work = run / 'calibration_errors'
        work.mkdir(exist_ok=True)
        errors = {}
        for i, name in enumerate(sorted(files), start=1):
            cp = work / (Path(name).stem+'.json')
            key = dict(registration_sha256=registration_sha, flight=name)
            if not cp.exists():
                pack.write_json(cp, dict(**key, errors=dataset.cutoff_errors(name)))
            r = first.read(cp)
            if any(r[k] != v for k, v in key.items()):
                raise ValueError('Calibration CV errors do not bind the approved source')
            errors[name] = r['errors']
            heartbeat(run, 'calibration_cutoffs', i, len(files), name)
        pack.write_json(output, dict(**identity, **protocol.fit_native_cutoffs(errors)))
    result = first.read(output)
    if any(result[k] != v for k, v in identity.items()) or not protocol.fixed_cuts(result):
        raise ValueError('Native cutoffs do not bind the approved calibration roster')
    return result, pack.sha(output)


def execute(run, command, spec, dataset):
    if command not in ('register', 'calibrate', 'confirm'):
        raise ValueError('Unknown evaluation phase')
    roles = protocol.assignments(spec, dataset.lineage['candidate_source_components'], dataset.sites)
    registration = run / 'registration.json'
    identity = dict(spec=spec, binding=dataset.binding, flight_roles=roles)
    # Normalize tuples before checking an existing JSON registration.
    identity = json.loads(json.dumps(identity, allow_nan=False))
    if command == 'register':
        pack.write_json(registration, identity)
        return dict(registered_flights=len(roles), registration_sha256=pack.sha(registration), outcomes_opened=False)
    if first.read(registration) != identity:
        raise ValueError('Evaluation inputs or approved protocol changed after registration')
    role = 'calibration' if command == 'calibrate' else 'confirmation'
    files = {f: component for f, (r, component) in roles.items() if r == role}
    if not files:
        raise ValueError(f'No flights assigned to {role}')
    registration_sha = pack.sha(registration)
    freeze = run / 'freeze.json'
    thresholds, freeze_sha = None, None
    only = protocol.matched_only(spec)
    dynamic = protocol.cutoff_rule(spec) == protocol.CALIBRATION_CUTOFF_RULE
    cutoff_binding = {}
    effective_spec = spec
    if role == 'confirmation':
        if spec['mode'] == 'zero_shot':
            entry = dict(evaluation='matched_only') if only else dict(thresholds=spec['release']['thresholds'])
            pack.write_json(freeze, dict(registration_sha256=registration_sha, **entry))
        bound = first.read(freeze)
        if bound['registration_sha256'] != registration_sha:
            raise ValueError('Release policy must bind the registered experiment')
        freeze_sha = pack.sha(freeze)
        if only:
            if bound.get('evaluation') != 'matched_only' or 'thresholds' in bound:
                raise ValueError('Matched-only run cannot use a release policy')
        else:
            thresholds = bound['thresholds']
            protocol.check_thresholds(thresholds)
        if dynamic:
            checked(run / 'calibration_cutoffs.json', bound['calibration_cutoffs_sha256'])
            cutoff_binding = {k: bound[k] for k in ('easy_cut', 'hard_cut', 'calibration_cutoffs_sha256')}
    elif dynamic:
        if freeze.exists() and not (run / 'calibration_cutoffs.json').is_file():
            raise ValueError('Existing freeze is missing its original calibration cutoffs')
        cuts, digest = fit_calibration_cutoffs(run, files, dataset, registration_sha)
        cutoff_binding = dict(easy_cut=cuts['easy_cut'], hard_cut=cuts['hard_cut'], calibration_cutoffs_sha256=digest)
    if cutoff_binding:
        effective_spec = dict(**spec, **cutoff_binding)
    out = freeze if role == 'calibration' else run / 'confirmation.json'
    header = dict(registration_sha256=registration_sha, release_freeze_sha256=freeze_sha, role=role, **cutoff_binding)
    if out.exists():
        cached = first.read(out)
        if any(cached.get(k) != v for k, v in header.items()):
            raise ValueError('Completed result belongs to a different protocol')
        return cached
    work = run / role
    work.mkdir(exist_ok=True)
    records = {}
    for i, f in enumerate(sorted(files), start=1):
        cp = work / (Path(f).stem+'.json')
        key = dict(**header, flight=f)
        if not cp.exists():
            pack.write_json(cp, dict(**key, scores=dataset.score(f, effective_spec, thresholds)))
        rec = first.read(cp)
        if any(rec[k] != v for k, v in key.items()):
            raise ValueError(f'Flight checkpoint does not match the registered experiment: {f}')
        records[f] = rec['scores']
        heartbeat(run, role, i, len(files), f)
    result = dict(**header, **(protocol.calibration(records) if role == 'calibration'
                               else protocol.summarize(records, files, only=only)))
    pack.write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'calibrate', 'confirm'))
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(RUN, args.command, first.read(args.protocol), Songdo())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
