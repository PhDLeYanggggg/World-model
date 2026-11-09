"""Audit the clocks, source lineage and paired label support of Songdo Traffic v2."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/songdo_traffic_intake_v1'
LOCAL = ROOT / 'external_data/Songdo_Traffic_source_audit'
DATE = '2022-10-04'
RELEASE = 'https://zenodo.org/records/17924857'
PUBLISHER = {
    'A': (137057275, '4ed59b8c13c2b0d6cec7f8c175ad8f0e'),
    'B': (173633025, '5f2e7216dadc259cf8b2888313f1a1c8'),
    'C': (70053815, 'eadddba9317fb2ad7c6a568ac3b3036f'),
    'E': (84943460, 'b73586263f160a7954fc3010ab3867a5'),
}
COLS = ('Vehicle_ID', 'Local_Time', 'Drone_ID', 'Vehicle_Class', 'Visibility')
DT_MS = 1001. / 30.
FRAGMENT_GAP_MS = 1000
STRIDE, PAST, FUTURE = 12, 8, 12


def clock(ms):
    """Fit publisher-rate frame increments within one uninterrupted timestamp fragment."""
    t, inv = np.unique(ms, return_inverse=True)
    dt = np.diff(t)
    steps = np.rint(dt / DT_MS).astype(np.int64)
    frame = np.r_[0, steps.cumsum()]
    err = t - t[0] - frame * DT_MS
    phase_err = float(np.ptp(err) / 2.)
    gap_err = float(np.max(np.abs(dt - steps * DT_MS), initial=0.))
    ok = (len(t) >= 2 and (steps > 0).all() and gap_err <= .25 * DT_MS
          and phase_err <= .35 * DT_MS and not (dt > FRAGMENT_GAP_MS).any())
    return frame[inv], dict(max_centered_error_ms=phase_err, max_gap_rounding_error_ms=gap_err,
                            clock_supported=bool(ok))


def affine_clock(ms, frame):
    t, index = np.unique(ms, return_index=True)
    f = frame[index]
    x, y = f - f.mean(), t - t.mean()
    rate = float((x @ y) / (x @ x)) if (x @ x) > 0 else 0.
    residual = float(np.abs(y - rate * x).max())
    safe = bool((np.diff(f) > 0).all() and abs(rate / DT_MS - 1.) <= .001
                and residual <= .35 * DT_MS and not (np.diff(t) > FRAGMENT_GAP_MS).any()
                and np.max(np.abs(np.diff(t) - np.diff(f) * DT_MS), initial=0.) <= .25 * DT_MS)
    return dict(period_ms=rate, max_residual_ms=residual, accepted=safe)


def suffix_rows(vehicle, frame, stride=STRIDE):
    order = np.lexsort((frame, frame % stride, vehicle))
    ids, t = vehicle[order], frame[order]
    if not len(t):
        return t, t
    split = np.flatnonzero((np.diff(ids) != 0) | (np.diff(t) != stride)) + 1
    ends = np.r_[split, len(t)]
    return t, np.repeat(ends, np.diff(np.r_[0, ends])) - np.arange(len(t))


def coverage(ids, frames, max_frame, stride=STRIDE, history=PAST, future=FUTURE):
    t, remaining = suffix_rows(ids, frames, stride)
    past = remaining >= history
    horizon = (history + future - 1) * stride
    valid = past & (t + horizon <= max_frame)
    full = valid & (remaining >= history + future)
    observed = np.bincount(t[valid], minlength=max_frame+1)
    complete = np.bincount(t[full], minlength=max_frame+1)
    all_pairs = observed * (observed - 1) // 2
    full_pairs = complete * (complete - 1) // 2
    starts = []
    next_start = 0
    for start in np.flatnonzero(observed >= 2):
        if start >= next_start:
            starts.append(start)
            next_start = start + horizon + 1
    return dict(eligible_agent_windows=int(valid.sum()), complete_agent_windows=int(full.sum()),
                end_censored_history_windows=int((past & ~valid).sum()), eligible_pairs=int(all_pairs.sum()),
                complete_pairs=int(full_pairs.sum()), multiagent_queries=int((observed >= 2).sum()),
                nonoverlapping_queries=len(starts), nonoverlapping_queries_with_complete_pair=int((complete[starts] >= 2).sum()),
                nonoverlapping_full_roster=int(((observed[starts] == complete[starts]) & (observed[starts] >= 2)).sum()),
                nonoverlapping_eligible_pairs=int(all_pairs[starts].sum()),
                nonoverlapping_complete_pairs=int(full_pairs[starts].sum()))


def sum_fields(items):
    result = Counter()
    for r in items:
        result.update(r)
    return dict(result)


def session(table, name, site, *, drift=False):
    seconds = pd.to_timedelta(table['Local_Time'], errors='raise')
    if seconds.isna().any():
        raise ValueError(f'Missing timestamps in {name}')
    table['ms'] = seconds.to_numpy().astype('timedelta64[ms]').astype(np.int64)
    if (not table['Visibility'].isin((0, 1)).all() or (table['ms'] < 0).any()
            or (table['ms'] >= 86400000).any()):
        raise ValueError(f'Unexpected visibility or time in {name}')
    per_drone = []
    for drone, data in table.groupby('Drone_ID'):
        ms = data['ms'].to_numpy()
        order = np.argsort(ms, kind='stable')
        ids = data['Vehicle_ID'].to_numpy()[order]
        flags = data['Visibility'].to_numpy(dtype=bool)[order]
        ms = ms[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ms) > FRAGMENT_GAP_MS)+1, len(ms)]
        for number, (lo, hi) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
            t, a, vis = ms[lo:hi], ids[lo:hi], flags[lo:hi]
            frame, fit = clock(t)
            raw_unique = np.unique(np.column_stack((a, t)), axis=0).shape[0]
            unique = np.unique(np.column_stack((a, frame)), axis=0).shape[0]
            audit = dict(drone=int(drone), fragment=number, site=site, session=name,
                         rows=len(a), first_ms=int(t[0]), last_ms=int(t[-1]), tracks=len(np.unique(a)),
                         exact_duplicates=len(a)-raw_unique, frame_collisions=raw_unique-unique, **fit)
            if drift and not fit['clock_supported']:
                audit['affine_sensitivity'] = affine_clock(t, frame)
            if ((fit['clock_supported'] or audit.get('affine_sensitivity', {}).get('accepted'))
                    and len(a) == unique):
                audit['any_visibility'] = coverage(a, frame, int(frame.max()))
                audit['full_visibility'] = coverage(a[vis], frame[vis], int(frame.max()))
            per_drone.append(audit)
    cross_drone_ids = table.groupby('Vehicle_ID')['Drone_ID'].nunique()
    return dict(site=site, session=name, rows=len(table), vehicle_ids=int(len(cross_drone_ids)),
                vehicle_ids_on_multiple_drones=int((cross_drone_ids > 1).sum()),
                type_rows={str(k): int(n) for k, n in table['Vehicle_Class'].value_counts().items()}, drones=per_drone)


def overlap(streams):
    """Potential cross-site shared video, even if different CSV IDs were assigned."""
    by_drone = defaultdict(list)
    for r in streams:
        by_drone[r['drone']].append(r)
    found = []
    for ds in by_drone.values():
        for i, a in enumerate(ds):
            for b in ds[i+1:]:
                ms = min(a['last_ms'], b['last_ms']) - max(a['first_ms'], b['first_ms'])
                if a['site'] != b['site'] and ms >= 0:
                    found.append(dict(drone=a['drone'], sessions=[a['session'], b['session']],
                                      fragments=[a['fragment'], b['fragment']], overlap_ms=ms))
    return sorted(found, key=lambda r: (r['drone'], r['sessions']))


def digest(path):
    with path.open('rb') as f:
        md5 = hashlib.file_digest(f, 'md5', _bufsize=2**20).hexdigest()
        f.seek(0)
        return md5, hashlib.file_digest(f, 'sha256', _bufsize=2**20).hexdigest()


def load(archive, name):
    with archive.open(name) as source:
        return pd.read_csv(source, usecols=COLS, dtype={
            'Vehicle_ID': 'int64', 'Drone_ID': 'int64', 'Vehicle_Class': 'int64',
            'Visibility': 'int64', 'Local_Time': 'str'})


def run(directory):
    sessions, archives = [], {}
    for site, (size, md5) in PUBLISHER.items():
        filename = f'{DATE}_{site}.zip'
        path = directory / filename
        received, sha = digest(path)
        if path.stat().st_size != size or received != md5:
            raise ValueError(f'Publisher bytes do not match: {filename}')
        archives[filename] = dict(bytes=size, publisher_md5=md5, sha256=sha)
        with zipfile.ZipFile(path) as archive:
            files = sorted(n for n in archive.namelist() if not n.endswith('/'))
            expected = {f'{DATE}_{site}_{phase}{number}.csv'
                        for phase in ('AM', 'PM') for number in range(1, 6)}
            if set(files) != expected or len(files) != len(expected):
                raise ValueError(f'Unrecognized member roster in {filename}: {files}')
            for name in files:
                s = session(load(archive, name), name, site)
                sessions.append(s)
                print(name, s['rows'], flush=True)
    streams = [r for s in sessions for r in s['drones']]
    def summary(items):
        out = dict(streams=len(items), rows=sum(r['rows'] for r in items),
                   grid_pass=sum(r['clock_supported'] for r in items),
                   scored_for_presence=sum('any_visibility' in r for r in items),
                   rows_scored_for_presence=sum(r['rows'] for r in items if 'any_visibility' in r),
                   max_grid_error_ms=max(r['max_centered_error_ms'] for r in items),
                   exact_duplicates=sum(r['exact_duplicates'] for r in items),
                   frame_collisions=sum(r['frame_collisions'] for r in items))
        out.update({k: sum_fields(r[k] for r in items if k in r) for k in ('any_visibility', 'full_visibility')})
        return out
    return dict(result_source='fresh_run', release=RELEASE, license='CC BY 4.0', archives=archives,
                selection='first date and first four alphabetically available physical intersection IDs; all session CSVs',
                coordinate_fields_read=False, forecast_outcomes_scored=False, independent_confirmation=False,
                new_split_assigned=False, reserved_outcomes_opened=False, source='Songdo single acquisition campaign',
                observed=PAST, future=FUTURE, nominal_frame_step=STRIDE, conditional_grid_ms=DT_MS,
                fragment_gap_ms=FRAGMENT_GAP_MS, max_gap_error_fraction=.25, max_centered_error_fraction=.35,
                site_summaries={s: summary([r for r in streams if r['site'] == s]) for s in PUBLISHER},
                totals=summary(streams), sessions=sessions, same_drone_cross_site_overlap=overlap(streams))


def sensitivity(directory, primary_path):
    primary = json.loads(primary_path.read_text())
    records = []
    for site in PUBLISHER:
        filename = f'{DATE}_{site}.zip'
        path = directory / filename
        md5, sha = digest(path)
        if (md5, sha) != (PUBLISHER[site][1], primary['archives'][filename]['sha256']):
            raise ValueError(f'Cannot replay changed archive: {filename}')
        with zipfile.ZipFile(path) as archive:
            for old in (s for s in primary['sessions'] if s['site'] == site):
                if all(x['clock_supported'] for x in old['drones']):
                    continue
                name = old['session']
                new = session(load(archive, name), name, site, drift=True)
                for prev, now in zip(old['drones'], new['drones'], strict=True):
                    unchanged = {k: now[k] for k in prev}
                    if unchanged != prev:
                        raise ValueError(f'Primary audit no longer replays: {name}')
                    if not prev['clock_supported']:
                        records.append(now)
                print('drift', name, flush=True)
    summary = {}
    for site in PUBLISHER:
        attempted = [r for r in records if r['site'] == site]
        recovered = [r for r in attempted if 'full_visibility' in r]
        summary[site] = dict(unsupported_fragments=len(attempted), recovered_fragments=len(recovered),
            recovered_rows=sum(r['rows'] for r in recovered),
            additional_strict_coverage=sum_fields(r['full_visibility'] for r in recovered))
    return dict(primary_sha256=hashlib.sha256(primary_path.read_bytes()).hexdigest(),
                role='diagnostic_only_same_integer_grid_no_change_to_primary',
                rate_tolerance_relative=.001, records=records, sites=summary,
                forecast_outcomes_scored=False, new_split_assigned=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, default=LOCAL)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--clock-sensitivity', type=Path, help='Existing primary receipt; check its unsupported clocks')
    args = ap.parse_args()
    if args.clock_sensitivity:
        result = sensitivity(args.directory, args.clock_sensitivity)
        target = args.output or PUBLIC / 'clock_sensitivity.json'
    else:
        result = run(args.directory)
        target = args.output or PUBLIC / 'analysis.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if target.exists() and target.read_text() != encoded:
        raise ValueError('Audit replay differs from the existing receipt')
    if not target.exists():
        pending = target.with_suffix('.pending')
        pending.write_text(encoded)
        pending.replace(target)
    print(json.dumps(result.get('totals', result.get('sites')), indent=2), flush=True)


if __name__ == '__main__':
    main()
