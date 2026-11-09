"""Build a split-neutral input pack and a separate label index, without loading future XY."""

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_traffic as audit
from src.world_model.m3w_wepdtof_matched_policy import analytic

np, pd = audit.np, audit.pd
PUBLIC = audit.PUBLIC.parent / 'songdo_prefix_pack_v1'
CACHE = audit.ROOT / 'data/stage_cvpr2027_experiments/songdo_prefix_pack_v1'
XY = ('Local_X', 'Local_Y')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, obj):
    encoded = json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f'Existing receipt differs: {path}')
        return
    tmp = path.with_suffix('.pending')
    tmp.write_text(encoded)
    tmp.replace(path)


def exact_indices(ids, frame, csv_rows, max_frame):
    """Earliest past-eligible query, retaining its full past-defined roster."""
    history, future, stride = audit.PAST, audit.FUTURE, audit.STRIDE
    ordered = np.lexsort((frame, frame % stride, ids))
    t, n = audit.suffix_rows(ids, frame, stride)
    people = ids[ordered]
    span = stride * (history + future - 1)
    good = (n >= history) & (t + span <= max_frame)
    online = np.bincount(t[good], minlength=max_frame+1)
    starts, after = [], 0
    for start in np.flatnonzero(online >= 2):
        if start >= after:
            starts.append(start)
            after = start + span + 1
    s = good & np.isin(t, starts)
    agent, begin = people[s], t[s]
    order = np.lexsort((agent, begin))
    agent, begin = agent[order], begin[order]

    # Dense identity codes preserve gaps in the clock without a dense track matrix.
    keys, track = np.unique(ids, return_inverse=True)
    width = max_frame + 1
    key = track * width + frame
    lookup = np.argsort(key)
    key = key[lookup]
    if np.any(np.diff(key) == 0):
        raise ValueError('Duplicate track and frame')
    queries = begin[:, None] + stride * np.arange(history + future)
    target = np.searchsorted(keys, agent)[:, None] * width + queries
    k = np.searchsorted(key, target)
    safe = np.minimum(k, max(len(key)-1, 0))
    if not len(key):
        return agent, begin, queries, np.full(queries.shape, -1, dtype=np.int64)
    found = (k < len(key)) & (key[safe] == target)
    indices = np.where(found, csv_rows[lookup[safe]], -1)
    if (indices[:, :history] < 0).any():
        raise ValueError('Past support was lost during coordinate join')
    return agent, begin, queries, indices


def fragments(table, expected):
    ms = pd.to_timedelta(table['Local_Time'], errors='raise').to_numpy().astype('timedelta64[ms]').astype(np.int64)
    receipts = {(r['drone'], r['fragment']): r for r in expected}
    for drone, df in table.groupby('Drone_ID'):
        ix = df.index.to_numpy()
        ix = ix[np.argsort(ms[ix], kind='stable')]
        for k, rows in enumerate(np.split(ix, np.flatnonzero(np.diff(ms[ix]) > audit.FRAGMENT_GAP_MS)+1)):
            r = receipts.pop((int(drone), k))
            if (len(rows), int(ms[rows[0]]), int(ms[rows[-1]])) != (r['rows'], r['first_ms'], r['last_ms']):
                raise ValueError('Session fragment no longer matches the intake')
            if 'full_visibility' not in r:
                continue
            frame, status = audit.clock(ms[rows])
            if not status['clock_supported']:
                raise ValueError('Primary clock did not replay')
            visible = table['Visibility'].to_numpy()[rows] == 1
            a, s, grid, index = exact_indices(table['Vehicle_ID'].to_numpy()[rows][visible],
                                               frame[visible], rows[visible], int(frame.max()))
            yield r, a, s, grid, index, ms
    if receipts:
        raise ValueError('An audited fragment did not replay')


def past_coordinates(archive, name, table, past):
    """Tokenizer skips all non-prefix rows before numeric coordinate conversion."""
    select = np.unique(past)
    wanted = np.zeros(len(table), dtype=bool)
    wanted[select] = True
    if not len(select):
        return np.empty(past.shape + (2,), dtype=float)
    if select[0] < 0:
        raise ValueError('Invalid past row')
    def skip(row):
        return row != 0 and (row > len(wanted) or not wanted[row-1])
    with archive.open(name) as source:
        d = pd.read_csv(source, usecols=(*audit.COLS, *XY), skiprows=skip,
                        dtype={'Local_Time': 'str', 'Local_X': 'float64', 'Local_Y': 'float64'})
    if not d[list(audit.COLS)].reset_index(drop=True).equals(table[list(audit.COLS)].iloc[select].reset_index(drop=True)):
        raise ValueError('Selective coordinate parser did not reproduce the audited metadata')
    values = d[list(XY)].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite past coordinate')
    return values[np.searchsorted(select, past)]


def arrays(archive, source):
    name = source['session']
    table = audit.load(archive, name)
    parts = list(fragments(table, source['drones']))
    pack = defaultdict(list)
    for r, a, s, grid, index, ms in parts:
        _, begin, counts = np.unique(s, return_index=True, return_counts=True)
        stop = begin + counts
        complete = (index[:, audit.PAST:] >= 0).all(axis=1)
        stats = Counter(nonoverlapping_queries=len(begin))
        for lo, hi in zip(begin, stop, strict=True):
            known = int(complete[lo:hi].sum())
            size = hi - lo
            stats.update(nonoverlapping_queries_with_complete_pair=int(known >= 2),
                         nonoverlapping_full_roster=int(known == size),
                         nonoverlapping_eligible_pairs=int(size*(size-1)//2),
                         nonoverlapping_complete_pairs=known*(known-1)//2)
        for k in ('nonoverlapping_queries', 'nonoverlapping_queries_with_complete_pair', 'nonoverlapping_full_roster',
                  'nonoverlapping_eligible_pairs', 'nonoverlapping_complete_pairs'):
            if stats[k] != r['full_visibility'][k]:
                raise ValueError(f'Past roster disagrees with the source audit: {k}')
        pack['agent'].append(a)
        pack['counts'].append(counts)
        pack['past_index'].append(index[:, :audit.PAST])
        pack['past_frame'].append(grid[:, :audit.PAST])
        pack['past_ms'].append(ms[index[:, :audit.PAST]])
        pack['future_index'].append(index[:, audit.PAST:])
        pack['start'].append(s[begin])
        pack['drone'].append(np.full(len(begin), r['drone']))
        pack['fragment'].append(np.full(len(begin), r['fragment']))
    if not parts:
        # Preserve the full audited file roster, including sessions with no accepted clock.
        return dict(history=np.empty((0, 8, 2)), offsets=np.array([0]), agent_id=np.empty(0, dtype=int),
                    start=np.empty(0, dtype=int), drone=np.empty(0, dtype=int), fragment=np.empty(0, dtype=int),
                    past_frame=np.empty((0, 8), dtype=int), past_ms=np.empty((0, 8), dtype=int)), dict(
                    future_index=np.empty((0, 12), dtype=int), origin=np.empty((0, 2)), future_mask=np.empty((0, 12), dtype=bool))
    data = {k: np.concatenate(v) for k, v in pack.items()}
    history = past_coordinates(archive, name, table, data['past_index'])
    offsets = np.r_[0, data['counts'].cumsum()]
    origin = history[offsets[:-1], -1]
    recent = history - np.repeat(origin, data['counts'], axis=0)[:, None]
    cv, ols = analytic(recent)
    if not np.isfinite(cv).all() or not np.isfinite(ols).all():
        raise ValueError('Reference rollout smoke check failed')
    return dict(history=recent, offsets=offsets, agent_id=data['agent'], start=data['start'], drone=data['drone'],
                fragment=data['fragment'], past_frame=data['past_frame'], past_ms=data['past_ms']), dict(
                future_index=data['future_index'], origin=origin, future_mask=data['future_index'] >= 0)


def save_npz(path, arrays):
    tmp = path.with_suffix('.pending')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)
    return sha(path)


def linkage(prefix, index, site, seen, boxes):
    h = prefix['history']
    if not len(h):
        return
    absolute = h + np.repeat(index['origin'], np.diff(prefix['offsets']), axis=0)[:, None]
    points = absolute.reshape(-1, 2)
    extent = np.array([points.min(axis=0), points.max(axis=0)])
    if site in boxes:
        extent = np.array([np.minimum(extent[0], boxes[site][0]), np.maximum(extent[1], boxes[site][1])])
    boxes[site] = extent
    packed = np.rint(absolute*100.).astype('<i8')
    for p, times in zip(packed, prefix['past_ms'], strict=True):
        stamp = hashlib.sha256(times.astype('<i8').tobytes() + p.tobytes()).digest()
        seen[stamp].add(site)


def run(source_dir, receipt, out, cache):
    source = json.loads(receipt.read_text())
    if source['new_split_assigned'] or source['forecast_outcomes_scored']:
        raise ValueError('Require the structural-only Songdo source audit')
    binding = dict(input_receipt=sha(receipt), implementation=sha(Path(__file__)), clock=sha(Path(audit.__file__)))
    results, seen, boxes = {}, defaultdict(set), {}
    for site in audit.PUBLISHER:
        file = f'{audit.DATE}_{site}.zip'
        archive_path = source_dir / file
        if sha(archive_path) != source['archives'][file]['sha256']:
            raise ValueError(f'Archive hash differs: {file}')
        with audit.zipfile.ZipFile(archive_path) as archive:
            for record in (r for r in source['sessions'] if r['site'] == site):
                name = record['session']
                stem = Path(name).stem
                p = cache / f'{stem}_prefix.npz'
                i = cache / f'{stem}_label_index.npz'
                s = cache / f'{stem}.json'
                if not s.exists():
                    a, b = arrays(archive, record)
                    info = dict(binding=binding, prefix_sha=save_npz(p, a), label_index_sha=save_npz(i, b))
                    write_json(s, info)
                info = json.loads(s.read_text())
                if info != dict(binding=binding, prefix_sha=sha(p), label_index_sha=sha(i)):
                    raise ValueError(f'Cached source does not match: {name}')
                with np.load(p, allow_pickle=False) as a, np.load(i, allow_pickle=False) as b:
                    linkage(a, b, site, seen, boxes)
                    results[name] = dict(site=site, queries=len(a['offsets'])-1, agents=len(a['history']),
                        complete_agent_futures=int(b['future_mask'].all(axis=1).sum()), **info)
                tmp = cache / 'heartbeat.pending'
                tmp.write_text(json.dumps(dict(pid=os.getpid(), sessions=len(results), last=name)) + '\n')
                tmp.replace(cache / 'heartbeat.json')
                print(name, results[name]['queries'], flush=True)
    counts = {s: audit.sum_fields({k: v[k] for k in ('queries', 'agents', 'complete_agent_futures')}
                                  for v in results.values() if v['site'] == s) for s in audit.PUBLISHER}
    total = sum(v['queries'] for v in results.values())
    if total != source['totals']['full_visibility']['nonoverlapping_queries']:
        raise ValueError('Pack query count differs from intake')
    overlap = []
    for n, (s, a) in enumerate(boxes.items()):
        for t, b in list(boxes.items())[n+1:]:
            if np.all(np.minimum(a[1], b[1]) >= np.maximum(a[0], b[0])):
                overlap.append([s, t])
    report = dict(result_source='fresh_run', source='cached_verified', binding=binding,
        sites=counts, sessions=results, total_queries=total,
        exact_timestamp_and_prefix_coordinate_hashes_crossing_sites=sum(len(x) > 1 for x in seen.values()),
        past_spatial_envelope_overlap=overlap, future_coordinate_fields_parsed=False,
        label_masks_separate=True, split_assigned=False, forecasts_scored=False,
        original_2pct_unchanged=True, independent_confirmation=False, reserved_outcomes_opened=False)
    write_json(out, report)
    print(json.dumps({k: report[k] for k in ('sites', 'total_queries', 'past_spatial_envelope_overlap',
                     'exact_timestamp_and_prefix_coordinate_hashes_crossing_sites')}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, default=audit.LOCAL)
    ap.add_argument('--receipt', type=Path, default=audit.PUBLIC / 'analysis.json')
    ap.add_argument('--cache', type=Path, default=CACHE)
    ap.add_argument('--output', type=Path, default=PUBLIC / 'receipt.json')
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    with (a.cache / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a.directory, a.receipt, a.output, a.cache)


if __name__ == '__main__':
    main()
