"""Check possible cross-site linkage using only the prepared observed prefixes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_prefix_pack as pack

np = pack.np
RADIUS = 2.
ANCHORS = 4


def close_during_overlap(a, t, b, u):
    """Use A's observed times within B's observed span; do not extrapolate."""
    inside = (t >= u[0]) & (t <= u[-1])
    if inside.sum() < ANCHORS:
        return False
    interp = np.column_stack([np.interp(t[inside], u, b[:, i]) for i in (0, 1)])
    return bool(np.linalg.norm(a[inside] - interp, axis=1).max() <= RADIUS)


def compare(a, t, b, u):
    order = np.argsort(u[:, 0], kind='stable')
    b, u = b[order], u[order]
    amin, amax, bmin, bmax = a.min(axis=1), a.max(axis=1), b.min(axis=1), b.max(axis=1)
    max_span = (u[:, -1] - u[:, 0]).max(initial=0)
    count = Counter()
    match_a, match_b = set(), set()
    for j, row in enumerate(t):
        lo = np.searchsorted(u[:, 0], row[0] - max_span)
        hi = np.searchsorted(u[:, 0], row[-1], side='right')
        if hi == lo:
            continue
        candidate = (u[lo:hi, -1] >= row[0]) & np.all(
            (amin[j] <= bmax[lo:hi] + RADIUS) & (amax[j] >= bmin[lo:hi] - RADIUS), axis=1)
        for k in lo + np.flatnonzero(candidate):
            count['envelope_time_candidates'] += 1
            if close_during_overlap(a[j], row, b[k], u[k]):
                count['possible_duplicate_prefix_pairs'] += 1
                match_a.add(j)
                match_b.add(k)
    return dict(prefixes_a=len(a), prefixes_b=len(b), envelope_time_candidates=count['envelope_time_candidates'],
                possible_duplicate_prefix_pairs=count['possible_duplicate_prefix_pairs'],
                matched_prefixes_a=len(match_a), matched_prefixes_b=len(match_b))


def load(receipt, cache, site):
    xy, t = [], []
    for name, info in receipt['sessions'].items():
        if info['site'] != site or not info['queries']:
            continue
        stem = Path(name).stem
        prefix = cache / f'{stem}_prefix.npz'
        index = cache / f'{stem}_label_index.npz'
        if (pack.sha(prefix), pack.sha(index)) != (info['prefix_sha'], info['label_index_sha']):
            raise ValueError('Cached inputs do not match prefix receipt')
        with np.load(prefix, allow_pickle=False) as a, np.load(index, allow_pickle=False) as b:
            # The origin is derived from the first observed agent, not from future labels.
            h = a['history'] + np.repeat(b['origin'], np.diff(a['offsets']), axis=0)[:, None]
            xy.append(h)
            t.append(a['past_ms'])
    return np.concatenate(xy), np.concatenate(t)


def run(receipt_path, cache, output):
    receipt = json.loads(receipt_path.read_text())
    if receipt['split_assigned'] or receipt['forecasts_scored'] or receipt['future_coordinate_fields_parsed']:
        raise ValueError('Require the split-neutral prefix-only receipt')
    groups = [{s} for s in receipt['sites']]
    matches = {}
    for a, b in receipt['past_spatial_envelope_overlap']:
        matches[f'{a}:{b}'] = compare(*load(receipt, cache, a), *load(receipt, cache, b))
        merge = [g for g in groups if a in g or b in g]
        groups = [g for g in groups if g not in merge] + [set.union(*merge)]
    result = dict(prefix_receipt_sha256=pack.sha(receipt_path), threshold_projected_units=RADIUS,
        minimum_a_anchors=ANCHORS, matches=matches, conservative_source_components=sorted(sorted(g) for g in groups),
        component_rule='union sites whose sampled prefix bounding boxes intersect, irrespective of duplicate matches',
        repeated_vehicle_across_times_ruled_out=False, future_xy_loaded=False,
        split_assigned=False, forecasts_scored=False, independent_confirmation=False)
    pack.write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt', type=Path, default=pack.PUBLIC / 'receipt.json')
    ap.add_argument('--cache', type=Path, default=pack.CACHE)
    ap.add_argument('--output', type=Path, default=pack.PUBLIC / 'linkage.json')
    a = ap.parse_args()
    run(a.receipt, a.cache, a.output)
