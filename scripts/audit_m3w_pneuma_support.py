"""Audit 8/12 past-selected rosters and source linkage on a bounded pilot."""

from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import intake_m3w_pneuma as source
from scripts.run_m3w_opendd_location import sha, write_json
from src.data_unification import m3w_pneuma_support as model


def main():
    receipt = source.PUBLIC/'samples.json'
    samples = json.loads(receipt.read_text())['members']
    tracks, rows = {}, {}
    for name, f in samples.items():
        path = source.LOCAL/name
        if not f['zip_crc_verified'] or sha(path) != f['sha256']:
            raise ValueError('Pilot does not match the verified publisher member')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            tracks[name] = t = model.scan(stream)
        coords = tuple(zip(*t.first_points, strict=True))
        bbox = [[round(min(z), 5), round(max(z), 5)] for z in coords]
        rows[name] = dict(date=f['day'], drone=f['drone'], period=f['period'],
            records=t.diagnostics, vehicle_types=t.types, first_observed_bbox_lat_lon=bbox,
            top_interval_microseconds=t.intervals.most_common(8),
            coverage={str(s): model.coverage(t, s) for s in (1, 5, 10)})
        print(name, 'tracks:', len(t.ids), 'queries stride 10:', rows[name]['coverage']['10']['past_selected_queries'],
              flush=True)
    overlaps = []
    for a, b in combinations(tracks, 2):
        ta, tb = tracks[a], tracks[b]
        ba, bb = (rows[x]['first_observed_bbox_lat_lon'] for x in (a, b))
        overlaps.append(dict(a=a, b=b, same_day=rows[a]['date']==rows[b]['date'],
            same_drone=rows[a]['drone']==rows[b]['drone'], shared_numeric_ids=len(ta.ids & tb.ids),
            exact_prefix_matches=len(ta.prefixes.keys() & tb.prefixes.keys()),
            first_observed_bbox_intersection=all(max(x[0], y[0])<=min(x[1], y[1]) for x, y in zip(ba, bb, strict=True))))
    out = dict(result_source='fresh_structural_intake', samples_sha256=sha(receipt), scan_code_sha256=sha(Path(model.__file__)),
        data='pNEUMA / Athens / October-November 2018', source_coordinates='publisher WGS84 latitude/longitude; no projection tested',
        nominal_tick_seconds=model.TICK, sampled_strides=[1, 5, 10], rows=rows, linkage=overlaps,
        clock_grid_policy='exact within 1 microsecond; count and exclude off-grid timestamps; no interpolation',
        selection='earliest nonoverlapping 8-step past roster with >=2 agents; 12 future token-presence counted only after selection',
        coordinate_decode='first 8 observations per track for structural linkage only; no subsequent coordinate decoding',
        support_is_timestamp_and_token_presence_only=True, relative_error_evaluated=False, predictive_roles_assigned=False,
        forecasts_scored=False, source_independence_established=False, external_confirmation_opened=False)
    write_json(source.PUBLIC/'support.json', out, immutable=True)


if __name__ == '__main__':
    main()
