"""Count 8/12 support in the Swiss release without parsing numeric positions."""

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fetch_m3w_swiss_roundabouts as source
from src.data_unification import m3w_swiss_roundabout_support as model


def run():
    p = source.OUT/'source.json'
    binding = json.loads(p.read_text())
    members = binding['members']
    source.verify_roster([{k: f[k] for k in ('key', 'size', 'checksum')} for f in members])
    stats, totals = {}, defaultdict(lambda: defaultdict(Counter))
    for f in members:
        name = f['key']
        m = model.FILE.fullmatch(name)
        if not m:
            continue
        path = source.LOCAL/name
        if source.digest(path) != f['sha256'] or source.digest(path, 'md5') != f['checksum'][4:]:
            raise ValueError(f'Cached trajectory changed: {name}')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            segments, info = model.scan(stream)
        drone, period, session, site = m.groups()
        counts = {}
        for s in (1, 6, 12):
            counts[str(s)] = c = Counter()
            for segment in segments:
                c.update(model.coverage(list(segment['tracks'].values()), segment['ticks'], s))
        stats[name] = dict(info, drone=int(drone), period=period, session=int(session),
                           roundabout=site, coverage=counts)
        for s, r in counts.items():
            totals[site][s].update(r)
        print(name, info['tracks'], 'tracks;', counts['12']['queries'], 'nonoverlap stride-12 queries', flush=True)
    linked = []
    for (x, a), (y, b) in combinations(stats.items(), 2):
        overlap = min(a['last_clock_centisecond'], b['last_clock_centisecond'])-max(
            a['first_clock_centisecond'], b['first_clock_centisecond'])
        if overlap >= 0:
            linked.append(dict(a=x, b=y, overlap_centiseconds=overlap, same_drone=a['drone'] == b['drone'],
                               same_roundabout=a['roundabout'] == b['roundabout']))
    source.write(source.OUT/'support.json', dict(result_source='fresh_structural_audit',
        source_receipt_sha256=source.digest(p), scan_code_sha256=source.digest(Path(model.__file__)),
        camera_rate_documented='29.97 fps', encoding_grid='30 Hz timestamp consistency only',
        coordinate_unit='publisher WGS84 longitude/latitude; no projection or accuracy verification',
        clock_dates=['2023-10-23'], sites=list(sorted(totals)), conservative_municipality_groups=['Frick', 'Laufenburg'],
        members=stats, totals_by_site=totals, overlapping_clock_intervals=linked,
        selection='earliest nonoverlapping 8-observation grid with two targets, within each uninterrupted clock segment',
        numeric_coordinates_parsed=False, forecasts_scored=False, roles_assigned=False, no_forecast_label_opening=True))


if __name__ == '__main__':
    run()
