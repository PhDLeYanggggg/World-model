"""Audit official metadata without reading trajectory outcomes or visiting webcams."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data/stage_cvpr2027_experiments/european_squares_intake_v1/reader_dependencies'))
from pyxlsb import open_workbook
import pandas as pd
from scripts.fetch_m3w_european_squares import RAW, PUBLIC, digest, save
from scripts.audit_m3w_european_squares import metadata_sites


def main():
    manifest = json.loads((PUBLIC / 'metadata_manifest.json').read_text())
    for item in manifest['private_files']:
        p = ROOT / item['path']
        if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise ValueError('Acquisition identity differs')
    overview = []
    with open_workbook(RAW / 'Place_Overview.xlsb') as book:
        with book.get_sheet(1) as sheet:
            for i, row in enumerate(sheet.rows()):
                values = [c.v for c in row]
                if i < 2:
                    continue
                if values[0] is None:
                    continue
                overview.append(dict(square_id=int(values[0]), place_name=values[1],
                                     city=values[2], country=values[3]))
    ids = [r['square_id'] for r in overview]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated overview square ID')
    cities = defaultdict(list)
    for r in overview:
        cities[(r['country'].casefold(), r['city'].casefold())].append(r['square_id'])
    reports = {}
    for name in ('comparative', 'season'):
        frame = pd.read_csv(RAW / f'stats_{name}.csv')
        sites = []
        for square, g in frame.groupby('No.'):
            variants = g[['City', 'Country']].drop_duplicates().to_dict('records')
            if len(variants) > 1:
                sites.append(dict(square_id=int(square), variants=variants,
                    records=g[['Date', 'timeslot', 'City', 'Country', 'lat', 'long']].to_dict('records')))
        reports[name] = dict(recordings=len(frame), square_ids=sorted(map(int, frame['No.'].unique())),
                             identity_conflicts=sites)
    with zipfile.ZipFile(RAW / '02.Resources.zip') as z:
        resources = [dict(name=i.filename, bytes=i.file_size, crc=i.CRC) for i in z.infolist()
                     if not i.is_dir() and not i.filename.startswith('__MACOSX/')]
    result = dict(result_source='fresh_run', metadata_manifest_sha256=digest(PUBLIC / 'metadata_manifest.json'),
                  script_sha256=digest(Path(__file__)), overview_squares=overview,
                  nominal_square_count=len(overview), metadata_recordings=reports,
                  same_city_groups_requiring_independence_review=[v for v in cities.values() if len(v)>1],
                  parsed_stats_sites=metadata_sites(), resource_inventory=resources,
                  parser='pyxlsb==1.0.10 isolated reader target',
                  role='quarantined_unassigned_source_audit',
                  coordinates_claim='raw_box_pixel_only_no_verified_world_transform',
                  forecast_errors_opened=False, dataset_roles_assigned=False,
                  automatic_conflict_repair=False, independent_sites_admitted=0)
    save(PUBLIC / 'metadata_audit.json', result)
    print(json.dumps(dict(overview_squares=len(overview),
                         metadata_recordings=sum(r['recordings'] for r in reports.values()),
                         conflict_groups=sum(len(r['identity_conflicts']) for r in reports.values()))))


if __name__ == '__main__':
    main()
