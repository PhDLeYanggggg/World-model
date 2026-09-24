"""Emit deterministic recording-group keys before any forecasting readout."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'data/stage_cvpr2027_experiments/european_squares_intake_v1/reader_dependencies'))
from pyxlsb import open_workbook
from scripts.fetch_m3w_european_squares import RAW, PUBLIC as SOURCE_PUBLIC, digest, save
from scripts.audit_m3w_european_squares_v2 import PUBLIC as AUDIT_PUBLIC
from src.evaluation.m3w_european_squares_grouping import locality_groups, camera_identity

OUT=ROOT/'outputs/publication_readiness_2026_09/european_squares_site_groups_v1'


def main():
    manifest=json.loads((SOURCE_PUBLIC/'metadata_manifest.json').read_text())
    for r in manifest['private_files']:
        if digest(ROOT/r['path'])!=r['sha256']:
            raise ValueError('Source metadata identity differs')
    audit=json.loads((AUDIT_PUBLIC/'analysis.json').read_text())
    verified=json.loads((AUDIT_PUBLIC/'verification.json').read_text())
    if not verified['exact'] or verified['analysis_sha256']!=digest(AUDIT_PUBLIC/'analysis.json'):
        raise ValueError('Raw audit replay missing')
    meta=json.loads((SOURCE_PUBLIC/'metadata_audit.json').read_text())
    positions={r['square_id']:r for r in meta['parsed_stats_sites'] if r['dataset']=='comparative'}
    sites=[]
    with open_workbook(RAW/'Place_Overview.xlsb') as book:
        with book.get_sheet(1) as sheet:
            for i,row in enumerate(sheet.rows()):
                values=[r.v for r in row]
                if i<2 or values[0] is None:
                    continue
                sid=int(values[0])
                p=positions[sid]
                sites.append(dict(square_id=sid,place_name=values[1],city=values[2],country=values[3],
                    camera_url=values[4] or '',latitude=p['latitude'],longitude=p['longitude'],
                    location_source='comparative_statistics_not_season_conflicted_values'))
    if set(r['square_id'] for r in sites)!=set(audit['source_square_ids']):
        raise ValueError('Unmapped raw square')
    grouped=locality_groups(sites)
    site_to_group={sid:'eu-locality-'+str(min(g)).zfill(3) for g in grouped['groups'] for sid in g}
    rows=[]
    for r in audit['recordings']:
        rows.append(dict(source_member=r['source_member'],source_square_id=r['square_id'],
                         locality_group=site_to_group[r['square_id']],rows_sha256=r['rows_sha256'],
                         data_role='unassigned_source_audit_no_forecast_readout'))
    identities={str(r['square_id']):camera_identity(r['camera_url']) for r in sites}
    result=dict(result_source='fresh_run',analysis_sha256=digest(AUDIT_PUBLIC/'analysis.json'),
        metadata_audit_sha256=digest(SOURCE_PUBLIC/'metadata_audit.json'),
        source_manifest_sha256=digest(SOURCE_PUBLIC/'metadata_manifest.json'),
        code_sha256=digest(Path(__file__)),module_sha256=digest(ROOT/'src/evaluation/m3w_european_squares_grouping.py'),
        grouping=grouped,nominal_sites=len(sites),locality_groups=len(grouped['groups']),
        sensitivity_group_counts={str(radius):len(locality_groups(sites,radius)['groups']) for radius in (1.,5.,10.)},
        camera_reference_identities=identities,recordings=rows,
        rules='same_overview_city_country_or_shared_camera_reference_or_transitive_5km_source_geographic_proximity',
        coordinate_note='geographic_site_metadata_only_not_trajectory_metric_calibration',
        date_conflicts_resolved=False,source_exposure_complete=False,partial_clip_duplicates_excluded=False,
        roles_assigned=False,independence_proven=False,training=False,forecast_errors_opened=False,
        stage5c_executed=False,smc_enabled=False)
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'site_groups.json',result)
    print(json.dumps(dict(nominal_sites=len(sites),locality_groups=len(grouped['groups']),edges=grouped['edges'],
                         radius_sensitivity=result['sensitivity_group_counts'])))


if __name__=='__main__':
    main()
