"""Freeze locality-level roles before forecasting; open training records only."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'data/stage_cvpr2027_experiments/european_squares_intake_v1/reader_dependencies'))
from pyxlsb import open_workbook
from scripts.fetch_m3w_european_squares import RAW, PUBLIC as SOURCE_PUBLIC, digest, save
from src.evaluation.m3w_european_squares_grouping import locality_groups
from src.evaluation.m3w_european_squares_roles import assign_roles, require_source_training, SALT

PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
OUT = PUBLIC/'european_squares_roles_v1'


def main():
    paths = {
        'raw_analysis': PUBLIC/'european_squares_intake_v2/analysis.json',
        'raw_replay': PUBLIC/'european_squares_intake_v2/verification.json',
        'locality': PUBLIC/'european_squares_site_groups_v1/site_groups.json',
        'overlap': PUBLIC/'european_squares_overlap_v1/analysis.json',
        'overlap_replay': PUBLIC/'european_squares_overlap_v1/verification.json',
        'exposure': PUBLIC/'european_squares_exposure_v1/analysis.json',
        'source_manifest': SOURCE_PUBLIC/'metadata_manifest.json',
        'metadata_audit': SOURCE_PUBLIC/'metadata_audit.json',
        'authorization': PUBLIC/'delegated_research_authorization_20260923.json',
        'registration': OUT/'registration.md',
        'code': Path(__file__),
        'module': ROOT/'src/evaluation/m3w_european_squares_roles.py',
        'tests': ROOT/'tests/test_m3w_european_squares_roles.py',
    }
    binding = {k: dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for k, p in paths.items()}
    raw = json.loads(paths['raw_analysis'].read_text())
    replay = json.loads(paths['raw_replay'].read_text())
    groups = json.loads(paths['locality'].read_text())
    overlap = json.loads(paths['overlap'].read_text())
    overlap_replay = json.loads(paths['overlap_replay'].read_text())
    exposure = json.loads(paths['exposure'].read_text())
    if not replay['exact'] or replay['analysis_sha256'] != digest(paths['raw_analysis']):
        raise ValueError('Raw replay identity missing')
    if not overlap_replay['exact'] or overlap_replay['analysis_sha256'] != digest(paths['overlap']):
        raise ValueError('Overlap replay identity missing')
    if overlap['identity']['analysis_sha256'] != replay['analysis_sha256'] or overlap['raw_recordings'] != len(raw['recordings']):
        raise ValueError('Incomplete or mismatched overlap audit')
    if any(m['cross_locality_pairs'] for m in overlap['modes'].values()):
        raise ValueError('Cross-locality candidate requires explicit review before roles')
    if exposure['git_text_matches'] or any(r['matches'] for r in exposure['private_identity_receipts']):
        raise ValueError('Prior exposure candidate requires review')
    metadata = json.loads(paths['source_manifest'].read_text())
    for r in metadata['private_files']:
        if digest(ROOT/r['path']) != r['sha256']:
            raise ValueError('Source metadata differs')
    meta = json.loads(paths['metadata_audit'].read_text())
    positions = {r['square_id']: r for r in meta['parsed_stats_sites'] if r['dataset'] == 'comparative'}
    sites = []
    with open_workbook(RAW/'Place_Overview.xlsb') as book:
        with book.get_sheet(1) as sheet:
            for i, row in enumerate(sheet.rows()):
                values = [r.v for r in row]
                if i < 2 or values[0] is None:
                    continue
                sid = int(values[0])
                p = positions[sid]
                sites.append(dict(square_id=sid, city=values[2], country=values[3],
                    camera_url=values[4] or '', latitude=p['latitude'], longitude=p['longitude']))
    if locality_groups(sites) != groups['grouping']:
        raise ValueError('Original grouping does not replay')
    grouped = locality_groups(sites, radius_km=10.0)
    mapping = {sid: 'eu-locality-'+str(min(g)).zfill(3) for g in grouped['groups'] for sid in g}
    if set(mapping) != set(raw['source_square_ids']):
        raise ValueError('Unmapped source site')
    support = defaultdict(int)
    for r in raw['recordings']:
        support[mapping[r['square_id']]] += r['support']['K8_stride12']['past_eligible']
    assignments = assign_roles(support)
    records = []
    for r in raw['recordings']:
        group = mapping[r['square_id']]
        role = assignments[group]['role']
        records.append(dict(source_member=r['source_member'], rows_sha256=r['rows_sha256'],
            source_square_id=r['square_id'], locality_group=group, role=role,
            training_access=role == 'source_training', predictive_readout_access=False,
            raw_rows=r['rows'], scoped_tracks=r['tracks'],
            past_eligible_k8_stride12=r['support']['K8_stride12']['past_eligible']))
    by_role = {}
    for role in sorted({r['role'] for r in records}):
        subset = [r for r in records if r['role'] == role]
        by_role[role] = dict(locality_groups=len({r['locality_group'] for r in subset}),
            recordings=len(subset), raw_rows=sum(r['raw_rows'] for r in subset),
            scoped_tracks=sum(r['scoped_tracks'] for r in subset),
            past_eligible_k8_stride12=sum(r['past_eligible_k8_stride12'] for r in subset))
    result = dict(result_source='fresh_run', status='restricted_source_training_admitted_reserved_roles_closed',
        dependency_bindings=binding, assignments=assignments, grouping=grouped,
        records_key='recordings', recordings=records, counts=by_role,
        assignment_salt=SALT, assignment_inputs='locality keys and past-support counts only',
        observation_steps=8, prediction_steps=12, raw_frame_stride=12,
        point_convention='unsmoothed_raw_bounding_box_center',
        coordinate_unit='image_pixel', time_unit='released_raw_frame_index',
        label_provenance='automated_detector_tracker_not_human_gold',
        primary_sdd_protocol_changed=False, universal_prior_nonexposure_proven=False,
        online_sensor_causality_proven=False, independence_proven=False,
        reserved_predictive_outcomes_opened=False, source_forecast_errors_opened=False,
        source_training_executed=False, calibrated_safety_guarantee=False,
        metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
    for r in records:
        if r['training_access']:
            require_source_training(result, r['source_member'])
        else:
            try:
                require_source_training(result, r['source_member'])
            except PermissionError:
                continue
            raise AssertionError('Reserved training access allowed')
    if any(len({r['role'] for r in records if r['locality_group'] == group}) != 1 for group in assignments):
        raise ValueError('Locality leaks between roles')
    save(OUT/'roles.json', result)
    print(json.dumps(dict(roles_sha256=digest(OUT/'roles.json'), counts=by_role, added_edges=grouped['edges'])))


if __name__ == '__main__':
    main()
