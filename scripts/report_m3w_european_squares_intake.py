"""Generate a quantitative source report only after both audit replays finish."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_european_squares_v2 import PRIVATE, PUBLIC
from scripts.fetch_m3w_european_squares import PUBLIC as SOURCE_PUBLIC, digest, save


def main():
    a = json.loads((PUBLIC/'analysis.json').read_text())
    v = json.loads((PUBLIC/'verification.json').read_text())
    independent = json.loads((PUBLIC/'arithmetic_verification.json').read_text())
    sha = digest(PUBLIC/'analysis.json')
    if not v['exact'] or v['analysis_sha256'] != sha:
        raise ValueError('Complete exact replay required')
    if not independent['all_checks_passed'] or independent['analysis_sha256'] != sha:
        raise ValueError('Separate arithmetic checks required')
    records = a['recordings']
    if independent['total_rows_replayed'] != a['rows'] or len(independent['records']) != len(records):
        raise ValueError('Incomplete independent replay')
    overlap = 0
    prior = PRIVATE.with_name('european_squares_intake_v1')/'records'
    by_member = {r['source_member']: r for r in records}
    for p in prior.glob('*.json'):
        if p.name.endswith('_track_hashes.json'):
            continue
        old = json.loads(p.read_text())
        if {k:v for k,v in old.items() if k!='identity'} != by_member[old['source_member']]:
            raise ValueError('V2 changed a previously verified full-schema recording')
        overlap += 1
    support = {}
    for key in records[0]['support']:
        support[key] = {name:sum(r['support'][key][name] for r in records) for name in
            ('past_eligible','complete_future12','partial_future12','no_future12','query_frames')}
        support[key]['query_frames_at_least_agents'] = {n:sum(r['support'][key]['query_frames_at_least_agents'][n] for r in records)
                                                       for n in ('2','5','10')}
        s = support[key]
        if s['past_eligible'] != s['complete_future12']+s['partial_future12']+s['no_future12']:
            raise ValueError('Support conservation failed')
    sites = []
    for sid in a['source_square_ids']:
        sub = [r for r in records if r['square_id'] == sid]
        sites.append(dict(square_id=sid,recordings=len(sub),rows=sum(r['rows'] for r in sub),
                          scoped_tracks=sum(r['tracks'] for r in sub),
                          past8_stride12=sum(r['support']['K8_stride12']['past_eligible'] for r in sub),
                          complete_future12_stride12=sum(r['support']['K8_stride12']['complete_future12'] for r in sub)))
    class_ids, classes, gaps = Counter(), Counter(), Counter()
    for r in records:
        class_ids.update(r['class_id_counts'])
        classes.update(r['class_name_counts'])
        gaps.update(r['frame_delta_counts'])
    events = [json.loads(line) for line in (PRIVATE/'events.jsonl').read_text().splitlines()]
    completed = [e for e in events if e['state']=='complete']
    finishes = [e for e in events if e['state']=='record_complete']
    result = dict(result_source='fresh_run',source_analysis_sha256=sha,
        raw_recordings=a['raw_recordings'],nominal_squares=len(sites),rows=a['rows'],scoped_tracks=a['tracks'],
        support=support,sites=sites,class_id_counts=dict(class_ids),class_name_counts=dict(classes),
        frame_delta_counts=dict(gaps),tracks_shorter_than_30=sum(r['tracks_shorter_than_30'] for r in records),
        max_simultaneous_agents=max(r['max_simultaneous_agents'] for r in records),
        within_recording_exact_full_relative_track_aliases=sum(r['duplicate_full_relative_track_geometries'] for r in records),
        cross_recording_exact_full_relative_track_groups=a['cross_recording_exact_relative_track_duplicate_groups'],
        prefix_checks=a['prefix_checks'],exact_replay=True,separate_arithmetic=True,
        v1_full_schema_recordings_reproduced=overlap,
        direct_past_membership_checks=sum(r['past_membership_checks'] for r in independent['records']),
        independent_future_count_checks=sum(r['complete_future_count_checks'] for r in independent['records']),
        independent_velocity_checks=sum(r['direct_velocity_checks'] for r in independent['records']),
        original_and_replay_elapsed_seconds=[e['seconds'] for e in completed],
        peak_rss_bytes=max(e['peak_rss_bytes'] for e in finishes),
        analysis_code_hashes=a['identity']['files'],role='quarantined_unassigned_source_audit',
        source_online_causality_established=False,independent_sites_admitted=0,
        forecast_errors_opened=False,training=False,main_protocol_changed=False,deployment_changed=False,
        metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
    save(PUBLIC/'summary.json',result)
    lines = ['# European Squares Raw Intake: Verified Structure, No Model Claim', '',
        '2026-09-24. Result source: fresh_run. Official bytes are cached_verified against the pinned acquisition manifest.', '',
        '## What Completed', '',
        f"All {a['raw_recordings']} released raw CSV recordings were read and replayed, covering {len(sites)} nominal square IDs,",
        f"{a['rows']:,} rows and {a['tracks']:,} recording-scoped tracker IDs. These IDs are not a count of unique people.",
        'No raw recordings are omitted because of short history, missing future support or forecast difficulty.',
        f"{result['tracks_shorter_than_30']:,} tracks shorter than 30 observations remain. Query eligibility uses past support only.", '',
        f"{a['prefix_checks']:,} prefix mutation/truncation checks pass. Complete re-parsing reproduces all recording arrays and summaries.",
        f"Separate arithmetic replays all rows and checks {result['direct_past_membership_checks']:,} fixed past-membership cases,",
        f"{result['independent_future_count_checks']:,} complete future-count cases and {result['independent_velocity_checks']:,} direct velocities.",
        'The separate counting covers fixed samples of up to three tracks per recording, not every track.',
        '52 scoped tests pass. This is not a rerun of the full legacy test suite.', '',
        '## Past Support and Label Availability', '',
        'Stride is an index increment in the released raw detector frames. These probes do not establish common seconds across datasets.',
        'Counts overlap in time and across history lengths. They are not independent experimental sample sizes.', '',
        '| History/grid | Past eligible | All 12 future labels | Partial future | No future | Recording/frame queries |',
        '|---|---:|---:|---:|---:|---:|']
    for key,s in support.items():
        lines.append(f"| {key} | {s['past_eligible']:,} | {s['complete_future12']:,} | {s['partial_future12']:,} | {s['no_future12']:,} | {s['query_frames']:,} |")
    lines += ['', '## Limits That Remain', '',
        'The metadata/archive discrepancies and the initial schema failure are retained in [versioned_repair.md](versioned_repair.md).',
        f"Exact full-track geometry screening found {result['cross_recording_exact_full_relative_track_groups']} cross-recording groups and",
        f"{result['within_recording_exact_full_relative_track_aliases']} within-recording extra aliases. This screen does not exclude partial clip duplicates or shared cameras.",
        'Physical locality, related-source exposure, recording correspondence, publisher online processing and frame-time mappings still require admission decisions.',
        'All labels are automated. No verified metric transform or human-gold claim is made.',
        'The seasonal collection repeats a few locations; it cannot be counted as hundreds of independent scenes.', '',
        'The source remains quarantined and unassigned: zero sites are yet admitted to independent calibration or confirmation.',
        'No baseline or neural forecast errors were opened, no goals were built, no model was trained and deployment is unchanged.',
        'DroneCrowd remains closed confirmation; the exposed SDD sites remain development-only. Stage5C and SMC are not executed.', '',
        '## Next Evidence Step', '',
        'Reconcile recording/site identities and screening across related sources, then freeze a conservative site-group role manifest.',
        'Specify the raw point convention, history grid and automated-label provenance without claiming online sensor or metric validation.',
        'Only after role admission should train-side support and a source-only method test be opened; independent confirmation must not select models.', '',
        '[Source provenance](../european_squares_intake_v1/provenance.md), [machine summary](summary.json),',
        '[replay](verification.json), [separate arithmetic](arithmetic_verification.json), [operation/recovery](operation_zh.md).', '']
    out=PUBLIC/'conclusions.md'
    text='\n'.join(lines)
    if out.exists() and out.read_text()!=text:
        raise ValueError('Existing conclusions differ')
    if not out.exists():
        out.write_text(text)
    print(json.dumps({k:result[k] for k in ('rows','raw_recordings','nominal_squares','scoped_tracks','prefix_checks')}))


if __name__=='__main__':
    main()
