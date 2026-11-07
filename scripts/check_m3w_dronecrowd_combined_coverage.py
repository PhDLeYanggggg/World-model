"""Join two verified image screens to the fixed three-seed training roster."""
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_dronecrowd_cross_archive import HOME, OLD, digest, once, read

BROAD = HOME/'movingdronecrowd_background_v2'
NARROW = HOME/'dronecrowd_cross_archive_v1'
BANK = HOME/'dronecrowd_three_seed_bank_v1/analysis.json'
PUBLIC = HOME/'dronecrowd_combined_forecaster_coverage_v1'


def covered_trials(trials, narrow, broad):
    result = []
    for t in trials:
        roster = set(t['lineage']['recordings'])
        if roster-narrow != set(t['lineage']['unscreened']):
            raise ValueError('Training roster or earlier narrow screen changed')
        result.append(dict(key=t['key'], training_recordings=len(roster),
            added_coverage=sorted((roster-narrow)&broad), still_uncovered=sorted(roster-narrow-broad)))
    return result


def main():
    broad, narrow = (read(p/'analysis.json') for p in (BROAD, NARROW))
    br, nr = (read(p/'registration.json') for p in (BROAD, NARROW))
    ver, replay = (read(BROAD/p) for p in ('verification.json', 'read_only_replay_receipt.json'))
    narrow_proof = read(NARROW/'execution_verification.json')
    fresh = read(PUBLIC/'broad_readonly_recheck.json')
    if (any(v['analysis_sha256'] != digest(BROAD/'analysis.json') for v in (ver, replay, fresh))
            or not ver['all_pair_receipts_and_graph_exact'] or fresh['saved_clip_pairs_and_graph_exact'] != 20580
            or narrow_proof['analysis_sha256'] != digest(NARROW/'analysis.json')
            or not narrow_proof['all_48_receipts_and_aggregate_exact']
            or narrow['registration_sha256'] != digest(NARROW/'registration.json')
            or broad['registration_sha256'] != digest(BROAD/'registration.json')
            or broad['cross_source_pairs'] != 13440 or broad['moving_nodes_linked_to_protected']
            or broad['status_counts']['cross_source'] != dict(unsupported=13440, ambiguous=0, strong_multiview=0)
            or narrow['supported_pairs'] or broad['insufficient_background_moving_nodes']):
        raise ValueError('Both complete, exact image screens must support the recorded coverage')
    grouping = OLD/'exclusion_constraints.json'; group = read(grouping)['recording_group']
    if group != {r['node']:r['exclusion_group'] for r in nr['external']} or len(group) != 112 or len(set(group.values())) != 45:
        raise ValueError('Require the same 112 external recordings and 45 groups')
    parent = read(HOME/'movingdronecrowd_background_v1/registration.json')
    if parent['protected_constraints_sha256'] != digest(grouping):
        raise ValueError('The broad audit must have used exactly the same external grouping')
    roster = {r['node'] for r in br['view_selection']}; views = defaultdict(set)
    for v in read(BROAD/'image_manifest.json')['views']:
        views[v['node']].add(v['slot'])
    if len(roster) != 120 or set(views) != roster or any(s != {1,2,3} for s in views.values()):
        raise ValueError('Require all three archived source views for each of 120 recordings')
    bank = read(BANK); n = {r['node'] for r in nr['exposed']}
    result = covered_trials(bank['trials'], n, roster)
    added = sorted({s for r in result for s in r['added_coverage']})
    if len(result) != 18 or len(added) != 10 or any(r['still_uncovered'] for r in result):
        raise ValueError('The full original three-seed predictor roster must be covered')
    hashes = hashlib.sha256()
    for s in added:
        for d in sorted(group):
            path = ROOT/'data/stage_cvpr2027_experiments/movingdronecrowd_background_v2/pairs'/f'{s}___drone_{d}.json'
            receipt = read(path)
            if (receipt['registration_sha256'] != broad['registration_sha256'] or
                    (receipt['left'],receipt['right'],receipt['kind'],receipt['status']) !=
                    (s,'drone_'+d,'cross_source','unsupported') or receipt['support']):
                raise ValueError('Unexpected pair identity or match on the uncovered roster')
            hashes.update(bytes.fromhex(digest(path)))
    inputs = (BANK, grouping, BROAD/'analysis.json', BROAD/'registration.json', BROAD/'image_manifest.json',
        BROAD/'verification.json', BROAD/'read_only_replay_receipt.json', NARROW/'analysis.json',
        NARROW/'registration.json', NARROW/'execution_verification.json', PUBLIC/'broad_readonly_recheck.json')
    value = dict(result_source='fresh_training_roster_join_cached_verified_image_screens',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs}, trials=result,
        covered_predictors=18, covered_only_by_narrow_screen=12, added_recordings=added,
        added_cached_pair_receipts_verified=len(added)*len(group), added_receipt_digest=hashes.hexdigest(),
        external_groups=45, external_recordings=112, physical_independence_certified=False,
        cost_head_admission=False, scientific_roles_assigned=False, forecast_outcomes_opened=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
    once(PUBLIC/'analysis.json', value)
    print(dict(covered_predictors=18, added_recordings=len(added), cached_pairs=len(added)*len(group),
        analysis_sha256=digest(PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
