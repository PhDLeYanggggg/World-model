"""Verify the main evidence roles and frozen artifacts without opening held outcomes."""

import argparse
from hashlib import file_digest
from itertools import combinations, product
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT/'outputs/publication_readiness_2026_09'
DATA = ROOT/'data/stage_cvpr2027_experiments'
OUT = PUB/'paper_evidence_lineage_v1/receipt.json'
SEEDS = (17, 29, 43)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()


def checked(ref):
    path = (ROOT/ref['path']).resolve()
    need(path.is_relative_to(ROOT) and path.is_file() and path.stat().st_size == ref.get('bytes', path.stat().st_size)
         and sha(path) == ref['sha256'], f'Frozen artifact mismatch: {path.name}')
    return path


def cohort_layout(recording, start, offsets, rows, stride):
    need(len(recording) == len(start) and len(offsets) == len(start)+1
         and stride > 0 and offsets[0] == 0 and offsets[-1] == rows
         and (offsets[1:] > offsets[:-1]).all(), 'Invalid query-to-row indexing')
    for name in np.unique(recording):
        gap = np.diff(np.sort(start[recording == name]))
        need((gap > 19*stride).all(), 'Duplicated or overlapping 8+12 observation windows')


def fitting_scope(identity, universe, excluded, key='fit'):
    used = identity[key]
    need(len(used) == len(set(used)) and set(used) == set(universe)-set(excluded), 'Checkpoint used a wrong fitting source')


def opendd():
    home = PUB/'recording_count_v2'
    reg, frozen, score = (read(home/(k+'.json')) for k in ('registration', 'head_freeze', 'analysis'))
    reg_sha, frozen_sha = (sha(home/(k+'.json')) for k in ('registration', 'head_freeze'))
    need(reg['no_held_tuning'] and score['original_2pct'] and reg['horizon'] == [8, 12]
         and reg['seeds'] == list(SEEDS) and reg['threads'] == 4 and reg['workers'] == 0
         and not score['confirmation_access'] and frozen['registration'] == score['registration'] == reg_sha
         and score['heads'] == frozen_sha, 'OpenDD count-head lineage changed')
    for entry in reg['parent']:
        checked(entry)
    parent = PUB/'matched_bridge_swaps_v1/head_freeze.json'
    need(any(checked(ref) == parent for ref in reg['parent']), 'OpenDD outer ranker bank is not registered')
    outer = read(parent)
    sites = set(score['by_site'])
    need(len(sites) == 7, 'Expected seven held-out development locations')
    banks = (('rank_heads', {f'rank_{a}_{b}_{s}': ([a, b], s, None)
                             for (a, b), s in product(combinations(sorted(sites), 2), SEEDS)}),
             ('count_heads', {f'count_{a}_{s}_{m}': ([a], s, m)
                              for a, s, m in product(sites, SEEDS, ('local', 'recording'))}))
    for bank, expected in banks:
        need(set(frozen[bank]) == set(expected), 'Incomplete source-excluded OpenDD checkpoint roster')
        for name, (omit, seed, mode) in expected.items():
            value = frozen[bank][name]
            i = value['identity']
            fitting_scope(i, sites, omit)
            need(i['registration'] == reg_sha and i['seed'] == seed and (
                (i['held'] == omit[0] and i['mode'] == mode) if mode else i['omitted'] == omit),
                'OpenDD checkpoint role identity changed')
            checked(value['artifact'])

    past_only = 0
    pool = PUB/'streaming_menu_capacity_v1'
    cap, cap_reg = (read(pool/(k+'.json')) for k in ('analysis', 'registration'))
    need({checked(ref) for ref in cap_reg['parent']} == {home/(k+'.json') for k in (
        'analysis', 'head_freeze', 'matched_diagnostic')} and cap['registration'] == sha(pool/'registration.json')
        and cap['labels_hindsight_only'] and cap['original_2pct'] and not cap['confirmation_access']
        and set(cap['by_site']) == sites, 'OpenDD hindsight and past-only results are not bound to the same setup')
    source = PUB/'opendd_location_diagnostic_v1/registration.json'
    stride = read(source)['stride']
    source_sha = sha(source)
    all_ids, recordings = set(), set()
    n_queries = n_rows = 0
    for site, views in score['by_site'].items():
        path = DATA/f'opendd_location_diagnostic_v1/{site}_cohort.npz'
        manifest = read(path.with_suffix('.json'))
        need(manifest['sha256'] == sha(path) and manifest['registration'] == source_sha, 'OpenDD cohort differs from its receipt')
        with np.load(path, allow_pickle=False) as z:
            rec, start, offsets, ids = (z[k] for k in ('recording', 'start', 'offsets', 'all_ids'))
        cohort_layout(rec, start, offsets, manifest['rows'], stride)
        found, unique_ids = set(rec.tolist()), set(ids.tolist())
        need(len(ids) == len(unique_ids) and not all_ids & unique_ids and not recordings & found,
             'Publisher object or recording IDs overlap across OpenDD locations')
        all_ids.update(unique_ids)
        recordings.update(found)
        n_queries += len(start)
        n_rows += manifest['rows']
        need(sorted(v['seed'] for v in views) == list(SEEDS) and cap['by_site'][site]['views'] == len(found)*len(SEEDS),
             'OpenDD seed or recording coverage changed')
        for v in views:
            seed = v['seed']
            need(set(v['teacher']) == sites-{site}, 'OpenDD teacher labels include held or omit donor location')
            head = outer['heads'][f'{site}/{seed}/additive_rank']
            fitting_scope(head['identity'], sites, [site], 'donors')
            need(head['identity']['held'] == site and head['identity']['seed'] == seed
                 and head['identity']['registration'] == outer['registration'], 'Wrong outer ranker for the held location')
            checked(head['artifact'])
            need(v['scores']['recording']['views'] == len(found), 'OpenDD past-only roster differs from cohort')
            past_only += v['scores']['recording']['dual_positive']
    n = len(recordings)
    hindsight = cap['summary']['nested_stream']['verified_positive_both']
    need((n, past_only, hindsight) == (202, 57, 330), 'OpenDD main paper counts changed')
    return dict(role='exposed_development', locations=len(sites), recordings=n, repeated_seed_views=n*len(SEEDS),
                queries=n_queries, agent_windows=n_rows, checked_unique_publisher_ids=len(all_ids),
                nonoverlapping_queries_within_recording=True, pair_excluded_rankers=len(frozen['rank_heads']),
                donor_only_count_heads=len(frozen['count_heads']), held_location_rankers=21,
                hindsight_verified=330, past_only_verified=57)


def moving():
    fit = PUB/'moving_dual_record_forecast_v1'
    audit = PUB/'moving_record_peak_allocation_v1'
    reg, bank = (read(fit/(k+'.json')) for k in ('registration', 'freeze'))
    protocol, score = (read(audit/(k+'.json')) for k in ('registration', 'readout'))
    need(reg['select_only_on_calibration'] and reg['seeds'] == list(SEEDS) and bank['before_held_scoring']
         and bank['registration'] == sha(fit/'registration.json') and score['registration'] == sha(audit/'registration.json')
         and fit/'freeze.json' in {checked(ref) for ref in protocol['parents']}
         and not any(score[k] for k in ('confirmation_access', 'a_labels_opened', 'independent_confirmation')),
         'Moving development provenance changed')
    roles, cells, selected_sources = {}, set(), {}
    for cell in bank['cells']:
        i = cell['identity']
        parts = tuple(frozenset(i[k]) for k in ('train', 'calibration', 'held'))
        need(all(len(i[k]) == len(p) == 3 for k, p in zip(('train', 'calibration', 'held'), parts, strict=True))
             and len(frozenset.union(*parts)) == 9, 'Moving fit, calibration and held sources overlap')
        fold, seed = i['fold'], i['seed']
        need((fold, seed) not in cells and roles.setdefault(fold, parts) == parts, 'Moving split differs by seed')
        cells.add((fold, seed))
        selected_sources[fold, seed] = parts[-1]
        for arm, model in cell['fits'].items():
            need(model['identity'] == {**i, 'registration': bank['registration'], 'arm': arm},
                 'Moving model was not fitted under the declared fold identity')
            checked(model['artifact'])
        checked(cell['paths'])
    universe = set().union(*(r[-1] for r in roles.values()))
    need(cells == set(product(range(3), SEEDS)) and len(universe) == 9
         and all(frozenset.union(*r) == universe for r in roles.values()),
         'Moving held sources do not partition the nine-source development roster')
    recording = {}
    for v in score['rows']:
        i = v['identity']
        need(i['source'] in selected_sources[i['fold'], i['seed']], 'Moving readout uses a nonheld source')
        source, seeds = recording.setdefault(i['recording'], (i['source'], set()))
        need(source == i['source'] and i['seed'] not in seeds, 'Moving recording is duplicated or relabeled')
        seeds.add(i['seed'])
    need(len(recording) == 24 and all(v == set(SEEDS) for _, v in recording.values()), 'Moving seed coverage changed')
    return dict(role='exposed_development_hindsight_bound', sources=9, recordings=24, repeated_seed_views=len(score['rows']),
                folds=3, fit_calibration_held_sources_per_fold=[3, 3, 3], frozen_path_banks=len(cells))


def wepdtof():
    root = PUB/'wepdtof_external_transfer_v1'
    reg, fits, pilot, bank, prereg, aug, scored = (read(root/(k+'.json')) for k in (
        'registration', 'head_freeze', 'pilot_freeze', 'forecast_freeze', 'evaluation_registration',
        'augmented_freeze', 'external_readout'))
    rh, fh = (sha(root/(k+'.json')) for k in ('registration', 'head_freeze'))
    need(reg['only_donor_fitting'] and reg['no_external_selection'] and fits['donor_only']
         and not fits['external_labels_read'] and not pilot['external_geometry_read']
         and bank['no_external_outcomes_read'] and aug['outcomes_unopened'] and not prereg['selection_on_external']
         and not scored['external_tuning'] and scored['external_scoring_completed'] and scored['original_2pct']
         and fits['registration'] == pilot['registration'] == bank['registration'] == prereg['training_registration'] == rh
         and pilot['head_sha256'] == fh and checked(bank['pilot']) == checked(prereg['pilot']) == root/'pilot_freeze.json'
         and checked(aug['original_prediction_freeze']) == root/'forecast_freeze.json'
         and aug['evaluation_registration'] == sha(root/'evaluation_registration.json')
         and scored['forecast_freeze'] == sha(root/'augmented_freeze.json'), 'WEPDTOF train, pilot or score freeze mismatch')
    roles = read(checked(reg['approved_roles']))
    approved = {(s, c) for s, g in roles['groups'].items() if g['role'] == 'frozen_external_once' for c in g['clips']}
    pilot_sources = [s for s, g in roles['groups'].items() if g['role'] == 'development']
    need(len(pilot_sources) == 1 and len({s for s, _ in approved}) == 13 and len(approved) == 15
         and roles['roles_frozen'] and not roles['selection_on_external'], 'Wrong WEPDTOF source roles')
    for roster in (bank['rows'], aug['rows'], scored['rows']):
        need({(v['source'], v['clip']) for v in roster} == approved and len(roster) == len(approved),
             'WEPDTOF external roster differs from preregistration')
    need(sorted(m['seed'] for m in fits['models']) == list(SEEDS), 'Expected all WEPDTOF donors and seeds')
    for m in fits['models']:
        need(m['identity']['registration'] == rh and m['identity']['seed'] == m['seed']
             and set(m['identity']['donors']) == set(reg['donor_cohorts']),
             'A WEPDTOF head was not donor-only')
        checked(m['checkpoint'])
    for v in (*bank['rows'], *aug['rows']):
        checked(v['artifact'])
    need(all(sorted(map(int, v['seeds'])) == list(SEEDS) for v in scored['rows']), 'WEPDTOF score lacks seeds')
    return dict(role='frozen_external_once', development_sources=1, external_sources=13, clips=15,
                repeated_seed_views=45, fit_checkpoints=3, frozen_forecast_packets=30)


def songdo():
    local, public = DATA/'songdo_confirmation_v1', PUB/'songdo_calibrated_20261110'
    reg, freeze = (read(local/(k+'.json')) for k in ('registration', 'freeze'))
    cal, score = (read(public/(k+'_receipt.json')) for k in ('calibration', 'confirmation'))
    b = score['binding']
    need(b['registration_sha256'] == cal['registration_sha256'] == freeze['registration_sha256'] == sha(local/'registration.json')
         and b['release_freeze_sha256'] == cal['freeze_sha256'] == sha(local/'freeze.json')
         and b['calibration_cutoffs_sha256'] == cal['calibration_cutoffs_sha256'] == freeze['calibration_cutoffs_sha256']
         == sha(local/'calibration_cutoffs.json') and score['seeds'] == list(SEEDS)
         and b['published_calibration_receipt_sha256'] == sha(public/'calibration_receipt.json')
         and score['original_2pct_unchanged'] and not cal['confirmation_outcomes_opened'] and freeze['thresholds'] == cal['thresholds']
         and score['no_fitting_on_confirmation'], 'Songdo calibration freeze changed')
    groups = {role: [g for r, g in reg['flight_roles'].values() if r == role] for role in ('calibration', 'confirmation')}
    need(len(reg['flight_roles']) == 200 and [(len(g), len(set(g))) for g in groups.values()] == [(40, 3), (160, 14)]
         and not set(groups['calibration']) & set(groups['confirmation']), 'Songdo roles overlap or omit flights')
    before, after = '0c7919432', '4c6c1a50e'
    subprocess.run(['git', 'merge-base', '--is-ancestor', before, after], cwd=ROOT, check=True)
    for commit, item in ((before, 'calibration'), (after, 'confirmation')):
        file = public/(item+'_receipt.json')
        content = subprocess.check_output(['git', 'show', f'{commit}:{file.relative_to(ROOT)}'], cwd=ROOT)
        need(content == file.read_bytes(), 'Songdo local receipts differ from their committed order')
    return dict(role='one_shot_with_source_disjoint_calibration_same_acquisition_day', calibration_flights=40,
                calibration_components=3, confirmation_flights=160, confirmation_components=14,
                repeated_seed_views=score['flight_seed_views'], pre_confirmation_freeze_commit_verified=True)


def dronecrowd():
    calibration = read(PUB/'dronecrowd_calibration_readout_v1/analysis.json')
    registered = PUB/'dronecrowd_confirmation_readout_v1'
    reg = read(registered/'registration.json')
    need(calibration['calibration_outcomes_opened'] and not calibration['calibration_pass']
         and not calibration['confirmation_outcome_access'] and not (registered/'analysis.json').exists()
         and reg['confirmation_role_counts'] == [30, 72, 216] and not reg['threshold_search_on_confirmation']
         and reg['original_risk_budget'] == .02,
         'DroneCrowd gate or confirmation status changed')
    return dict(role='calibration_only', calibration_pass=False, reserved_source_groups=30,
                reserved_recordings=72, confirmation_readout_found=False, sealed_outcomes_opened_by_audit=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUT)
    a = p.parse_args()
    report = dict(opendd=opendd(), moving=moving(), wepdtof=wepdtof(), songdo=songdo(), dronecrowd=dronecrowd(),
                  outcome_arrays_read_by_audit=False, training_performed=False, new_independent_confirmation=False,
                  result_source='fresh_lineage_audit_of_frozen_artifacts', submission_ready=False)
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        need(a.output.read_bytes() == raw, 'Refusing to replace a different evidence receipt')
    else:
        with a.output.open('xb') as f:
            f.write(raw)
    print(raw.decode(), end='')
