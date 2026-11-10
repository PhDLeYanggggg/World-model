"""Audit the size of frozen Songdo contrasts without loading outcomes."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.m3w_songdo_matched_readout import ARMS, SEEDS, matched_paths


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'data/stage_cvpr2027_experiments'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
SOURCE = PUBLIC/'songdo_first_day_v1'
PROPOSALS = PUBLIC/'songdo_matched_proposals_v1/receipt.json'
OUT = PUBLIC/'songdo_contrast_support_v1/receipt.json'
TOL = 1e-9


def contrast(left, right, left_choices, right_choices, windows):
    """Upper magnitudes for signed and conservative contrasts on a fixed grid."""
    delta = left-right
    d = np.linalg.norm(delta, axis=-1)
    n = len(d)
    i, j = np.triu_indices(n, k=1)
    relative = np.linalg.norm(delta[i]-delta[j], axis=-1)
    shared = set(left_choices) & set(right_choices)
    return dict(shared_recipients=len(shared), family_changes_on_shared=sum(
                left_choices[r] != right_choices[r] for r in shared),
                changed_rows=int((d.max(1) > TOL).sum()),
                relative_nonzero=bool(relative.max() > TOL),
                query_mean_waypoint_upper_native=float(d.mean()),
                flight_grid_mean_waypoint_upper_native=float(d.mean()/windows),
                query_pair_grid_mean_upper_native=float(relative.mean()))


def seed_view(cv, offsets, z, seed, info):
    prefix = f'seed{seed}_'
    lo, hi = z[prefix+'bounds']
    query = int(np.searchsorted(offsets, lo))
    if (query >= len(offsets)-1 or offsets[query] != lo or offsets[query+1] != hi
            or (query, int(hi-lo)) != (info['query'], info['agents'])
            or z[prefix+'query_index'].item() != query or z[prefix+'legal_pairs'].item() != info['pairs']):
        raise ValueError('Frozen query does not match the receipt')
    radius = z[prefix+'radius'].item()
    if radius != info['radius']:
        raise ValueError('Frozen radius does not match the receipt')
    choices, edits = {}, {}
    for arm in ARMS:
        ids = z[prefix+arm+'_rows']-lo
        families = z[prefix+arm+'_families']
        if families.shape != (2,) or not np.issubdtype(families.dtype, np.integer) or not ((0 <= families) & (families < 3)).all():
            raise ValueError('Invalid frozen forecast family')
        choices[arm] = dict(zip(ids.tolist(), families.tolist(), strict=True))
        edits[arm] = ids, z[prefix+arm+'_paths']
    _, arms = matched_paths(cv[lo:hi], np.array([0, hi-lo]), dict(bounds=(0, hi-lo), radius=radius, arms=edits))
    if len(set(choices['same_family'].values())) != 1:
        raise ValueError('The common-family control changed')
    result = {}
    for arm in ARMS[:-1]:
        a, j = choices[arm], choices['joint']
        if (a != j) != info['joint_differs'][arm]:
            raise ValueError('Archived discrete contrast does not reproduce')
        result[arm] = contrast(arms['joint'], arms[arm], j, a, len(offsets)-1)
    return result


def digest(path):
    import hashlib
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(values):
    def stats(field):
        return dict(zip(('median', 'p90', 'maximum'), np.quantile(
            [v[field] for _, v in values], [.5, .9, 1.]).tolist(), strict=True))

    groups = {g for g, _ in values}
    return dict(views=len(values), candidate_source_components=len(groups),
                no_shared_recipients=sum(v['shared_recipients'] == 0 for _, v in values),
                one_shared_recipient=sum(v['shared_recipients'] == 1 for _, v in values),
                both_recipients_shared=sum(v['shared_recipients'] == 2 for _, v in values),
                same_recipients_family_change=sum(v['shared_recipients'] == 2 and v['family_changes_on_shared'] > 0
                                                  for _, v in values),
                different_trajectory_views=sum(v['changed_rows'] > 0 for _, v in values),
                different_relative_motion_views=sum(v['relative_nonzero'] for _, v in values),
                components_with_trajectory_difference=len({g for g, v in values if v['changed_rows'] > 0}),
                upper={k: stats(k) for k in ('query_mean_waypoint_upper_native', 'flight_grid_mean_waypoint_upper_native',
                                            'query_pair_grid_mean_upper_native')})


def run():
    records = read(PROPOSALS)
    parents = {k: SOURCE/p for k, p in (('prefix_receipt', 'input_receipt.json'),
                                        ('candidate_bank', 'vehicle_bank.json'), ('lineage', 'lineage.json'))}
    if (records['arms'] != list(ARMS) or not records['original_2pct_unchanged'] or any(
            records[k] for k in ('split_assigned', 'future_index_loaded', 'future_coordinates_loaded', 'forecasts_scored',
                                 'calibration_thresholds_fitted', 'independent_confirmation'))
            or any(records['binding'][k] != digest(p) for k, p in parents.items())):
        raise ValueError('Require split-neutral predictions and the original role-free receipt')
    prefix, bank, lineage = (read(p) for p in parents.values())
    sources = {s: i for i, part in enumerate(lineage['candidate_source_components']) for s in part}
    roster = records['sessions']
    if (len(roster) != 200 or len(set(sources.values())) != 17 or set(roster) != set(bank['sessions'])
            or set(roster) != set(prefix['sessions']) or any(
                b['split_assigned'] or b['forecasts_scored'] for b in (prefix, bank))
            or bank['future_index_loaded'] or bank['future_coordinates_loaded']):
        raise ValueError('The frozen observation-only roster changed')
    data = {a: [] for a in ARMS[:-1]}
    eligible, eligible_components = 0, set()
    by_seed = {str(s): 0 for s in SEEDS}
    for name, entry in roster.items():
        stem, site = Path(name).stem, entry['site']
        predicted, observed = bank['sessions'][name], prefix['sessions'][name]
        proposal = LOCAL/'songdo_matched_proposals_v1'/f'{stem}.npz'
        paths = LOCAL/'songdo_first_day_v1/vehicle_bank'/f'{stem}.npz'
        if (site != predicted['site'] or site != observed['site'] or observed['prefix_sha'] != predicted['prefix_sha']
                or digest(proposal) != entry['packet_sha256'] or digest(paths) != predicted['sha256']):
            raise ValueError('Frozen input differs from its receipt')
        flags = [entry['seeds'][str(s)] is not None for s in SEEDS]
        if any(flags) != all(flags):
            raise ValueError('Expected the same eligible flights for all seeds')
        with np.load(proposal, allow_pickle=False) as z, np.load(paths, allow_pickle=False) as b:
            for s, active in zip(SEEDS, flags, strict=True):
                if active != (f'seed{s}_bounds' in z):
                    raise ValueError('Frozen packet eligibility does not match the receipt')
            if not any(flags):
                continue
            eligible += 1
            group = sources[site]
            eligible_components.add(group)
            cv, offsets = b['cv'], b['offsets']
            if (cv.shape != (observed['agents'], 12, 2) or offsets.shape != (observed['queries']+1,)
                    or offsets[0] != 0 or offsets[-1] != len(cv) or (np.diff(offsets) < 2).any()):
                raise ValueError('The original 8-to-12 prediction roster changed')
            for s in map(str, SEEDS):
                by_seed[s] += 1
                for arm, view in seed_view(cv, offsets, z, s, entry['seeds'][s]).items():
                    data[arm].append((group, view))
    if (eligible, len(eligible_components), set(by_seed.values())) != (191, 17, {191}):
        raise ValueError('The original matched coverage did not reproduce')
    return dict(flights=len(roster), matched_flights=eligible, matched_flight_seed_views=sum(by_seed.values()),
                matched_source_components=len(eligible_components), matched_flights_by_seed=by_seed,
                joint_minus={a: summarize(v) for a, v in data.items()},
                inputs_sha256=dict(proposals=digest(PROPOSALS), **{k: digest(p) for k, p in parents.items()}),
                result_source='fresh_prediction_contrast_audit_cached_verified_frozen_proposals',
                original_2pct_unchanged=True, split_assigned=False, outcomes_opened=False,
                future_index_opened=False, selection_performed=False, submission_ready=False)


if __name__ == '__main__':
    result = (json.dumps(run(), sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_bytes() != result:
        raise ValueError('Refusing to replace a different contrast audit')
    if not OUT.exists():
        with OUT.open('xb') as f:
            f.write(result)
    print(result.decode(), end='')
