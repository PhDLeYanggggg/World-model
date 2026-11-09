"""Test fixed-score release transport without changing the risk contract."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.vehicle_recording_replay import verify

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'
FIT = 'opendd_vehicle_fixed_release_v1'
OUT = PUBLIC/'opendd_vehicle_release_transport_v1'
ARMS = ('archived', 'direct', 'nested')
POLICIES = ('numeric', 'matched_count', 'matched_fraction')


def groups(scores):
    sets = defaultdict(list)
    for i, s in enumerate(scores):
        if s is not None:
            if not isfinite(s):
                raise ValueError('Expected finite scores or missing')
            sets[s].append(i)
    return [sets[k] for k in sorted(sets, reverse=True)]


def count_prefix(groups_, n):
    """Never split a tied group to fill the requested budget."""
    use = []
    for group in groups_:
        if len(use)+len(group) > n:
            break
        use.extend(group)
    return use


def outcome(labels, use):
    risk = sum(labels[i][0] for i in use)
    missing = sum(labels[i][3] for i in use)
    return dict(released=len(use), directions=int(bool(use)), risk=risk,
                missing_easy=missing, other_risk_failure=len(use)-risk-missing,
                full=sum(labels[i][2] for i in use),
                all_full_directions=int(bool(use) and all(labels[i][2] for i in use)))


def comparison(cal, cal_labels, held, labels):
    threshold, n = verify.prefix(zip(cal, (x[2] for x in cal_labels), strict=True))
    eligible_cal = sum(x is not None for x in cal)
    grouped = groups(held)
    eligible = sum(map(len, grouped))
    if eligible == 0 or eligible_cal == 0:
        raise ValueError('A direction lacks finite scores')
    finite = [i for g in grouped for i in g]
    ordered = []
    for group in grouped:
        if not all(labels[i][2] for i in group):
            break
        ordered.extend(group)
    available = sum(labels[i][2] for i in finite)
    floor_fraction = n*eligible//eligible_cal
    choices = dict(numeric=[i for i in finite if threshold is not None and held[i] >= threshold],
                   matched_count=count_prefix(grouped, min(n, eligible)),
                   matched_fraction=count_prefix(grouped, floor_fraction))
    output = {}
    for a, picked in choices.items():
        stats = outcome(labels, picked)
        stats.update(uniform_expected_full=len(picked)*available/eligible,
                     hindsight_any_order_full=min(len(picked), available))
        output[a] = stats
    top = grouped[0]
    return dict(threshold=threshold, calibration_prefix=n, calibration_available=eligible_cal,
                held_available=eligible, held_full=available, held_hindsight_rank_prefix=len(ordered),
                held_top_tie=outcome(labels, top), held_below_numeric_cut=int(n > 0 and held[top[0]] < threshold),
                policies=output)


def inputs():
    def get(path):
        raw = path.read_bytes()
        return json.loads(raw), sha256(raw).hexdigest()

    rank, parent_hash = get(PUBLIC/FIT/'readout.json')
    f, freeze_hash = get(PUBLIC/FIT/'freeze.json')
    prior, prior_hash = get(PUBLIC/'opendd_vehicle_anchor_rank_v1/freeze.json')
    registration, reg_hash = get(PUBLIC/FIT/'registration.json')
    sites = get(PUBLIC/'opendd_vehicle_anchor_rank_v1/readout.json')[0]['sites']
    refs = registration['files']
    relative_path = str((PUBLIC/'opendd_vehicle_anchor_rank_v1/freeze.json').relative_to(ROOT))
    scalar_path = PUBLIC/'vehicle_recording_replay_v1/receipt.json'
    scalar, scalar_hash = get(scalar_path)
    if (rank['freeze'] != freeze_hash or rank['registration'] != reg_hash or f['registration'] != reg_hash
            or refs[relative_path] != prior_hash or refs[str(scalar_path.relative_to(ROOT))] != scalar_hash
            or not rank['same_proposals'] or rank['held_tuning'] or rank['external_outcomes_read']
            or not rank['original_2pct_unchanged'] or not f['scores_frozen']
            or scalar['freeze_sha256'] != prior_hash or set(f['cells']) != set(prior['cells'])):
        raise ValueError('Require the original frozen release experiment')
    bundle = PRIVATE/'submission_admin/vehicle_recording_replay.zip'
    if sha256(bundle.read_bytes()).hexdigest() != scalar['archive_sha256']:
        raise ValueError('Anonymous scoring terms changed')
    with ZipFile(bundle) as z:
        evidence = json.loads(z.read('evidence.json'))
    data = {(c['site'], c['calibration'], c['seed']): c for c in evidence['directions']}
    scores, ids = {}, {}
    for name, cell in prior['cells'].items():
        r, h = get(PRIVATE/FIT/f'{name}.json')
        if h != f['cells'][name] or r['registration'] != reg_hash:
            raise ValueError('Private frozen score receipt changed')
        for s, p in r['packets'].items():
            k = (sites.index(s), sites.index(cell['rules'][s]['calibration_site']), cell['seed'])
            records = data[k]['recordings']
            if [ids.setdefault(x, len(ids)) for x in p['recordings']] != [x['id'] for x in records]:
                raise ValueError('Recorded order changed')
            scores[k] = dict(archived=[x['arms']['relative']['score'] for x in records], **p['scores'])
    if len(ids) != 493 or len(data) != 126 or set(data) != set(scores):
        raise ValueError('Incomplete roster')
    labels = {k: [verify.qualification(x['arms']['relative']) for x in c['recordings']] for k, c in data.items()}
    return labels, scores, rank, evidence['source_resamples'], dict(
        fixed_readout_sha256=parent_hash, fixed_freeze_sha256=freeze_hash,
        replay_archive_sha256=scalar['archive_sha256'])


def run():
    labels, scores, rank, draws, source = inputs()
    saved = {(r['site'], r['calibration'], r['seed']): r['arms'] for r in rank['rules']}
    totals = {a: Counter() for a in ARMS}
    policies = {a: {p: Counter() for p in POLICIES} for a in ARMS}
    gains = {a: {p: defaultdict(list) for p in POLICIES} for a in ARMS}
    for k, y in labels.items():
        s, c, seed = k
        opposite = (c, s, seed)
        for a in ARMS:
            z = comparison(scores[opposite][a], labels[opposite], scores[k][a], y)
            original = saved[k][a]
            n, t = z['calibration_prefix'], z['threshold']
            measured = z['policies']['numeric']
            want = original['held']
            if ([t, n] != [original['threshold'], original['calibration_count']] or
                    [measured[x] for x in ('released', 'risk', 'full', 'missing_easy')] !=
                    [want[x] for x in ('admitted', 'risk_pass', 'full_win', 'missing')]):
                raise ValueError('Numeric-threshold control does not replay')
            prefix = z['held_hindsight_rank_prefix']
            totals[a].update(
                calibration_enabled=int(n > 0),
                held_full=z['held_full'], held_has_full=int(z['held_full'] > 0),
                hindsight_prefix_full=prefix, hindsight_prefix_directions=int(prefix > 0),
                enabled_has_full=int(n > 0 and z['held_full'] > 0),
                enabled_top_fully_qualified=int(n > 0 and prefix > 0),
                enabled_score_below_cut=z['held_below_numeric_cut'],
                enabled_score_below_cut_but_top_qualifies=int(z['held_below_numeric_cut'] and prefix > 0),
                enabled_top_blocked_by_missing_easy=int(n > 0 and prefix == 0 and z['held_top_tie']['missing_easy'] > 0))
            for p, q in z['policies'].items():
                policies[a][p].update(q)
                gains[a][p][s].append(100*(q['full']-q['uniform_expected_full'])/z['held_available'])
    return dict(binding=source, result_source='fresh_replay_of_frozen_scores', original_2pct_unchanged=True,
                external_outcomes_read=False, independent_confirmation=False, deployed_policy_changed=False,
                diagnostics_only=True, score_tuning=False, directions=len(labels), recordings=rank['recordings'],
                views=rank['views'], topology=totals, policies=policies,
                full_gain_pp_per_available_vs_matched_uniform={
                    a: {p: verify.interval(rows, draws) for p, rows in q.items()} for a, q in gains.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT/'readout.json')
    p = parser.parse_args().output
    result = run()
    value = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text() != value:
        raise ValueError('Recorded transport readout differs')
    if not p.exists():
        with p.open('x') as f:
            f.write(value)
    print(json.dumps({k: result[k] for k in ('topology', 'policies')}, indent=2))
