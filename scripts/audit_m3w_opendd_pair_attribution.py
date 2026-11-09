"""Attribute matched-menu OpenDD differences without selecting a new policy."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_slate_shared_menu as shared
from scripts import run_m3w_pair_effect as direct

exp, study, base, np = shared.exp, shared.study, shared.base, shared.np
OUT = shared.OUT.parent / 'opendd_pair_attribution_v1'
QUERY_KINDS = ('same_path', 'path_only', 'recipient_change')
EDGE_KINDS = ('both_changed', 'changed_with_edited', 'changed_with_cv', 'unchanged')
EPS = 1e-9


def query(cv, additive, joint, labels, valid):
    radii = [np.linalg.norm(p - cv, axis=-1).max(axis=1) for p in (additive, joint)]
    a, b = [r > 0. for r in radii]
    np.testing.assert_allclose(np.sort(radii[0]), np.sort(radii[1]), rtol=EPS, atol=EPS)
    if a.sum() != b.sum():
        raise ValueError('Mismatched edit count')
    changed = ~np.isclose(additive, joint, rtol=EPS, atol=EPS).all(axis=(1, 2))
    kind = 'same_path' if not changed.any() else 'path_only' if np.array_equal(a, b) else 'recipient_change'
    if kind != 'same_path' and a.sum() != 2:
        raise ValueError('Joint changes outside two-edit queries')

    known, lower = study.paired(additive, joint, labels, valid)
    mass = float(np.where(valid, np.linalg.norm(cv - labels, axis=-1), 0.).sum() / len(cv))
    known, lower = float(known.sum() / len(cv)), float(lower.sum() / len(cv))
    # For relative error use the original all-co-observed-step query denominator.
    i, j = np.triu_indices(len(cv), 1)
    common = valid[i] & valid[j]
    relative = labels[i] - labels[j]
    ref_err = np.linalg.norm(additive[i] - additive[j] - relative, axis=-1)
    joint_err = np.linalg.norm(joint[i] - joint[j] - relative, axis=-1)
    delta = np.where(common, ref_err - joint_err, 0.)
    both = changed[i] & changed[j]
    one = changed[i] ^ changed[j]
    edited = np.where(changed[i], a[j] | b[j], a[i] | b[i])
    bins = (both, one & edited, one & ~edited, ~(one | both))
    edge = {k: dict(pairs=int(mask.sum()), steps=int(common[mask].sum()), gain=float(delta[mask].sum()))
            for k, mask in zip(EDGE_KINDS, bins)}
    return dict(kind=kind, edits=int(a.sum()), changed_actors=int(changed.sum()),
                replaced_slots=int((a & ~b).sum()), changed_actor_steps=int(valid[changed].sum()),
                changed_missing_steps=int((~valid[changed]).sum()), mass=mass, known=known, lower=lower,
                edge=edge, common_steps=int(common.sum()))


def recording(queries):
    mass = sum(q['mass'] for q in queries)
    active = mass > 0 and any(q['edits'] for q in queries)
    stats = Counter()
    gains = defaultdict(Counter)
    total_edge = Counter()
    supported = sum(q['common_steps'] > 0 for q in queries)
    for q in queries:
        key = q['kind']
        if q['edits']:
            stats['edited_queries'] += 1
            stats['edits'] += q['edits']
            stats[key] += 1
        for field in ('changed_actors', 'replaced_slots', 'changed_actor_steps', 'changed_missing_steps'):
            stats[f'{key}/{field}'] += q[field]
        if key != 'same_path':
            stats[f'{key}/known_win'] += q['known'] > EPS
            stats[f'{key}/lower_win'] += q['lower'] > EPS
            stats[f'{key}/lower_loss'] += q['lower'] < -EPS
        if active:
            for field, amount in (('known', q['known']), ('lower', q['lower']),
                                  ('debit', q['known'] - q['lower']), ('hindsight', max(0., q['lower']))):
                gains[key][field] += 100. * amount / mass
        for k, v in q['edge'].items():
            if key != 'same_path':
                stats[f'{key}/{k}/pairs'] += v['pairs']
                stats[f'{key}/{k}/observed_steps'] += v['steps']
            if active and supported and q['common_steps']:
                total_edge[k] += v['gain'] / q['common_steps'] / supported
    return (stats, {k: dict(v) for k, v in gains.items()} if active else None,
            dict(total_edge) if active and supported else None)


def summarize(rows):
    count = sum((r[0] for r in rows), Counter())
    gains = [r[1] for r in rows if r[1] is not None]
    edges = [r[2] for r in rows if r[2] is not None]
    by_kind = {k: {f'{field}_pp': float(np.mean([g.get(k, {}).get(field, 0.) for g in gains]))
                   if gains else None for field in ('known', 'lower', 'debit', 'hindsight')} for k in QUERY_KINDS}
    total = {k: sum(v[k] for v in by_kind.values()) if gains else None for k in by_kind[QUERY_KINDS[0]]}
    by_edge = {k: float(np.mean([v.get(k, 0.) for v in edges])) if edges else None for k in EDGE_KINDS}
    return dict(views=len(rows), estimable=len(gains), geometry_estimable=len(edges), count=dict(count),
                utility_by_query=by_kind, total=total, geometry_by_edge_native=by_edge,
                geometry_total_native=sum(by_edge.values()) if edges else None)


def interval(values):
    x = np.asarray([v for v in values if v is not None])
    draws = np.random.default_rng(3183).choice(x, size=(3000, len(x))).mean(axis=1)
    return dict(sites=len(x), mean=float(x.mean()), ci95=np.quantile(draws, [.025, .975]).tolist())


def menu_hindsight(queries):
    mass = sum(a['mass'] for a, _ in queries)
    if mass <= 0 or not any(a['edits'] for a, _ in queries):
        return None
    return {k: 100. * sum(max(0., a[k], b[k]) for a, b in queries) / mass for k in ('known', 'lower')}


def inputs():
    refs = {}
    for name, path in (('imitation', shared.OUT), ('direct', direct.OUT)):
        reg, fr, readout = [path / p for p in ('registration.json', 'freeze.json', 'readout.json')]
        f, r = study.read(fr), study.read(readout)
        if (f['registration'] != base.sha(reg) or study.checked(r['freeze']) != fr
                or r['reserved_outcomes_opened'] or r['independent_confirmation']):
            raise ValueError('Require the original development-only receipts')
        refs[name] = (f, r, [study.artifact(x) for x in (reg, fr, readout)])
    original_reg = study.checked(study.read(shared.OUT / 'registration.json')['parent'])
    cohorts = study.read(original_reg)['cohorts']
    if set(cohorts) != set(shared.SITES):
        raise ValueError('Unregistered source')
    return refs, cohorts


def run():
    refs, cohorts = inputs()
    data = exp.loader(cohorts)
    rows = {name: defaultdict(list) for name in refs}
    hindsight = defaultdict(list)
    for site, d in data.items():
        for seed in shared.SEEDS:
            fold = f'{site}_{seed}'
            paths = {}
            for name, (freeze, _, _) in refs.items():
                with np.load(study.checked(freeze['forecasts'][fold]), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['q'], d['q'])
                    np.testing.assert_array_equal(z['cv'], d['cv'])
                    paths[name] = {k: z[k] for k in ('matched_independent', 'joint')}
            np.testing.assert_array_equal(paths['imitation']['matched_independent'],
                                          paths['direct']['matched_independent'])
            records = defaultdict(list)
            for ix in np.split(np.arange(len(d['q'])), np.flatnonzero(np.diff(d['q'])) + 1):
                if len(ix) == 0 or np.any(d['rec'][ix] != d['rec'][ix[0]]):
                    raise ValueError('Query is empty or spans multiple recordings')
                records[d['rec'][ix[0]]].append(tuple(
                    query(d['cv'][ix], p['matched_independent'][ix], p['joint'][ix], d['y'][ix], d['v'][ix])
                    for p in paths.values()))
            for qs in records.values():
                for name, arm in zip(paths, zip(*qs)):
                    rows[name][site].append(recording(arm))
                score = menu_hindsight(qs)
                if score is not None:
                    hindsight[site].append(score)
        print(site, flush=True)
    arms = {}
    for name, groups in rows.items():
        per_site = {s: summarize(r) for s, r in groups.items()}
        _, old, _ = refs[name]
        for site, v in per_site.items():
            prev = old['groups'][site]
            np.testing.assert_equal((v['views'], v['estimable'], v['count'].get('edits', 0)),
                                    (prev['views'], prev['estimable'], prev['arms']['joint']['edits']))
            if not v['estimable']:
                continue
            for k in ('known_pp', 'lower_pp'):
                np.testing.assert_allclose(v['total'][k], prev[k], atol=1e-12, rtol=0.)
            geometry_key = ('observed_pair_error_reduction_native' if name == 'imitation'
                            else 'pair_error_gain_native')
            np.testing.assert_allclose(v['geometry_total_native'], prev[geometry_key], atol=1e-12, rtol=0.)
        counts = sum((Counter(v['count']) for v in per_site.values()), Counter())
        expected_changes = (old['inference_totals']['joint_differs_additive'] if name == 'imitation'
                            else old['totals']['direct_differs_additive'])
        if counts['path_only'] + counts['recipient_change'] != expected_changes:
            raise ValueError('Published action-disagreement count differs')
        arms[name] = dict(per_site=per_site, pooled_counts=dict(counts),
                          total_intervals={k: interval(v['total'][k] for v in per_site.values())
                                           for k in ('known_pp', 'lower_pp', 'debit_pp', 'hindsight_pp')},
                          geometry_intervals={k: interval(v['geometry_by_edge_native'][k] for v in per_site.values())
                                              for k in EDGE_KINDS})
    ceilings = {s: {k: float(np.mean([v[k] for v in rs])) for k in ('known', 'lower')}
                for s, rs in hindsight.items()}
    return dict(result_source='fresh_run', forecasts_and_cohorts='cached_verified', arms=arms,
                combined_hindsight=dict(per_site=ceilings, risk_qualification_scored=False,
                                        contrasts={k: interval(v[k] for v in ceilings.values())
                                                   for k in ('known', 'lower')}),
                prior_receipts={k: v[2] for k, v in refs.items()}, cohorts=cohorts,
                original_2pct_unchanged=True, independent_confirmation=False, deployment_changed=False,
                reserved_outcomes_opened=False, stage5c=False, smc=False, submission_ready=False)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    base.immutable(OUT / 'readout.json', result)
    for name, arm in result['arms'].items():
        print(name, arm['total_intervals'], flush=True)


if __name__ == '__main__':
    main()
