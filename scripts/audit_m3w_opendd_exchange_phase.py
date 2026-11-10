"""Test residual timing at identical OpenDD exchange recipients and radii."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_stability as stability
from src.world_model import m3w_exchange_phase as phase


run, before, base, np, torch = stability.run, stability.before, stability.base, stability.np, stability.torch
matched, parent = stability.matched, stability.parent
OUT = stability.OUT.parent/'opendd_exchange_phase_v1'
ARMS = (*phase.SHAPES, 'prefix_ols', 'uniform_phase')
ORACLES = (*phase.SHAPES, 'prefix_ols', 'expanded')
FIELDS = ('dual', 'risk_pass', 'undefined')


def assess(d, cv, ols, p, cut):
    s = parent.score(d, cv, ols, p, cut)
    return s, dict(dual=s['dual_positive'], risk_pass=s['risk_pass'], undefined=s['undefined'])


def closure(view, result, anchor, stable):
    for k in ORACLES:
        available = list(result.values()) if k == 'expanded' else [result[k]]
        pass_scores = [min(s['vs_cv'], s['vs_ols4']) for s in available if s['risk_pass']]
        floor = min(anchor['vs_cv'], anchor['vs_ols4']) if anchor['risk_pass'] else -np.inf
        best = max([floor, *pass_scores])
        for f, t in (('positive', 1e-6), ('gain_gt_0_10pp', .10), ('gain_gt_0_50pp', .50)):
            if best > t:
                view[k].add(f)
        if not anchor['dual_positive'] and (any(stable.values()) if k == 'expanded' else stable[k]):
            view[k].add('stable_rescue')


def evaluate(site, seed, motion_key, d, bank, menu, width, nets, cut):
    x, legal, rows, path, _ = menu
    paths = phase.phase_paths(bank[0], path)
    original_radius = np.linalg.norm(path-bank[0], axis=-1).max(1)
    for p in paths.values():
        np.testing.assert_allclose(np.linalg.norm(p-bank[0], axis=-1).max(1), original_radius, atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(p[:, -1], path[:, -1])
    phase_choice = phase.prefix_choices(paths, bank[1])
    move = base.model.movement_prefix(bank[0], path, rows)
    totals, counts = {k: Counter() for k in ARMS}, Counter()
    oracle, gain = {k: Counter() for k in ORACLES}, Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = before.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion_key, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound anchor')
        cv, ols, original = (v[ai] for v in (bank[0], bank[1], path))
        local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}
        query = np.searchsorted(np.flatnonzero(q), d['query'][ai])
        phases = {k: v[ai] for k, v in paths.items()}
        select = phase_choice[ai]
        p0, anchor = before.measure(d, ai, bank, path, r, a, cut)
        pairs = before.model.exchange_menu(a, legal[q], move[q], cap)
        chosen = None
        if len(pairs):
            feat = before.model.features(x[q], legal[q], move[q], a, cap, pairs)[0]
            with torch.inference_mode():
                chosen = before.model.rank(nets['aligned'](matched.exp.design(torch.from_numpy(feat), width, 'aligned')))
        counts.update(views=1, edits=int(a.sum()), candidates=len(pairs), switched=int(chosen is not None))
        view = {k: set() for k in ORACLES}
        closure(view, dict.fromkeys(phase.SHAPES, anchor) | {'prefix_ols': anchor}, anchor,
                dict.fromkeys(ORACLES, False))
        if chosen is None:
            stats = dict(dual=anchor['dual_positive'], risk_pass=anchor['risk_pass'], undefined=anchor['undefined'])
            for arm in ARMS:
                totals[arm].update(stats)
        for j, (src, dst) in enumerate(pairs):
            plan = before.model.exchanged(a, src, dst)
            p, s = before.measure(d, ai, bank, path, r, plan, cut)
            recipient = r[dst, a[dst]]
            if recipient < 0 or not np.array_equal(p[recipient], original[recipient]):
                raise ValueError('Changed exchange recipient')
            variants, score, scalars, stable = {}, {}, {}, {}
            for k in phase.SHAPES:
                v = p.copy()
                v[recipient] = phases[k][recipient]
                z, w = assess(local, cv, ols, v, cut)
                if z['edits'] != s['edits'] or z['unknown_edits'] != s['unknown_edits']:
                    raise ValueError('A timing change altered the edited support')
                variants[k], score[k], scalars[k] = v, z, w
                stable[k] = bool(not anchor['dual_positive'] and z['dual_positive'] and stability.stress(
                    local, cv, ols, p0, v, cut, z, query, len(a), src, dst)[0]['stable_dual'])
                counts[f'{k}_distinct'] += int(not np.array_equal(v, p))
            if score['original'] != s:
                raise ValueError('Original scores did not reproduce')
            selected = phase.SHAPES[select[recipient]]
            score['prefix_ols'], stable['prefix_ols'] = score[selected], stable[selected]
            closure(view, score, anchor, stable)
            if j != chosen:
                continue
            counts[f'prefix_chose_{selected}'] += 1
            variants['prefix_ols'], scalars['prefix_ols'] = variants[selected], scalars[selected]
            for k in ARMS:
                alternatives = phase.SHAPES if k == 'uniform_phase' else (k,)
                totals[k].update({f: sum(scalars[t][f] for t in alternatives)/len(alternatives) for f in FIELDS})
                if k != 'original':
                    gain[k] += sum(float(parent.transfer.paired(p, variants[t], local['y'], local['valid'])[1].sum())
                                   for t in alternatives)/len(alternatives)
        for k, v in view.items():
            oracle[k].update(v)
    mass = np.linalg.norm(bank[0]-d['y'], axis=-1)[d['valid']].sum()
    return dict(counts=dict(counts), arms={k: dict(v) for k, v in totals.items()}, oracle={k: dict(v) for k, v in oracle.items()},
                gain_pp={k: float(100*gain[k]/mass) for k in ARMS if k != 'original'})


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(fn):
        x = np.asarray([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    return dict(counts=dict(sum((Counter(r['counts']) for r in rows), Counter())),
                arms={k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in ARMS},
                oracle={k: dict(sum((Counter(r['oracle'][k]) for r in rows), Counter())) for k in ORACLES},
                paired={k: dict(gain=ci(lambda r: r['gain_pp'][k]), dual=ci(lambda r: 100.*
                    (r['arms'][k]['dual']-r['arms']['original']['dual'])/r['counts']['views']))
                    for k in ARMS if k != 'original'})


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p = OUT/'registration.json'
    files = [Path(__file__), Path(phase.__file__), OUT/'protocol.md', stability.OUT/'readout.json',
             stability.OUT/'registration.json', stability.audit.OUT/'readout.json']
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], shapes=phase.SHAPES,
        no_training=True, external_outcome_access=False, original_2pct=True)))
    groups = {}
    for args, ref in matched.context_cells():
        site, seed = args[:2]
        row = evaluate(*args)
        if row['counts']['switched'] != ref['transitions']['aligned']['exchanged'] or any(
                row['arms']['original'][a] != ref['scores']['aligned'][b]
                for a, b in (('dual','dual_positive'), ('risk_pass','risk_pass'), ('undefined','undefined'))):
            raise ValueError('Original chosen plans did not reproduce')
        groups.setdefault(site, []).append(dict(seed=seed, **row))
        print(site, seed, 'complete', flush=True)
    s = summarize(groups)
    c, o = s['counts'], s['oracle']['original']
    if (c['views'], c['edits'], c['candidates'], c['switched'], o['positive'], o['stable_rescue']) != (
            606, 780, 7198, 101, 181, 93):
        raise ValueError('Changed original exchange or stability frontier')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(p), summary=s, by_site=groups,
        original_2pct=True, external_outcome_access=False, no_training=True, deployment_changed=False)))
    print(s, flush=True)


if __name__ == '__main__':
    main()
