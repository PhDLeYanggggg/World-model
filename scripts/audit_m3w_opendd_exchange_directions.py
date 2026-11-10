"""Test causal direction proposals at identical per-step correction magnitudes."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_phase as prior
from src.world_model import m3w_exchange_direction_bank as model


run, old, base, np, torch, matched = prior.run, prior.before, prior.base, prior.np, prior.torch, prior.matched
OUT = prior.OUT.parent/'opendd_exchange_directions_v1'
ARMS = (*model.NAMES, 'suffix', 'unanimous', 'uniform')
MENUS = (*model.NAMES, 'suffix', 'unanimous', 'expanded')
FLOORS = {'positive': 1e-6, 'gain_gt_0_10pp': .10, 'gain_gt_0_50pp': .50}


def metric_rows(p, cv, ols, local, cut):
    s, fields = prior.assess(local, cv, ols, p, cut)
    err = np.linalg.norm(p-local['y'], axis=-1)
    v = local['valid']
    fields.update(step_error=float(err[v].sum()), endpoint_error=float(err[v[:, -1], -1].sum()))
    return s, fields


def evaluate(site, seed, motion, d, bank, menu, width, nets, cut):
    x, legal, rows, path, _ = menu
    cv0, ols0 = bank[:2]
    paths = model.direction_bank(d['h'], cv0, path)
    a0 = np.linalg.norm(path-cv0, axis=-1)
    a1 = np.linalg.norm(paths-cv0[:, None], axis=-1)
    np.testing.assert_allclose(a1, np.broadcast_to(a0[:, None], a1.shape), atol=1e-11, rtol=1e-11)
    fraction = model.bridge_fraction(cv0, ols0, path)
    p0, p1 = model.select(model.suffix_scores(d['h'], fraction))
    move = base.model.movement_prefix(cv0, path, rows)
    count, delta = Counter(), Counter()
    arms, oracle = {k: Counter() for k in ARMS}, {k: Counter() for k in MENUS}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = old.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound original edit budget')
        cv, ols = cv0[ai], ols0[ai]
        local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}
        query = np.searchsorted(np.flatnonzero(q), d['query'][ai])
        p_anchor, anchor = old.measure(d, ai, bank, path, r, a, cut)
        pairs = old.model.exchange_menu(a, legal[q], move[q], cap)
        choice = None
        if len(pairs):
            z = old.model.features(x[q], legal[q], move[q], a, cap, pairs)[0]
            with torch.inference_mode():
                choice = old.model.rank(nets['aligned'](matched.exp.design(torch.from_numpy(z), width, 'aligned')))
        count.update(views=1, edits=int(a.sum()), candidates=len(pairs), switched=int(choice is not None),
                     known_steps=int(local['valid'].sum()), endpoints=int(local['valid'][:, -1].sum()))
        floor = min(anchor['vs_cv'], anchor['vs_ols4']) if anchor['risk_pass'] else -np.inf
        best = dict.fromkeys(MENUS, floor)
        stable = dict.fromkeys(MENUS, False)
        if choice is None:
            _, scores = metric_rows(p_anchor, cv, ols, local, cut)
            for k in arms:
                arms[k].update(scores)
        for j, (src, dst) in enumerate(pairs):
            p, s = old.measure(d, ai, bank, path, r, old.model.exchanged(a, src, dst), cut)
            row = r[dst, a[dst]]
            if row < 0:
                raise ValueError('Illegal receiver')
            g = np.flatnonzero(ai)[row]
            variants, scores, full, survive = {}, {}, {}, {}
            for k, line in zip(model.NAMES, paths[g], strict=True):
                v = p.copy()
                v[row] = line
                raw, slim = metric_rows(v, cv, ols, local, cut)
                if raw['edits'] != s['edits'] or raw['unknown_edits'] != s['unknown_edits']:
                    raise ValueError('Direction edit changed selected support')
                full[k] = min(raw['vs_cv'], raw['vs_ols4']) if raw['risk_pass'] else -np.inf
                survive[k] = bool(not anchor['dual_positive'] and raw['dual_positive'] and prior.stability.stress(
                    local, cv, ols, p_anchor, v, cut, raw, query, len(a), src, dst)[0]['stable_dual'])
                variants[k], scores[k] = v, slim
                if k == 'original' and raw != s:
                    raise ValueError('Original trajectory scores did not reproduce')
                if k != 'original':
                    count[f'{k}_distinct'] += int(not np.array_equal(p[row], line))
            for k, family in (('suffix', model.NAMES[p0[g]]), ('unanimous', model.NAMES[p1[g]])):
                full[k], survive[k] = full[family], survive[family]
                variants[k], scores[k] = variants[family], scores[family]
                if j == choice:
                    count[f'{k}_chose_{family}'] += 1
            full['expanded'], survive['expanded'] = max(full.values()), any(survive.values())
            for k in MENUS:
                best[k] = max(best[k], full[k])
                stable[k] |= survive[k]
            if j == choice:
                for k in ARMS:
                    z = model.NAMES if k == 'uniform' else (k,)
                    arms[k].update({f: sum(scores[a][f] for a in z)/len(z) for f in scores['original']})
                    delta[k] += sum(float(prior.parent.transfer.paired(
                        p, variants[a], local['y'], local['valid'])[1].sum()) for a in z)/len(z)
        for k in MENUS:
            oracle[k].update({f: int(best[k] > t) for f, t in FLOORS.items()})
            oracle[k].update(stable_rescue=int(stable[k]))
    mass = np.linalg.norm(cv0-d['y'], axis=-1)[d['valid']].sum()
    return dict(counts=dict(count), arms={k: dict(v) for k, v in arms.items()}, oracle={k: dict(v) for k, v in oracle.items()},
                gain_pp={k: float(100*delta[k]/mass) for k in ARMS if k != 'original'})


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(fn):
        x = np.asarray([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    c = dict(sum((Counter(r['counts']) for r in rows), Counter()))
    arms = {k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in ARMS}
    for a in arms.values():
        a.update(available_step_mean_error=a.pop('step_error')/c['known_steps'],
                 available_endpoint_mean_error=a.pop('endpoint_error')/c['endpoints'])
    return dict(counts=c, arms=arms,
                oracle={k: dict(sum((Counter(r['oracle'][k]) for r in rows), Counter())) for k in MENUS},
                paired={k: dict(gain=ci(lambda r: r['gain_pp'][k]), dual=ci(lambda r: 100.*
                    (r['arms'][k]['dual']-r['arms']['original']['dual'])/r['counts']['views'])) for k in ARMS if k != 'original'})


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    source = prior.OUT/'readout.json'
    p = OUT/'registration.json'
    files = [Path(__file__), Path(model.__file__), OUT/'protocol.md', source, prior.OUT/'registration.json']
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], directions=model.NAMES,
        no_training=True, external_outcome_access=False, original_2pct=True)))
    groups = {}
    for args, ref in matched.context_cells():
        site, seed = args[:2]
        row = evaluate(*args)
        if row['counts']['switched'] != ref['transitions']['aligned']['exchanged'] or any(
                row['arms']['original'][a] != ref['scores']['aligned'][b]
                for a, b in (('dual','dual_positive'), ('risk_pass','risk_pass'), ('undefined','undefined'))):
            raise ValueError('Frozen chosen plans did not reproduce')
        groups.setdefault(site, []).append(dict(seed=seed, **row))
        print(site, seed, 'complete', flush=True)
    s = summarize(groups)
    prev = run.read(source)['summary']
    if (s['oracle']['original'] != prev['oracle']['original'] or any(
            s['counts'][k] != prev['counts'][k] for k in ('views', 'edits', 'candidates', 'switched'))):
        raise ValueError('Changed full original frontier')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(p), summary=s, by_site=groups,
        original_2pct=True, external_outcome_access=False, no_training=True, deployment_changed=False)))
    print(s, flush=True)


if __name__ == '__main__':
    main()
