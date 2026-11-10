"""Separate displacement timing from the total amount of deviation."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_phase as ref
from src.world_model import m3w_exchange_matched_amplitude as amplitude


run, before, base, np, torch, matched = ref.run, ref.before, ref.base, ref.np, ref.torch, ref.matched
OUT = ref.OUT.parent/'opendd_exchange_matched_amplitude_v1'


def evaluate(site, seed, motion, d, bank, menu, width, nets, cut):
    x, legal, rows, path, _ = menu
    paths, moved = amplitude.matched_amplitude_paths(bank[0], path)
    a0 = np.linalg.norm(path-bank[0], axis=-1)
    for p in paths.values():
        a1 = np.linalg.norm(p-bank[0], axis=-1)
        np.testing.assert_allclose(a1.max(1), a0.max(1), atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(a1.sum(1), a0.sum(1), atol=1e-10, rtol=1e-11)
        np.testing.assert_array_equal(p[:, -1], path[:, -1])
    pick = ref.phase.prefix_choices(paths, bank[1])
    move = base.model.movement_prefix(bank[0], path, rows)
    counts, gain = Counter(), Counter()
    arms, oracles = {k: Counter() for k in ref.ARMS}, {k: Counter() for k in ref.ORACLES}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = before.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound frozen count or cap')
        local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}
        cv, ols = bank[0][ai], bank[1][ai]
        p0, s0 = before.measure(d, ai, bank, path, r, a, cut)
        query = np.searchsorted(np.flatnonzero(q), d['query'][ai])
        pairs = before.model.exchange_menu(a, legal[q], move[q], cap)
        chosen = None
        if len(pairs):
            feat = before.model.features(x[q], legal[q], move[q], a, cap, pairs)[0]
            with torch.inference_mode():
                chosen = before.model.rank(nets['aligned'](matched.exp.design(torch.from_numpy(feat), width, 'aligned')))
        counts.update(views=1, edits=int(a.sum()), candidates=len(pairs), switched=int(chosen is not None))
        view = {k: set() for k in ref.ORACLES}
        ref.closure(view, dict.fromkeys(ref.ORACLES, s0), s0, dict.fromkeys(ref.ORACLES, False))
        if chosen is None:
            for k in arms:
                arms[k].update(dual=s0['dual_positive'], risk_pass=s0['risk_pass'], undefined=s0['undefined'])
        for j, (src, dst) in enumerate(pairs):
            p, s = before.measure(d, ai, bank, path, r, before.model.exchanged(a, src, dst), cut)
            row = r[dst, a[dst]]
            if row < 0:
                raise ValueError('Illegal recipient')
            source = np.flatnonzero(ai)[row]
            sel = ref.phase.SHAPES[pick[source]]
            score, slim, st, mass = {}, {}, {}, {}
            for k in ref.phase.SHAPES:
                v = p.copy()
                v[row] = paths[k][source]
                z, scalars = ref.assess(local, cv, ols, v, cut)
                if z['edits'] != s['edits'] or z['unknown_edits'] != s['unknown_edits']:
                    raise ValueError('Matched amplitude altered edited support')
                score[k], slim[k] = z, scalars
                st[k] = bool(not s0['dual_positive'] and z['dual_positive'] and ref.stability.stress(
                    local, cv, ols, p0, v, cut, z, query, len(a), src, dst)[0]['stable_dual'])
                if k != 'original':
                    counts[f'{k}_distinct'] += int(moved[k][source] > 0.)
                if j == chosen:
                    mass[k] = float(ref.parent.transfer.paired(p, v, local['y'], local['valid'])[1].sum())
            if score['original'] != s:
                raise ValueError('Original score did not reproduce')
            score['prefix_ols'], st['prefix_ols'] = score[sel], st[sel]
            ref.closure(view, score, s0, st)
            if j == chosen:
                counts[f'prefix_chose_{sel}'] += 1
                slim['prefix_ols'], mass['prefix_ols'] = slim[sel], mass[sel]
                for k in arms:
                    ks = ref.phase.SHAPES if k == 'uniform_phase' else (k,)
                    arms[k].update({f: sum(slim[t][f] for t in ks)/len(ks) for f in ref.FIELDS})
                    gain[k] += sum(mass[t] for t in ks)/len(ks)
        for k, v in view.items():
            oracles[k].update(v)
    mass = np.linalg.norm(bank[0]-d['y'], axis=-1)[d['valid']].sum()
    return dict(counts=dict(counts), arms={k: dict(v) for k, v in arms.items()},
                oracle={k: dict(v) for k, v in oracles.items()},
                gain_pp={k: float(100*gain[k]/mass) for k in arms if k != 'original'})


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p = OUT/'registration.json'
    source = ref.OUT/'readout.json'
    files = [Path(__file__), Path(amplitude.__file__), OUT/'protocol.md', source, ref.OUT/'registration.json']
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], original_2pct=True,
        no_training=True, external_outcome_access=False, preserves_cumulative_displacement=True)))
    groups = {}
    for args, expected in matched.context_cells():
        site, seed = args[:2]
        row = evaluate(*args)
        if row['counts']['switched'] != expected['transitions']['aligned']['exchanged'] or any(
                row['arms']['original'][a] != expected['scores']['aligned'][b]
                for a, b in (('dual', 'dual_positive'), ('risk_pass', 'risk_pass'), ('undefined', 'undefined'))):
            raise ValueError('Changed original score or exchange mask')
        groups.setdefault(site, []).append(dict(seed=seed, **row))
        print(site, seed, 'complete', flush=True)
    s = ref.summarize(groups)
    prev = run.read(source)['summary']
    if (s['oracle']['original'] != prev['oracle']['original'] or s['arms']['original'] != prev['arms']['original']
            or any(s['counts'][k] != prev['counts'][k] for k in ('views', 'edits', 'candidates', 'switched'))):
        raise ValueError('Changed original full-roster comparison')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(p), summary=s, by_site=groups,
        original_2pct=True, external_outcome_access=False, no_training=True, deployment_changed=False)))
    print(s, flush=True)


if __name__ == '__main__':
    main()
