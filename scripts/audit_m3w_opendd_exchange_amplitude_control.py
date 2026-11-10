"""Compare the late timing result to a pure amplitude control."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_phase as ref
from src.world_model import m3w_exchange_amplitude_control as control


run, old, base, np, torch = ref.run, ref.before, ref.base, ref.np, ref.torch
OUT = ref.OUT.parent/'opendd_exchange_amplitude_control_v1'
CONTRASTS = (('late', 'original'), ('amplitude', 'original'), ('late', 'amplitude'))


def evaluate(site, seed, motion, d, bank, menu, width, nets, cut):
    x, legal, rows, path, _ = menu
    cv0 = bank[0]
    late = ref.phase.phase_paths(cv0, path)['late']
    amplitude, possible = control.amplitude_control(cv0, path, late)
    a = np.linalg.norm(amplitude-cv0, axis=-1)
    np.testing.assert_allclose(a.max(1), np.linalg.norm(path-cv0, axis=-1).max(1), atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(a.sum(1)[possible], np.linalg.norm(late-cv0, axis=-1).sum(1)[possible], atol=1e-10, rtol=1e-11)
    np.testing.assert_array_equal(amplitude[:, -1], path[:, -1])
    movement = base.model.movement_prefix(cv0, path, rows)
    counts, gains = Counter(), Counter()
    arms = {k: Counter() for k in ('original', 'late', 'amplitude')}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = old.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        plan, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or plan.sum() != f['edits']:
            raise ValueError('Unbound frozen count or cap')
        pairs = old.model.exchange_menu(plan, legal[q], movement[q], cap)
        chosen = None
        if len(pairs):
            z = old.model.features(x[q], legal[q], movement[q], plan, cap, pairs)[0]
            with torch.inference_mode():
                chosen = old.model.rank(nets['aligned'](ref.matched.exp.design(torch.from_numpy(z), width, 'aligned')))
        counts.update(views=1, edits=int(plan.sum()), switched=int(chosen is not None))
        if chosen is None:
            s = old.measure(d, ai, bank, path, r, plan, cut)[1]
            for k in arms:
                arms[k].update(dual=s['dual_positive'], risk_pass=s['risk_pass'], undefined=s['undefined'])
            continue
        src, dst = pairs[chosen]
        row = r[dst, plan[dst]]
        if row < 0:
            raise ValueError('Illegal recipient')
        g = np.flatnonzero(ai)[row]
        if not possible[g]:
            raise ValueError('The matched-amplitude comparison is not identifiable on this mask')
        counts.update(matched=1)
        p0, s = old.measure(d, ai, bank, path, r, old.model.exchanged(plan, src, dst), cut)
        local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}
        paths = dict(original=p0, late=p0.copy(), amplitude=p0.copy())
        paths['late'][row], paths['amplitude'][row] = late[g], amplitude[g]
        for k, v in paths.items():
            z, scalars = ref.assess(local, cv0[ai], bank[1][ai], v, cut)
            if z['edits'] != s['edits'] or z['unknown_edits'] != s['unknown_edits']:
                raise ValueError('Matching amplitude changed the edited support')
            arms[k].update(scalars)
        for left, right in CONTRASTS:
            gains[f'{left}_minus_{right}'] += float(ref.parent.transfer.paired(
                paths[right], paths[left], local['y'], local['valid'])[1].sum())
    mass = np.linalg.norm(cv0-d['y'], axis=-1)[d['valid']].sum()
    return dict(counts=dict(counts), arms={k: dict(v) for k, v in arms.items()},
                gain_pp={f'{a}_minus_{b}': float(100*gains[f'{a}_minus_{b}']/mass) for a, b in CONTRASTS})


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    return dict(counts=dict(sum((Counter(r['counts']) for r in rows), Counter())),
                arms={k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in rows[0]['arms']},
                paired={f'{a}_minus_{b}': dict(gain=ci(lambda r: r['gain_pp'][f'{a}_minus_{b}']),
                    dual=ci(lambda r: 100.*(r['arms'][a]['dual']-r['arms'][b]['dual'])/r['counts']['views']))
                    for a, b in CONTRASTS})


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    source = ref.OUT/'readout.json'
    p = OUT/'registration.json'
    files = [Path(__file__), Path(control.__file__), OUT/'protocol.md', source, ref.OUT/'registration.json']
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], original_2pct=True,
        no_training=True, external_outcome_access=False, no_threshold_changes=True)))
    groups = {}
    for args, refrow in ref.matched.context_cells():
        site, seed = args[:2]
        v = evaluate(*args)
        if v['counts']['switched'] != refrow['transitions']['aligned']['exchanged']:
            raise ValueError('Changed frozen switch mask')
        groups.setdefault(site, []).append(dict(seed=seed, **v))
    s = summarize(groups)
    prev = run.read(source)['summary']
    if (any(s['arms'][k] != prev['arms'][k] for k in ('original', 'late')) or s['counts']['matched'] != 101
            or not np.isclose(s['paired']['late_minus_original']['gain']['point_pp'],
                              prev['paired']['late']['gain']['point_pp'], atol=1e-13, rtol=0.)):
        raise ValueError('The original timing result did not reproduce')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(p), summary=s, by_site=groups,
        original_2pct=True, external_outcome_access=False, no_training=True, deployment_changed=False)))
    print(s, flush=True)


if __name__ == '__main__':
    main()
