"""Remove the inherited admission mask from the stability comparison."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_exchange_stability_head as fit


run, np, torch, old, matched, risk = fit.run, fit.np, fit.torch, fit.old, fit.matched, fit.risk
OUT = fit.OUT.parent/'opendd_exchange_stability_forced_v1'


def evaluate(args, nets):
    s, seed, motion, d, bank, menu, width, previous, cut = args
    x, legal, rows, path, _ = menu
    move = risk.base.model.movement_prefix(bank[0], path, rows)
    counts, arms = Counter(), {a: Counter() for a in fit.ARMS}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = old.indices(d, rec, rows)
        f = run.read(risk.base.PRIVATE/f'{s}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion, site=s, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound edit count')
        pairs = old.model.exchange_menu(a, legal[q], move[q], cap)
        choices = {}
        if len(pairs):
            z = torch.from_numpy(old.model.features(x[q], legal[q], move[q], a, cap, pairs)[0])
            with torch.inference_mode():
                z = fit.exp.design(z, width, 'aligned')
                prior = previous['aligned'](z)
                b, u = nets['control'](z), nets['stability'](z)
                choices = dict(aligned=matched.matched_choice(prior), control=fit.head.rank(b, False),
                               aux_full=fit.head.rank(u, False), stable=fit.head.rank(u, True))
                counts.update(original_guarded=int(old.model.rank(prior) is not None))
        p0, anchor = old.measure(d, ai, bank, path, r, a, cut)
        counts.update(views=1, edits=int(a.sum()), switched=int(bool(choices)), candidates=len(pairs))
        local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}
        query = np.searchsorted(np.flatnonzero(q), d['query'][ai])
        full, stable = [], []
        for src, dst in pairs:
            p, z = old.measure(d, ai, bank, path, r, old.model.exchanged(a, src, dst), cut)
            full.append(z['dual_positive'])
            stable.append(int(not anchor['dual_positive'] and z['dual_positive'] and fit.stress.stress(
                local, bank[0][ai], bank[1][ai], p0, p, cut, z, query, len(a), src, dst)[0]['stable_dual']))
        for k in fit.ARMS:
            ok, st = anchor['dual_positive'], 0
            if k in choices:
                ok, st = full[choices[k]], stable[choices[k]]
            elif k == 'random' and choices:
                ok, st = float(np.mean(full)), float(np.mean(stable))
            arms[k].update(dual_pass=ok, stable_rescue=st, rescued=ok if not anchor['dual_positive'] else 0,
                           destroyed=1-ok if anchor['dual_positive'] else 0)
    return dict(**counts, arms={k: dict(v) for k, v in arms.items()}, donor_stability_labels=0, donor_stable_positives=0)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p, r, h = (fit.OUT/k for k in ('registration.json', 'readout.json', 'head_freeze.json'))
    report, freeze = run.read(r), run.read(h)
    reg = run.sha(p)
    if (report['registration'] != reg or freeze['registration'] != reg or report['checkpoint_manifest'] != run.sha(h)
            or report['external_outcome_access'] or not report['original_2pct']):
        raise ValueError('Require the immutable completed stability head')
    protocol = OUT/'registration.json'
    run.once(protocol, run.stable(dict(inputs=[run.artifact(f) for f in (Path(__file__), OUT/'protocol.md', r, h)],
        no_training=True, original_2pct=True, external_outcome_access=False)))
    groups = {}
    for args, ref in matched.context_cells():
        s, seed = args[:2]
        nets = {}
        for k in ('control', 'stability'):
            m = freeze['checkpoints'][f'{s}_{seed}_{k}']
            if (m['identity']['held'] != s or s in m['identity']['train'] or m['identity']['seed'] != seed
                    or m['identity']['registration'] != reg or m['identity']['arm'] != k):
                raise ValueError('Stability head has an invalid training role')
            nets[k] = fit.study.load(m, lambda: fit.head.StabilityRanker(4*args[5][0].shape[1]+9), old.STEPS)
        row = evaluate(args, nets)
        if row.get('original_guarded', 0) != ref['transitions']['aligned']['exchanged']:
            raise ValueError('The original outcome-blind guard changed')
        groups.setdefault(s, []).append(dict(seed=seed, **row))
        print(s, seed, 'complete', flush=True)
    result = fit.summarize(groups)
    del result['donor_stability_labels'], result['donor_stable_positives']
    result['coverage'] = {s: {f: sum(r.get(f, 0) for r in g) for f in ('views', 'switched', 'original_guarded', 'candidates')}
                          for s, g in groups.items()}
    if ((result['views'], result['edits'], result['arms']['frozen']['dual_pass']) != (606, 780, 60)
        or sum(r['candidates'] for r in result['coverage'].values()) != 7198
        or sum(r['original_guarded'] for r in result['coverage'].values()) != 101):
        raise ValueError('Forced comparison changed the legal opportunity')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(protocol), summary=result,
        by_site=groups, original_2pct=True, no_training=True, external_outcome_access=False,
        offline_development=True, deployment_changed=False)))
    print(result, flush=True)


if __name__ == '__main__':
    main()
