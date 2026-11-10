"""Compare context rankings when every arm exchanges on the same views."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_exchange_context as exp


run, before, model, base, parent, prefix, np, torch = (exp.run, exp.before, exp.model, exp.base, exp.parent,
    exp.prefix, exp.np, exp.torch)
CONTRASTS = ('motion', 'misaligned', 'random')
ARMS = (*exp.context.MODES, 'random')


def matched_choice(scores):
    i = model.rank(scores, margin=-np.inf)
    return i if i is not None else model.rank(scores, margin=-np.inf, guard=False)


def evaluate(site, seed, motion_key, d, bank, menu, width, nets, cut):
    x, legal, rows, path, _ = menu
    move = base.model.movement_prefix(bank[0], path, rows)
    summary, counts = {k: Counter() for k in ARMS}, Counter()
    effect = Counter({k: 0. for k in CONTRASTS})
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = before.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        if f['identity'] != dict(registration=motion_key, site=site, seed=seed, ordinal=i):
            raise ValueError('Matched swap mask uses an unbound view')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if a.sum() != f['edits']:
            raise ValueError('Frozen count changed')
        pairs = model.exchange_menu(a, legal[q], move[q], cap)
        chosen = {}
        if len(pairs):
            feat = torch.from_numpy(model.features(x[q], legal[q], move[q], a, cap, pairs)[0])
            with torch.inference_mode():
                aligned = model.rank(nets['aligned'](exp.design(feat, width, 'aligned')))
                if aligned is not None:
                    chosen['aligned'] = aligned
                    for k in exp.context.MODES:
                        if k != 'aligned':
                            chosen[k] = matched_choice(nets[k](exp.design(feat, width, k)))
        _, anchor = before.measure(d, ai, bank, path, r, a, cut)
        counts.update(views=1, edits=int(a.sum()), exchanged=int(bool(chosen)))
        if not chosen:
            for k in ARMS:
                summary[k].update({f: anchor[f] for f in ('dual_positive', 'risk_pass', 'undefined')})
            continue
        y, valid = d['y'][ai], d['valid'][ai]
        result = [before.measure(d, ai, bank, path, r, model.exchanged(a, src, dst), cut) for src, dst in pairs]
        p = result[chosen['aligned']][0]
        for k in ARMS:
            if k == 'random':
                score = {f: float(np.mean([v[1][f] for v in result])) for f in ('dual_positive', 'risk_pass', 'undefined')}
            else:
                score = {f: result[chosen[k]][1][f] for f in ('dual_positive', 'risk_pass', 'undefined')}
            summary[k].update(score)
        for k in CONTRASTS:
            available = result if k == 'random' else [result[chosen[k]]]
            effect[k] += float(np.mean([parent.transfer.paired(path0, p, y, valid)[1].sum() for path0, _ in available]))
    mass = np.linalg.norm(bank[0]-d['y'], axis=-1)[d['valid']].sum()
    return dict(**counts, arms={k: dict(v) for k, v in summary.items()}, effect={k: float(100.*v/mass) for k, v in effect.items()})


def summarize(groups):
    flat = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    return dict(**{k: sum(r[k] for r in flat) for k in ('views', 'edits', 'exchanged')}, arms={k: {
        f: sum(r['arms'][k][f] for r in flat) for f in ('dual_positive', 'risk_pass', 'undefined')} for k in ARMS},
        aligned_minus={k: dict(gain_pp=ci(lambda r: r['effect'][k]), dual_pp=ci(lambda r: 100.*(
            r['arms']['aligned']['dual_positive']-r['arms'][k]['dual_positive'])/r['views'])) for k in CONTRASTS})


def context_cells():
    reg, _, _, motion_key, cells, ranks, _, _, _, _, assets = exp.register()
    p = exp.OUT/'readout.json'
    score, fit = run.read(p), run.read(exp.OUT/'head_freeze.json')
    if (fit['registration'] != reg or score['registration'] != reg or score['external_outcome_access']
            or score['checkpoint_manifest'] != run.sha(exp.OUT/'head_freeze.json') or not score['original_2pct']):
        raise ValueError('Require the completed and verified context fit')
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        path = parent.prior.prior.old.source.prior.PRIVATE/f'{s}_cohort.npz'
        with np.load(path, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
        bank = parent.prior.analytic(d)
        raster = exp.road.raster_for(s, assets[s])[0]
        cut = np.quantile(np.concatenate([v['cut_samples'] for t, v in sites.items() if t != s]), [.25, .75])
        for cell in cells['by_site'][s]:
            seed = cell['seed']
            rank = exp.study.load(ranks['heads'][f'{s}/{seed}/additive_rank'],
                                  lambda: exp.study.prior.model.SwapValue(False), exp.study.prior.STEPS)
            base_menu = prefix.menu_for(d, bank, rank)
            menu, _ = exp.augment(d, bank, base_menu, raster)
            nets = {}
            for k in exp.context.MODES:
                m = fit['checkpoints'][f'{s}_{seed}_{k}']
                if m['identity']['held'] != s or s in m['identity']['train']:
                    raise ValueError('A score used held labels')
                nets[k] = exp.study.load(m, lambda: model.ExchangeRanker(4*menu[0].shape[1]+9), before.STEPS)
            ref = next(r for r in score['by_site'][s] if r['seed'] == seed)
            yield (s, seed, motion_key, d, bank, menu, base_menu[0].shape[1], nets, cut), ref


def audit():
    p = exp.OUT/'readout.json'
    groups = {}
    for inputs, ref in context_cells():
        s, seed = inputs[:2]
        row = evaluate(*inputs)
        if (row['exchanged'] != ref['transitions']['aligned']['exchanged'] or any(
                row['arms']['aligned'][f] != ref['scores']['aligned'][f] for f in ('dual_positive', 'risk_pass', 'undefined'))):
            raise ValueError('The pre-scoring aligned mask or predictions changed')
        groups.setdefault(s, []).append(dict(seed=seed, **row))
    out = dict(rank_readout_sha256=run.sha(p), summary=summarize(groups), by_site=groups, original_2pct=True,
               external_outcome_access=False, offline_matching=True, no_training=True,
               common_intervention_mask='guarded_aligned_history_features')
    run.once(exp.OUT/'matched_swap_readout.json', run.stable(out))
    print(out['summary'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    audit()
