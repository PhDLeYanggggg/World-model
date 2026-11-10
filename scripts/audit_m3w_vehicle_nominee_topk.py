"""Find the strict capacity of fixed causal shortlists, not their policy performance."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_opportunity as screen

action, ctx, np, torch, base = screen.action, screen.ctx, screen.np, screen.torch, screen.base
sha, write, read = screen.sha, screen.write, screen.read
OUT = screen.OUT.parent/'opendd_vehicle_policy_anchor_v1'
K = (1, 3, 6, 24)


def hit(wins, order):
    if wins.shape != (2, 12) or sorted(order.tolist()) != list(range(24)):
        raise ValueError('Expected an ordering of the full two-recipient menu')
    y = wins.reshape(-1)[order]
    return [int(y[:k].any()) for k in K]


@torch.inference_mode()
def evaluate(p, frozen, d, target, cut, ego):
    cv, ols, y, v, w = p['cv'], p['ols4'], target['future'], target['valid'], target['weight']
    anchor = screen.pair.prediction(p, 'additive')
    names, index, ols0, cap = screen.offsets(cv, ols, target, cut)
    hits, switches = [], []
    for (lo, hi), a, f, name in zip(p['bounds'], p['additive_rows'], frozen['choices'], p['records'], strict=True):
        q = slice(lo, hi)
        k = index[lo]
        if names[k] != name or not np.all(index[q] == k):
            raise ValueError('Query is not bound to the archived recording')
        feats, _ = action.inputs(cv[q], anchor[q], a-lo, d, q)
        s, _ = action.model.merit(ego(torch.from_numpy(feats)).numpy())
        order = np.argsort(-s.reshape(-1), kind='stable')
        if int(order[0]) != 12*int(f[0])+int(f[1])-1:
            raise ValueError('Top one no longer reproduces the frozen nominee')
        wins = screen.oracle.legal_wins(cv[q], ols[q], y[q], v[q], w[q], anchor[q], a-lo, cut, ols0[k], cap[k])
        hits.append(hit(wins, order))
    x = np.asarray(hits)
    for k in (int(frozen['old_mask'].sum()), int(np.ceil(.15*len(hits)))):
        switches.append(np.minimum(x.sum(0), k).tolist())
    return dict(views=len(hits), hits=x.sum(0).tolist(), caps=switches)


def main():
    key, source, _, parent, ego, _ = screen.register()
    f = screen.OUT/'freeze.json'
    frozen = read(f)
    if frozen['registration'] != key or not frozen['before_held_scoring']:
        raise ValueError('Require unchanged causal nominee')
    data = ctx.load_features(source)
    by_site = {s: [] for s in base.SITES}
    for name, c in parent['cells'].items():
        excluded = sorted(c['packets'])
        ref = ego['cells'][name]['checkpoints']['ego']
        net = base.load_net(action.LOCAL/f"{ref['identity']['name']}.pt", ref,
                            action.model.AngleValue(False), excluded)
        for s, r in c['packets'].items():
            with screen.packet_file(ctx.study.LOCAL, r) as p, screen.packet_file(screen.LOCAL, frozen['cells'][name]['packets'][s]) as z:
                result = evaluate(p, z, data[s], action.labels(data[s]['file']), c['cut'], net)
            by_site[s].append(result)

    def summarize(rows):
        return dict(views=sum(r['views'] for r in rows), hits=np.sum([r['hits'] for r in rows], axis=0).tolist(),
                    caps=np.sum([r['caps'] for r in rows], axis=0).tolist())

    all_ = summarize([r for rows in by_site.values() for r in rows])
    if (all_['views'], all_['hits'][0], all_['hits'][-1], all_['caps'][0][0]) != (8874, 420, 1393, 238):
        raise ValueError('The shortlist did not reproduce both registered ceilings')
    d = dict(result_source='fresh_hindsight_diagnostic', top_k=K, quota_names=['ego_count', 'fixed_15pct'],
        summary=all_, by_site={s: summarize(r) for s, r in by_site.items()}, original_2pct=True,
        no_policy_or_threshold_changed=True, external_outcomes_read=False,
        frozen_nominee_sha256=sha(f), code_sha256=sha(Path(__file__)))
    write(OUT/'topk_capacity.json', d, immutable=True)
    print(all_, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    main()
