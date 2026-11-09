"""Train a causal step-wise bridge between CV and OLS4 on Moving development."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_moving_dual_record_forecast as prior
from src.world_model import m3w_reference_bridge as model

old, np, torch, base = prior.old, prior.np, prior.torch, prior.base
OUT = prior.OUT.parent / 'moving_reference_bridge_v1'
PRIVATE = prior.PRIVATE.parent / OUT.name
FRACTIONS = (.02, .10, .35, .70, 1.)
STEPS, BATCH = 800, 256
RAW = (*model.MODES, 'strict', 'v155_ols', 'v155_dual', 'lead_mean_matched', 'causal_support_matched')


def register():
    parent, *_ = prior.register()
    f, r = (prior.OUT / name for name in ('freeze.json', 'readout.json'))
    s, q = map(old.read, (f, r))
    if (s['registration'] != parent or q['freeze'] != old.sha(f) or not s['before_held_scoring']
            or not q['original_2pct'] or any(q[k] for k in ('a_labels_opened', 'confirmation_access', 'independent_confirmation'))):
        raise ValueError('Require frozen development-only parent')
    p = OUT / 'registration.json'
    old.once(p, old.stable(dict(parents=[old.artifact(z) for z in (f, r)],
        code=[old.artifact(z) for z in (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
        steps=STEPS, batch=BATCH, seeds=base.SEEDS, modes=model.MODES, fractions=FRACTIONS,
        threads=4, workers=0, original_2pct=True, a_labels_opened=False, confirmation_access=False)))
    return old.sha(p), s['cells']


def sample(d, inputs, cv, ols):
    y, v = d['target'].astype(np.float32), d['valid']
    w = old.weight(d['query']).astype(np.float32)
    if len(set(d['sites'])) != 3:
        raise ValueError('Require three fitting sources')
    groups = [[np.flatnonzero((d['sites'] == s) & (d['recording'] == r))
               for r in np.unique(d['recording'][d['sites'] == s])] for s in sorted(set(d['sites']))]
    tensors = (*inputs, cv.astype(np.float32), ols.astype(np.float32), y, v, w)

    def draw(rng):
        recs = groups[rng.integers(len(groups))]
        b = rng.choice(recs[rng.integers(len(recs))], size=BATCH, replace=True)
        return tuple(torch.from_numpy(z[b]) for z in tensors)

    return draw, torch.from_numpy(np.average(v, weights=w, axis=0).astype(np.float32))


@torch.inference_mode()
def predict(net, inputs, cv, ols, lead, mode):
    c, o = cv.astype(np.float32), ols.astype(np.float32)
    paths, support = [], []
    for j in range(0, len(cv), 2048):
        sl = slice(j, j+2048)
        x = [torch.from_numpy(z[sl]) for z in (*inputs, c, o)]
        p, logits = net(*x, lead, mode)
        paths.append(p.numpy()-c[sl])
        support.append(logits.sigmoid().numpy())
    p, b = cv+np.concatenate(paths), np.concatenate(support)
    if not (np.isfinite(p).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite predictions')
    return p, b


def shrink_pair(cv, left, right):
    a, b = left-cv, right-cv
    r, s = (np.linalg.norm(z, axis=-1).max(1) for z in (a, b))
    common = np.minimum(r, s)
    pair = tuple(cv + (common / np.maximum(peak, 1e-20))[:, None, None]*z for z, peak in ((a, r), (b, s)))
    for p in pair:
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1).max(1), common, atol=1e-8, rtol=1e-8)
    np.testing.assert_array_equal((pair[0] != cv).any((1, 2)), (pair[1] != cv).any((1, 2)))
    return pair


def screen(d, cv, ols, paths, cut, i):
    parent = prior.previous.parent
    candidates = {f'{k}_{j}': cv + a*(p-cv)
                  for k, p in paths.items() for j, a in enumerate(FRACTIONS)}
    rows = parent.rows_for(d, {base.CV: cv, base.OLS: ols, **candidates}, cv, cut,
                           parent.vs_references(candidates), i)
    merge = prior.prior.prior.parent.readout.merge_comparisons
    by_site = [[r for r in rows if r['source'] == s] for s in sorted(set(d['sites']))]
    table = {}
    for k in candidates:
        risk = sum(z['metrics'][k]['risk']['supported'] for z in rows)
        gain = [float(np.mean([merge([r['comparisons'][f'{k}_vs_{b}'] for r in site])['completion_lower_pp']
                               for site in by_site])) for b in ('cv', 'ols4')]
        table[k] = dict(risk=risk, views=len(rows), gains=gain)
    allowed = [k for k, v in table.items() if v['risk'] == v['views'] and min(v['gains']) > 0.]
    return table, max(allowed, key=lambda k: min(table[k]['gains'])) if allowed else None


def fit_cell(d, inputs, linear, parent, reg, replay, beat):
    i, cut = parent['identity'], parent['cut']
    roles = [set(i[k]) for k in ('train', 'calibration', 'held')]
    if any(len(s) != 3 for s in roles) or len(set.union(*roles)) != 9:
        raise ValueError('Source roles changed')
    (it, train), (ic, cal), (ih, held) = [base.subset(d, i[k]) for k in ('train', 'calibration', 'held')]
    cv, ols = linear.transpose(1, 0, 2, 3)
    xt, xc, xh = ([a[z] for a in inputs] for z in (it, ic, ih))
    draw, lead = sample(train, xt, cv[it], ols[it])
    fits, ca, he, prob, n_updates, matched_rng = {}, {}, {}, {}, 0, None
    for mode in model.MODES:
        def cost(net, history, peers, mask, frame, scale, velocity, c, o, y, v, w):
            p, log = net(history, peers, mask, frame, scale, velocity, c, o, lead, mode)
            return model.loss(p, log, c, o, y, v, w, cut[0])

        tag = f'{mode}_{i["fold"]}_{i["seed"]}'
        net, receipt, n = prior.prior.prefix.trainer.fit(dict(i, registration=reg, arm=mode), tag, i['seed'],
            model.Forecast, STEPS, draw, cost, replay, beat)
        state = torch.load(old.checked(receipt['artifact']), map_location='cpu', weights_only=True)['rng']
        if matched_rng is not None and state != matched_rng:
            raise ValueError('Unmatched training draws')
        matched_rng, fits[mode] = state, receipt
        n_updates += n
        ca[mode], _ = predict(net, xc, cv[ic], ols[ic], lead, mode)
        he[mode], prob[mode] = predict(net, xh, cv[ih], ols[ih], lead, mode)
    table, chosen = screen(cal, cv[ic], ols[ic], dict(ca, ols4=ols[ic]), cut, i)
    if chosen is None:
        he['strict'] = cv[ih]
    else:
        k, j = chosen.rsplit('_', 1)
        he['strict'] = cv[ih] + FRACTIONS[int(j)]*((ols[ih] if k == 'ols4' else he[k])-cv[ih])
    he['lead_mean_matched'], he['causal_support_matched'] = shrink_pair(cv[ih], he['lead_mean'], he['causal_support'])
    with np.load(old.checked(parent['paths']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], held['ids'])
        he.update(v155_ols=z['ols_only'], v155_dual=z['dual_risk'])
    artifact = prior.prior.study.parent.parent.save_actions(PRIVATE / f"paths_{i['fold']}_{i['seed']}.npz",
        dict(ids=held['ids'], lead=lead.numpy(), **he, **{f'prob_{k}': b for k,b in prob.items()}))
    return dict(identity=i, cut=cut, fits=fits, selected=chosen, calibration=table, paths=artifact), n_updates


def evaluate(d, linear, cells):
    parent, stats = prior.previous.parent, prior.prior.prefix.grouped
    pairs = parent.vs_references(RAW)
    pairs.update(support_vs_mean=('lead_mean','causal_support'),
                 support_vs_mean_matched=('lead_mean_matched','causal_support_matched'))
    rows, aux, absolute, peaks = [], [], {}, {}
    for c in cells:
        i = c['identity']
        ix, h = base.subset(d, i['held'])
        cv, ols = linear[ix].transpose(1, 0, 2, 3)
        with np.load(old.checked(c['paths']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            paths = {base.CV:cv, base.OLS:ols, **{k:z[k] for k in RAW}}
            v = h['valid']
            brier = {'mean':np.square(v-z['lead']).mean(1), **{k:np.square(v-z[f'prob_{k}']).mean(1) for k in model.MODES}}
        rows.extend(parent.rows_for(h, paths, cv, c['cut'], pairs, i))
        for s in sorted(set(h['sites'])):
            at = h['sites'] == s
            sub = {k:h[k][at] for k in ('target','valid','query')}
            w = old.weight(sub['query'])
            absolute[str(s),i['seed']] = {k:prior.previous.absolute_masses(sub,p[at]) for k,p in paths.items()}
            peaks[str(s),i['seed']] = {k:float(w @ np.linalg.norm(p[at]-cv[at],axis=-1).max(1) / w.sum()) for k,p in paths.items()}
            aux.append(dict(source=str(s), seed=i['seed'], **{k:float(w @ a[at]/w.sum()) for k,a in brier.items()}))
    roster = {r['recording']:r['source'] for r in rows}
    if len(roster)!=24 or len(rows)!=72 or len(set(roster.values()))!=9:
        raise ValueError('Held population changed')
    summary = stats.grouped_readout(rows,roster,base.SEEDS)
    dual = {k:sum(r['metrics'][k]['risk']['supported'] and all(
        r['comparisons'][f'{k}_vs_{b}']['known_delta'] > r['comparisons'][f'{k}_vs_{b}']['unknown_radius']
        for b in ('cv','ols4')) for r in rows) for k in RAW}
    forecast = {}
    for k in (*RAW,base.CV,base.OLS):
        a = np.array([x[k] for x in absolute.values()])
        forecast[k] = dict(ade_pixel=float((a[:,0]/a[:,1]).mean()), fde_pixel=float((a[:,2]/a[:,3]).mean()),
                           mean_peak_pixel=float(np.mean([x[k] for x in peaks.values()])))
    draws = np.random.default_rng(61877).integers(9,size=(3000,9))
    sites = sorted(set(roster.values()))
    brier = {k:float(np.mean([r[k] for r in aux])) for k in ('mean',*model.MODES)}
    delta = np.array([np.mean([r['causal_support']-r['mean'] for r in aux if r['source']==s]) for s in sites])
    return dict(summary=summary, dual=dual, forecast=forecast, brier=brier, brier_support_minus_mean=dict(
        point=float(delta.mean()), interval95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()),
        strict_selected=sum(c['selected'] is not None for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    replay = p.parse_args().replay
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    prior.prior.prefix.trainer.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / 'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**kwargs):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start,3), replay=replay, **kwargs)
            prior.prior.donor.prior.prior.source.atomic(PRIVATE / 'heartbeat.json', z)
            if kwargs.get('phase')!='fit':
                print(z, flush=True)
        reg, parents = register()
        d = base.fit_population(beat)
        inputs = prior.previous.model.causal_inputs(d['history'], prior.prior.query_groups(d['query']))
        linear = prior.prior.study.motion.bank(d['history'])[:, :2]
        cells, n = [], 0
        for parent in parents:
            cell, updates = fit_cell(d, inputs, linear, parent, reg, replay, beat)
            cells.append(cell)
            n += updates
            beat(phase='cell_frozen', fold=cell['identity']['fold'], seed=cell['identity']['seed'],
                 selected=cell['selected'], updates=n)
        f = OUT / 'freeze.json'
        old.once(f, old.stable(dict(registration=reg, cells=cells, before_held_scoring=True,
            configured_updates=len(cells)*STEPS*len(model.MODES))))
        r = evaluate(d, linear, old.read(f)['cells'])
        old.once(OUT / 'readout.json', old.stable(dict(freeze=old.sha(f), **r, original_2pct=True,
            a_labels_opened=False, confirmation_access=False, independent_confirmation=False, deployment_changed=False)))
        beat(phase='complete', updates=n, strict_selected=r['strict_selected'])


if __name__ == '__main__':
    main()
