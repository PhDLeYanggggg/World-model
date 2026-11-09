"""Train matched recipient policies from exact risk-aware and gain-only teachers."""
import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_stream_spline_calibration as prior
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_recipient_slot as slot
from src.world_model import m3w_feasible_recipient_imitation as model

run, np, torch, study, m = prior.run, prior.np, prior.torch, prior.study, prior.m
control, v = prior.control, prior.v
OUT = prior.OUT.parent/'opendd_feasible_recipient_imitation_v1'
PRIVATE = prior.PRIVATE.parent/OUT.name
ARMS = ('gain_teacher', 'risk_teacher')
STEPS = 600


def register():
    pkey, source, forecasts, _, old, _, pair_ranks, _ = prior.register()
    f, r = [prior.OUT/k for k in ('freeze.json', 'readout.json')]
    stream, readout = map(run.read, (f, r))
    if (stream['registration'] != pkey or readout['freeze'] != run.sha(f) or not stream['before_held_scoring']
            or readout['confirmation_access'] or readout['independent_confirmation'] or not readout['original_2pct']):
        raise ValueError('Require bound streaming development results')
    p = OUT/'registration.json'
    run.once(p, run.stable(dict(parents=[run.artifact(k) for k in (f, r)], steps=STEPS, seeds=m.SEEDS,
        code=[run.artifact(k) for k in (Path(__file__), Path(model.__file__), Path(slot.__file__), OUT/'protocol.md')],
        threads=4, workers=0, original_2pct=True, confirmation_access=False, held_tuning=False)))
    return run.sha(p), source, forecasts, old, stream, pair_ranks


def design(d, bank, original, spline):
    cv, ols = bank[:2]
    peak = control.radius.peak
    native = peak(ols-cv)
    old_rows = np.flatnonzero((original != cv).any((1, 2)))
    query = d['query']
    first, last = np.searchsorted(query, query[old_rows], side='left'), np.searchsorted(query, query[old_rows], side='right')
    rows, slots, amps, size, dist = [], [], [], [], []
    for i, row in enumerate(old_rows):
        roster = np.arange(first[i], last[i])
        radius = peak(original[row]-cv[row])
        cand = roster[(native[roster] > 0.) & (native[roster]+1e-10 >= radius)]
        rows.extend(cand)
        slots.extend([i]*len(cand))
        amps.extend((radius/native[cand]).clip(max=1.))
        size.extend([len(roster)]*len(cand))
        dist.extend(np.linalg.norm(d['h'][cand, -1]-d['h'][row, -1], axis=1))
    rows, slots, amps = np.asarray(rows, int), np.asarray(slots, int), np.asarray(amps)
    if not len(rows):
        return dict(rows=rows, slots=slots, anchors=np.empty(0, int), query=query[old_rows], old_rows=old_rows)
    x, c, _, analytic, scale = slot.features(d['h'][rows], amps)
    np.testing.assert_allclose(c, cv[rows], atol=1e-10, rtol=0.)
    is_anchor = rows == old_rows[slots]
    np.testing.assert_allclose(analytic[is_anchor], original[rows[is_anchor]], atol=1e-9, rtol=0.)
    analytic[is_anchor] = original[rows[is_anchor]]
    alt = control.radius.match_radius(c, analytic, (spline[rows]-c)[:, None])[:, 0]
    _, rot, _ = control.radius.motion_frame(d['h'][rows])
    norm = lambda p: (((p-c) @ rot)/scale[:, None, None])[:, [2,5,8,11]].reshape(len(c), -1)
    x = np.c_[x, np.log1p(size), np.log1p(np.asarray(dist)/scale), is_anchor]
    features = np.concatenate([np.c_[x, norm(p), np.full(len(x), k)] for k,p in enumerate((analytic,alt))])
    anchors = np.flatnonzero(is_anchor)
    np.testing.assert_array_equal(slots[anchors], np.arange(len(old_rows)))
    return dict(x=features.clip(-16.,16.).astype(np.float32), rows=np.tile(rows, 2), slots=np.tile(slots, 2),
        paths=np.concatenate((analytic,alt)), anchors=anchors, query=query[old_rows], old_rows=old_rows,
        analytic_edges=len(rows))


def install(cv, p, selected):
    q = cv.copy()
    if len(selected):
        q[p['rows'][selected]] = p['paths'][selected]
    return q


def labels(d, bank, p, cuts, arms, analytic_only=False):
    cv, ols = bank[:2]
    y, valid, rec = d['y'], d['valid'], d['rec']
    indices = p['rows']
    chosen = {k: p['anchors'].copy() for k in arms}
    stats = {k: Counter() for k in arms}
    if len(indices):
        path = p['paths']
        r = moments(cv[indices], path, y[indices], valid[indices], cuts[0])
        easy = moments(cv, cv, y, valid, cuts[0])['ref_easy']
        risk = np.column_stack([r[k] for k in ('all_excess','easy_excess','easy_signed')])
        offset = control.paired(ols, cv, y, valid)[1].sum(1)
        gain = np.column_stack([control.paired(q[indices], path, y[indices], valid[indices])[1].sum(1)
                                for q in (cv, ols)])
        gain[:, 1] -= offset[indices]
    for record in np.unique(rec):
        at = np.flatnonzero(rec[indices] == record)
        if analytic_only:
            at = at[at < p.get('analytic_edges', 0)]
        if not len(at):
            for k in arms:
                stats[k]['empty'] += 1
            continue
        use = rec == record
        for k in arms:
            guarded = k == 'risk_teacher'
            take, status = model.teacher(indices[at], p['slots'][at], gain[at], np.array([0., offset[use].sum()]),
                risk[at], easy[indices[at]] > 0., np.array([0.,0.,.02*easy[use].sum()]), guarded)
            if take is not None and guarded:
                loc = np.flatnonzero(use)
                out = cv[use].copy()
                out[np.searchsorted(loc, indices[at[take]])] = path[at[take]]
                metrics = control.parent.score(dict(y=y[use], valid=valid[use], rec=rec[use]), cv[use], ols[use], out, cuts)
                if metrics['dual_positive'] != 1:
                    take, status = None, 'unresolved'
            if take is not None:
                chosen[k][p['slots'][at[take]]] = at[take]
            stats[k][status] += 1
    return chosen, stats


def batcher(training, arm):
    data = []
    for p, chosen in training:
        if not len(p['rows']):
            continue
        target = chosen[arm]
        groups = [np.flatnonzero((p['slots'] == j) & (np.arange(len(p['slots'])) != z)) for j,z in enumerate(target)]
        legal = [j for j, z in enumerate(groups) if len(z)]
        if legal:
            data.append((torch.from_numpy(p['x']), target[legal], [groups[j] for j in legal]))
    if not data:
        raise ValueError('No donor ranking pairs')

    def sample(rng):
        n = np.bincount(rng.integers(len(data), size=128), minlength=len(data))
        pairs = []
        for k, (x, pos, groups) in zip(n, data, strict=True):
            if k:
                ids = rng.integers(len(pos), size=k)
                neg = [rng.choice(groups[j]) for j in ids]
                pairs.append((x[pos[ids]], x[neg]))
        return tuple(torch.cat(z) for z in zip(*pairs, strict=True))
    return sample


def decode(p, score):
    at = p['anchors'].copy()
    for q in np.unique(p['query']):
        which = np.flatnonzero(p['query'] == q)
        edges = np.flatnonzero(np.isin(p['slots'], which))
        value = score[edges]
        local = np.searchsorted(which, p['slots'][edges])
        take = slot.assign(p['rows'][edges], local, value, np.zeros(len(edges)), len(which))
        at[which] = edges[take]
    return at


def fit(data, source, reg, forecasts, old, stream, ranks, replay, beat):
    sites = {s:z['d'] for s,z in data.items()}
    banks = {s:control.parent.prior.analytic(d) for s,d in sites.items()}
    original = {(c['held'],c['seed']):c for c in stream['cells']}

    @lru_cache(maxsize=60)
    def donor_packet(held, s, seed):
        ex = tuple(sorted((held,s)))
        name = '_'.join(ex)+f'_{seed}'
        rank, count = ranks['rank_'+name], stream['count_heads']['count_'+name]
        for r in (rank,count):
            prior.cal.require_roles(r['identity'], sites, ex, seed)
        if count['identity']['registration'] != stream['registration']:
            raise ValueError('Streaming count artifact does not match registration')
        d, bank = sites[s], banks[s]
        with np.load(source.PRIVATE/f'{s}_cohort.npz', allow_pickle=False) as z:
            rec, start = z['recording'], z['start']
        net = study.load(rank, lambda: study.prior.model.SwapValue(False), study.RANK_STEPS)
        menu = study.menus(d, bank, net)
        x, _, _ = prior.base.features(d, bank, menu)
        net = study.load(count, lambda: prior.base.model.Value(x['motion'].shape[1]), prior.base.STEPS)
        with torch.inference_mode():
            score = prior.base.model.score(net(torch.from_numpy(x['motion'])).numpy(), risk=False)
        n = study.model.streaming_count(score, menu['legal'], rec, start)
        ref = next(z for z in stream['roster'][held][s] if z['seed'] == seed)
        if int(n.sum()) != ref['actual'] or int((n == 2).sum()) != ref['two_slot_queries']:
            raise ValueError('Donor streaming counts did not replay')
        use = np.arange(2) < n[:,None]
        rows, amps = menu['row'][use], v.AMPLITUDES[menu['arm'][use]]
        p = bank[0].copy()
        p[rows] = slot.features(d['h'][rows], amps)[3] if len(rows) else p[rows]
        ref = old['producer_fits']['producer_'+name]
        ident = ref['identity']
        prior.cal.require_roles(dict(ident, omitted=ident['excluded']), sites, ex, seed)
        spline = m.forecast(study.load(ref, m.model.Spline, m.STEPS), data[s]['inputs']['comoving_spline'])
        return design(d,bank,p,spline)

    fitted, cells, counts, updates = {}, [], {}, 0
    for held in sites:
        donor = [s for s in sites if s != held]
        cuts = np.quantile(np.concatenate([sites[s]['cut_samples'] for s in donor]), [.25,.75])
        for seed in m.SEEDS:
            name = f'{held}_{seed}'
            training, status = [], {k:Counter() for k in ARMS}
            for s in donor:
                p = donor_packet(held,s,seed)
                cutoff = np.quantile(np.concatenate([sites[t]['cut_samples'] for t in donor if t != s]), [.25,.75])
                choices, stat = labels(sites[s],banks[s],p,cutoff,ARMS)
                training.append((p,choices))
                for k in ARMS:
                    status[k].update(stat[k])
            counts[name] = status
            c = original[held,seed]
            with np.load(run.checked(c['paths']), allow_pickle=False) as z:
                paths = {k:z[k] for k in ('anchor','prefix_match')}
            sf = next(z for z in forecasts['cells'] if (z['held'],z['seed']) == (held,seed))
            if sf['fit'] != donor:
                raise ValueError('Unexpected held forecast fit role')
            with np.load(run.checked(sf['paths']), allow_pickle=False) as z:
                spline = z['comoving_spline']
            d, bank = sites[held],banks[held]
            p = design(d,bank,paths['anchor'],spline)
            width = next(z['x'].shape[1] for z,_ in training if len(z['rows']))
            chosen = {}
            for arm in ARMS:
                ident = dict(registration=reg, held=held, fit=donor, seed=seed, arm=arm)
                net, fitted[f'{name}_{arm}'], n = study.fit(ident, f'{name}_{arm}', seed,
                    lambda:model.Ranker(width), STEPS, batcher(training,arm), model.objective, replay, beat)
                updates += n
                with torch.inference_mode():
                    z = net(torch.from_numpy(p['x'])).numpy() if len(p['rows']) else np.empty(0)
                at = decode(p,z)
                cv = bank[0]
                paths[arm] = install(cv,p,at)
                chosen[arm] = dict(reassigned=int((p['rows'][at]!=p['old_rows']).sum()),
                    spline=int((at >= p.get('analytic_edges',0)).sum()))
                rows = p['rows'][at]
                if len(set(rows)) != len(at) or len(rows) != len(p['old_rows']):
                    raise ValueError('Decoder violated original counts')
                np.testing.assert_array_equal(d['query'][rows],p['query'])
                np.testing.assert_allclose(control.radius.peak(paths[arm][rows]-cv[rows]),
                    control.radius.peak(paths['anchor'][p['old_rows']]-cv[p['old_rows']]), atol=1e-9, rtol=0.)
            cells.append(dict(held=held,seed=seed,cut=cuts.tolist(), chosen=chosen,
                              paths=m.save(PRIVATE/f'{name}_paths.npz',paths), spline=sf['paths']))
            beat(phase='frozen',held=held,seed=seed,updates=updates)
    return dict(cells=cells, fits=fitted, teacher_counts=counts, before_held_scoring=True), updates


def evaluate(data, frozen):
    rows = []
    arms = ('anchor','prefix_match',*ARMS)
    pairs = (('gain_teacher','risk_teacher'),('anchor','risk_teacher'),('prefix_match','risk_teacher'))
    for c in frozen['cells']:
        s, seed = c['held'],c['seed']
        d = data[s]['d']
        bank = control.parent.prior.analytic(d)
        cv, ols = bank[:2]
        with np.load(run.checked(c['paths']),allow_pickle=False) as z:
            paths = {k:z[k] for k in arms}
        with np.load(run.checked(c['spline']),allow_pickle=False) as z:
            p = design(d,bank,paths['anchor'],z['comoving_spline'])
        caps = {}
        for name, only in (('analytic',True),('both_shapes',False)):
            _, counts = labels(d,bank,p,c['cut'],('risk_teacher',),only)
            caps[name] = counts['risk_teacher']
        y, valid = d['y'],d['valid']
        mass = np.linalg.norm(cv-y,axis=-1)[valid].sum()
        scores = {k:control.parent.score(d,cv,ols,a,c['cut']) for k,a in paths.items()}
        tail = {}
        for k,a in paths.items():
            z = prior.policy(a,cv,cv,y,valid,d['rec'],easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            easy = [v['known_easy_net_degradation'] for v in z['recordings'].values()
                    if v['known_easy_net_degradation'] is not None]
            tail[k] = dict(worst_easy=100.*max(easy) if easy else None,
                           hard_gain=z['subsets']['hard']['gain_percent'])
        effect = {f'{b}_vs_{a}':float(100.*control.paired(paths[a],paths[b],y,valid)[1].sum()/mass) for a,b in pairs}
        rows.append(dict(site=s,seed=seed,scores=scores,effects=effect,tail=tail,capacity=caps,chosen=c['chosen']))
    draw = np.random.default_rng(817).integers(len(data), size=(3000,len(data)))
    ci = lambda fn: prior.geometry.bootstrap([[fn(r) for r in rows if r['site']==s] for s in data], draw)
    summary = dict(policies={k:{f:sum(r['scores'][k][f] for r in rows) for f in
        ('edits','risk_pass','dual_positive','undefined','views')} for k in arms},
        effects={k:ci(lambda r:r['effects'][k]) for k in rows[0]['effects']},
        capacity={k:sum((r['capacity'][k] for r in rows),Counter()) for k in rows[0]['capacity']},
        tail={k:dict(worst_easy=max(r['tail'][k]['worst_easy'] for r in rows if r['tail'][k]['worst_easy'] is not None),
                     hard_gain=ci(lambda r:r['tail'][k]['hard_gain'])) for k in arms},
        chosen={k:{f:sum(r['chosen'][k][f] for r in rows) for f in ('reassigned','spline')} for k in ARMS})
    if (len(rows)!=21 or any((v['edits'],v['views'])!=(1457,606) for v in summary['policies'].values())
            or summary['policies']['anchor']['dual_positive'] != 77):
        raise ValueError('Original evaluation roster or reference changed')
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v0):
            z = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v0)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',z)
            if v0.get('phase') != 'fit':
                print(z,flush=True)
        reg,source,forecasts,old,stream,ranks = register()
        data = m.load(source)
        payload,updates = fit(data,source,reg,forecasts,old,stream,ranks,args.replay,beat)
        p = OUT/'freeze.json'
        run.once(p,run.stable(dict(registration=reg,**payload,configured_updates=STEPS*len(payload['fits']))))
        rows,summary = evaluate(data,run.read(p))
        run.once(OUT/'readout.json',run.stable(dict(freeze=run.sha(p),rows=rows,summary=summary,
            original_2pct=True,confirmation_access=False,independent_confirmation=False,deployment_changed=False)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__ == '__main__':
    main()
