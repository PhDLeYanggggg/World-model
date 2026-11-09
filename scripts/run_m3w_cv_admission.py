"""Train and compare support-aware optional selection inside fitting sources."""

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
from scripts import run_m3w_cv_joint_selector as prior
from src.world_model import m3w_cv_admission as model

np, torch, read, sha, once, stable, artifact, checked = (prior.np, prior.torch, prior.read, prior.sha, prior.once,
                                                        prior.stable, prior.artifact, prior.checked)
OUT = prior.OUT.parent / 'moving_cv_admission_v1'
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS, SEEDS, CAP = 1000, prior.SEEDS, 8
FRACTIONS = (.05, .15, .35, 1.)
GRID = tuple((t, p) for t in (-.005, 0., .002, .01, .03) for p in (0., .10, .25))
ARMS = ('forced', 'calibrated', 'strict', 'strict_joint', 'strict_gain_match',
        'relaxed', 'relaxed_joint', 'relaxed_gain_match')
CV, OLS = prior.CV, prior.OLS
PAIRS = {f'{arm}_vs_{key}': (b, arm) for arm in ARMS for key, b in (('cv', CV), ('ols4', OLS))}
PAIRS.update({f'{k}_joint_vs_independent': (k, k+'_joint') for k in ('strict', 'relaxed')})
PAIRS.update({f'{k}_risk_vs_gain_match': (k+'_gain_match', k) for k in ('strict', 'relaxed')})
PAIR = ('strict_joint_vs_independent', 'relaxed_joint_vs_independent')


def register():
    ref = prior.donor.OUT / 'freeze.json'
    ver = prior.donor.OUT / 'verification.json'
    d = read(ref)
    if d['held_source_draws'] or d['a_labels_accessed'] or read(ver)['parent']['sha256'] != sha(ref):
        raise ValueError('Require verified held donor paths')
    groups = read(prior.donor.OUT / 'registration.json')['donor_groups']
    roles = [dict(held=groups[i], train=groups[(i+1)%3], calibration=groups[(i+2)%3]) for i in range(3)]
    reg = OUT / 'registration.json'
    once(reg, stable(dict(donors=artifact(ref), verification=artifact(ver), files=[artifact(p) for p in
        (Path(__file__), Path(model.__file__), Path(prior.__file__), OUT / 'protocol.md')], roles=roles,
        seeds=SEEDS, cap=CAP, steps=STEPS, thresholds=GRID, blend_fractions=FRACTIONS,
        a_labels_opened=False, independent_confirmation=False, original_2pct=True)))
    return sha(reg), d, roles


def fit_population(beat):
    _, jobs = prior.study.parent.parent.register()
    p = prior.donor.prior.prior
    d = p.fitting_only(jobs[0]['identity'], beat)
    h, q = prior.layout(d, jobs[0]['identity'])
    adm = p.prep.load_admission(p.ROOT, p.ROOT / read(p.source.CONFIG)['admission'])
    lookup = {r['clip']: r for r in read(p.bank.old.HOME / 'movingdronecrowd_intake_v1/analysis.json')['recordings']}
    counts = [lookup[adm['recordings'][r]['clip']]['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
              for r in d['recordings']]
    d.update(history=h, query=q, recording=np.repeat(np.arange(len(counts)), counts))
    for i, name in enumerate(d['recordings']):
        if not (d['sites'][d['recording'] == i] == adm['recordings'][name]['exclusion_group']).all():
            raise ValueError('Fitting recording layout changed')
    return d


def subset(d, groups):
    ix = np.isin(d['sites'], groups)
    return ix, {k: d[k][ix] for k in ('history', 'query', 'recording', 'sites', 'target', 'valid', 'ids')}


def expanded(cv, paths):
    return np.concatenate([cv[:, None]+f*(paths-cv[:, None]) for f in FRACTIONS], axis=1)


def thresholds(cv, d):
    e, _ = prior.prior.parent.readout.errors(cv, d['target'], d['valid'])
    return float(np.quantile(e[np.isfinite(e) & (e > 0.)], .25)), float(np.quantile(e[np.isfinite(e)], .75))


def row_sampler(cv, ols, bank, d, easy):
    past, _ = prior.study.motion.features(d['history'], 'prefix')
    s = prior.model.scale(d['history'])
    groups = [np.flatnonzero(d['sites'] == k) for k in sorted(set(d['sites']))]

    def sample(rng):
        i = prior.draw(groups, rng)
        j = rng.integers(bank.shape[1], size=len(i))
        p, c, o = bank[i, j], cv[i], ols[i]
        y, v = d['target'][i], d['valid'][i]
        _, _, g, _ = prior.utility.candidate_terms(c, o, p[:, None], y, v)
        cc, pp = [prior.prior.parent.readout.errors(z, y, v)[0] for z in (c, p)]
        known = v.any(1)
        ee = known & (cc > 0.) & (cc <= easy)
        mass = np.column_stack((np.where(known, cc, 0.), np.where(known, np.maximum(pp-cc, 0.), 0.)))
        label = np.concatenate((g[:, 0]/(12*s[i, None]), mass/s[i, None],
            np.where(ee[:, None], mass/s[i, None], 0.), np.column_stack((ee, ~known, known & ~ee))), axis=1)
        return (torch.from_numpy(model.features(past[i], p-c, s[i])),
                torch.from_numpy(label.clip(-8., 8.).astype(np.float32)))
    return sample, past.shape[1]+25


@torch.inference_mode()
def nominate(net, cv, bank, h):
    past, _ = prior.study.motion.features(h, 'prefix')
    scale = prior.model.scale(h)
    legal = np.linalg.norm(bank-cv[:, None], axis=-1).max(-1) > 1e-8
    n = len(h)
    score, risk, easy = np.full(n, -1e9), np.zeros((n, 2)), np.zeros(n)
    choice = np.full(n, -1, dtype=np.int8)
    for j in range(bank.shape[1]):
        for a in range(0, n, 4096):
            ix = slice(a, a+4096)
            d = bank[ix, j]-cv[ix]
            x = model.features(past[ix], d, scale[ix])
            value = [v.numpy() for v in model.decode(net(torch.from_numpy(x)))]
            g, r, e = model.predicted(*value, np.linalg.norm(d, axis=-1).max(-1)/scale[ix])
            take = legal[ix, j] & (g > score[ix])
            score[ix][take], risk[ix][take], easy[ix][take], choice[ix][take] = g[take], r[take], e[take], j
    return choice, score, risk, easy


@torch.inference_mode()
def pair_estimates(net, h, cv, edited, groups):
    (i, j), local, off = prior.edges(h, groups)
    val = np.zeros((len(i), 3))
    for start in range(0, len(i), 4096):
        sl = slice(start, start+4096)
        a, b = i[sl], j[sl]
        c = prior.model.pair_context(h[a], h[b], cv[a], cv[b])
        x, y = edited[a], edited[b]
        z = np.zeros_like(x)
        f = lambda p, q: net(torch.from_numpy(prior.model.pair_features(c, p, q))).numpy().ravel()
        base = f(z, z)
        val[sl] = np.column_stack((f(x,z)-base, f(z,y)-base, f(x,y)-base))
    return local, off, val


def choices(nom, groups, gate, pair=None):
    action, value, risk, p_easy = nom
    select = np.full(len(action), -1, dtype=np.int8)
    if gate is None:
        return select
    eligible = model.admissible(value, risk, p_easy, action >= 0, *gate)
    for j, g in enumerate(groups):
        terms = (None, None) if pair is None else (pair[0][j], pair[2][pair[1][j]:pair[1][j+1]])
        ix = g[model.select(value[g], eligible[g], CAP, *terms)]
        select[ix] = action[ix]
    return select


def matched(nom, groups, choice=None):
    action, value, _, _ = nom
    out = np.full(len(action), -1, dtype=np.int8)
    for g in groups:
        k = CAP if choice is None else int((choice[g] >= 0).sum())
        ix = g[model.select(value[g], action[g] >= 0, k)]
        out[ix] = action[ix]
    return out


def metrics(cv, ols, bank, d, act, cut):
    out = []
    for rec in np.unique(d['recording']):
        ix = d['recording'] == rec
        q = d['query'][ix]
        y, mask = d['target'][ix], d['valid'][ix]
        c, o = cv[ix], ols[ix]
        paths = {CV:c, OLS:o, **{k: prior.apply(c, bank[ix], v[ix]) for k, v in act.items()}}
        scorer = prior.prior.parent.readout
        r = scorer.score_recording(paths, {}, y, mask, q, str(rec), easy_cut=cut[0], hard_cut=cut[1])
        _, inv, sizes = np.unique(q, return_inverse=True, return_counts=True)
        w = 1./sizes[inv]
        pairs = {k: p for k, p in PAIRS.items() if set(p) <= paths.keys()}
        r['comparisons'] = {k: prior.study.motion.full_masses(paths[a], paths[b], c, y, mask, w) for k, (a,b) in pairs.items()}
        s, = np.unique(d['sites'][ix])
        if all(k in paths for k in ARMS):
            geo = np.zeros((2,4))
            for g in prior.query_groups(q):
                arr = np.stack([paths[k][g] for k in ('strict','strict_joint','relaxed','relaxed_joint')])
                _, score, _, (steps, known, _, _) = prior.geometry.query(d['history'][ix][g], arr, y[g], mask[g], ((0,1),(2,3)))
                if steps: geo[:,1] += score[:,1]; geo[:,3] += 1
                if known: geo[:,0] += score[:,0]; geo[:,2] += 1
            r['geometry'] = geo.tolist()
        out.append(dict(r, source=str(s)))
    return out


def calibrate(net, d, cv, ols, paths, cut):
    nom = nominate(net, cv, paths, d['history'])
    groups = prior.query_groups(d['query'])
    result = []
    for t, p in GRID:
        a = choices(nom, groups, (t,p))
        rows = metrics(cv, ols, paths, d, {'strict':a}, cut)
        by = []
        for s in sorted(set(d['sites'])):
            r = [v for v in rows if v['source'] == s]
            z = [prior.prior.parent.readout.merge_comparisons([v['comparisons'][f'strict_vs_{k}'] for v in r])
                 for k in ('cv','ols4')]
            by.append([v['completion_lower_pp'] for v in z])
        result.append(dict(tolerance=t, min_easy=p, edits=int((a >= 0).sum()),
            passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),
            gain_pp=np.mean(by, axis=0).tolist()))
    return result, model.choose_setting(result, len(np.unique(d['recording'])))


def setup_cells(reg, donor, roles, d, beat, replay):
    cells, updates = [], 0
    for fold, roles0 in enumerate(roles):
        ti, train = subset(d, roles0['train'])
        _, cal = subset(d, roles0['calibration'])
        hi, held = subset(d, roles0['held'])
        if (ti & hi).any() or set(train['sites']) & set(cal['sites']):
            raise ValueError('Disjoint source roles required')
        cv, ols, *rest = prior.study.motion.bank(train['history']).swapaxes(0,1)
        bank = expanded(cv, np.stack((ols,*rest), axis=1))
        cut = thresholds(cv, train)
        row, width = row_sampler(cv, ols, bank, train, cut[0])
        edge, ew = prior.edge_sample(train['history'], cv, bank, train['target'], train['valid'], train['sites'],
                                     prior.query_groups(train['query']))
        cp = prior.study.motion.bank(cal['history'])
        ca = expanded(cp[:,0], cp[:,1:])
        for seed in SEEDS:
            ident = dict(registration=reg, fold=fold, seed=seed, **roles0)
            nets, fits = [], []
            for name, sampler, w, n, loss in (('risk',row,width,8,model.objective), ('pair',edge,ew,1,prior.edge_loss)):
                net, ck, u = prior.prefix.trainer.fit(dict(ident,head=name), f'{name}_{fold}_{seed}', seed,
                    lambda: prior.model.head(w,n), STEPS, sampler, loss, replay, beat)
                nets.append(net); fits.append(ck); updates += u
            c = PRIVATE / f'cal_{fold}_{seed}.json'
            if not c.exists():
                if replay: raise ValueError('Missing calibration receipt')
                ladder, selected = calibrate(nets[0], cal, cp[:,0], cp[:,1], ca, cut)
                once(c, stable(dict(identity=ident, ladder=ladder, setting=selected,
                                     records=len(np.unique(cal['recording'])))))
            cc = read(c)
            if cc['identity'] != ident: raise ValueError('Calibration role changed')
            pred = prior.read_donor(donor, seed, d)
            cvh, _, paths = prior.bank(held['history'], {k:v[hi] for k,v in pred.items()})
            paths = expanded(cvh, paths)
            g = prior.query_groups(held['query'])
            nom = nominate(nets[0], cvh, paths, held['history'])
            delta = prior.apply(cvh, paths, nom[0])-cvh
            pair = pair_estimates(nets[1], held['history'], cvh, delta, g)
            a = dict(forced=matched(nom,g), calibrated=choices(nom,g,cc['setting']))
            for k,t in (('strict',0.), ('relaxed',.03)):
                a[k] = choices(nom,g,(t,.10))
                a[k+'_joint'] = choices(nom,g,(t,.10),pair)
                a[k+'_gain_match'] = matched(nom,g,a[k])
                for block in g:
                    count = (a[k][block] >= 0).sum()
                    if any((a[x][block] >= 0).sum() != count for x in (k+'_joint',k+'_gain_match')):
                        raise ValueError('Matched query quota changed')
            path = prior.study.parent.parent.save_actions(PRIVATE / f'choice_{fold}_{seed}.npz', dict(ids=held['ids'], **a))
            cells.append(dict(identity=ident, cut=cut, calibration=artifact(c), fits=fits, actions=path))
            beat(phase='held_frozen',fold=fold,seed=seed,updates=updates,calibrated=cc['setting'])
    f = OUT / 'freeze.json'
    once(f,stable(dict(registration=reg, before_held_scoring=True, cells=cells, updates=len(cells)*2*STEPS)))
    return f, updates


def readout(reg, f, donor, d):
    binding = read(f)
    if binding['registration'] != reg or len(binding['cells']) != 9 or not binding['before_held_scoring']:
        raise ValueError('Require all outer choices frozen')
    rows, calibration, swapped = [], [], {k:0 for k in ('strict','relaxed')}
    for cell in binding['cells']:
        ident = cell['identity']
        seed = ident['seed']
        hi, held = subset(d, ident['held'])
        cv, ols, bank = prior.bank(held['history'], {k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        with np.load(checked(cell['actions']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], held['ids'])
            a = {k:z[k] for k in ARMS}
        for k in swapped: swapped[k] += int(((a[k+'_joint'] >= 0) & (a[k] < 0)).sum())
        rows.extend(dict(r,seed=seed) for r in metrics(cv,ols,expanded(cv,bank),held,a,cell['cut']))
        c = read(checked(cell['calibration']))
        calibration.append(dict(fold=ident['fold'],seed=seed,setting=c['setting'],
            max_passes=max(v['passes'] for v in c['ladder']), records=c['records'],
            max_edits=max(v['edits'] for v in c['ladder'])))
    summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},SEEDS)
    if (summary['views'], summary['source_groups']) != (72,9):
        raise ValueError('Fitting-only evaluation roster changed')
    sources = sorted(set(r['source'] for r in rows))
    draw = np.random.default_rng(73107).integers(len(sources), size=(3000,len(sources)))
    t = np.array([[np.sum([r['geometry'] for r in rows if r['source'] == s and r['seed'] == k],axis=0)
                   for k in SEEDS] for s in sources])
    p = np.divide(t[...,:2],t[...,2:],out=np.full_like(t[...,:2],np.nan),where=t[...,2:]>0.)
    pair = {k:dict(observed_pixel=prior.geometry.bootstrap(p[:,:,i,0],draw),
                   lower_pixel=prior.geometry.bootstrap(p[:,:,i,1],draw)) for i,k in enumerate(PAIR)}
    result = stable(dict(registration=reg,freeze=sha(f),summary=summary,pair=pair,calibration=calibration,
        qualified=sum(v['setting'] is not None for v in calibration),swapped_recipients=swapped,
        original_2pct=True,a_labels_opened=False,independent_confirmation=False,submission_ready=False))
    once(OUT / 'readout.json',result)
    once(PRIVATE / 'recording_scores.json',stable(dict(parent=sha(OUT/'readout.json'),rows=rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start=time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

        def beat(**v):
            info=dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',info)
            if v.get('step',STEPS) in (100,STEPS): print(info,flush=True)

        r, oof, roles=register()
        d=fit_population(beat)
        f,u=setup_cells(r,oof,roles,d,beat,args.replay)
        readout(r,f,oof,d)
        beat(phase='complete',updates=u,replay=args.replay)


if __name__=='__main__':
    main()
