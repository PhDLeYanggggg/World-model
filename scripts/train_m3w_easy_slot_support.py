"""Learn donor-only easy reference support and test exact-budget placement protection."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts import audit_m3w_opendd_paired_window_budget as prior
from src.world_model import m3w_easy_slot_support as model

base, study, np, torch, exp = prior.base, prior.study, prior.np, prior.torch, prior.exp
SITES, SEEDS = prior.SITES, prior.SEEDS
OUT = prior.OUT.parent/'opendd_easy_slot_support_v1'
LOCAL = prior.LOCAL.parent/OUT.name
STEPS, BATCH = 200, 128
ARMS = tuple(k+'_'+v for k in ('fixed', 'pooled', 'guarded') for v in ('additive', 'joint'))


def register():
    key, cohorts, *_ = prior.register()
    paths = [prior.OUT/f'{k}.json' for k in ('freeze', 'readout', 'risk_attribution')]
    f, r, a = map(study.read, paths)
    if (f['registration'] != key or f['outcomes_accessed'] or r['freeze'] != study.artifact(paths[0])
            or a['parent'] != study.artifact(paths[1]) or r['reserved_outcomes_opened']):
        raise ValueError('Require the matched budget study and paired risk attribution')
    p = OUT/'registration.json'
    base.immutable(p, dict(parent=[study.artifact(x) for x in paths], arms=ARMS,
        code=[study.artifact(q) for q in (Path(__file__), Path(model.__file__), ROOT/'src/world_model/m3w_easy_reference_rank.py')],
        sites=SITES, seeds=SEEDS, updates=STEPS, batch=BATCH, checkpoint_every=50,
        labels='known positive CV ADE below donor-only easy cut; conditional CV reference amount',
        gate='keep pooled pair only if predicted easy reference sum is no lower than fixed pair',
        cpu_threads=4, workers=0, original_2pct=True, threshold_tuning=False,
        offline_two_prefixes=True, independent_confirmation=False))
    return base.sha(p), cohorts, f, r


def fit(key, held, seed, src, features):
    name = f'{held}_{seed}'
    donors = tuple(s for s in SITES if s != held)
    groups = []
    for site in donors:
        d = src[site]
        cut = float(np.quantile(np.concatenate([src[s]['e'] for s in donors if s != site]), .25))
        e, m = model.targets(d['cv'], d['y'], d['v'], cut)
        groups.append(tuple(torch.from_numpy(v.astype(np.float32, copy=False)) for v in (features[site], e, m)))
    ident = dict(registration=key, held=held, seed=seed, donors=donors)
    path = LOCAL/f'{name}.pt'
    torch.manual_seed(seed)
    net = model.EasySupport(features[held].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+227)
    step, losses = 0, []
    if path.exists():
        s = torch.load(path, weights_only=True, map_location='cpu')
        if s['identity'] != ident or s['step'] not in range(50, STEPS+1, 50):
            raise ValueError('Support checkpoint used a different source split')
        net.load_state_dict(s['net'])
        opt.load_state_dict(s['opt'])
        step, losses, rng.bit_generator.state = s['step'], s['losses'], s['rng']
    while step < STEPS:
        n = rng.multinomial(BATCH, np.full(len(groups), 1./len(groups)))
        ix = [rng.integers(len(g[0]), size=k) for g, k in zip(groups, n, strict=True)]
        b = [torch.cat([g[i][at] for g, at in zip(groups, ix, strict=True)]) for i in range(3)]
        loss = net.loss(*b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        step += 1
        if step % 50 == 0:
            losses.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), opt=opt.state_dict(), losses=losses,
                            rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            heartbeat(name, step)
    return net.eval(), dict(held=held, donors=donors, checkpoint=study.artifact(path), loss=losses)


def heartbeat(name, step):
    p = LOCAL/'heartbeat.pending'
    p.write_text(study.json.dumps(dict(pid=os.getpid(), fold=name, step=step, total=STEPS))+'\n')
    p.replace(LOCAL/'heartbeat.json')


def freeze(key, cohorts, parent):
    d = exp.loader(cohorts)
    x = {s: model.prefix_features(a['h'], a['q']) for s, a in d.items()}
    fits, files, movement = {}, {}, {}
    for site in SITES:
        q = d[site]['q']
        with np.load(study.checked(cohorts[site]), allow_pickle=False) as src:
            blocks = [np.flatnonzero(np.isin(q, b)) for b in prior.pairs(src['recording'], src['start'], study.donor.STRIDE)]
        for seed in SEEDS:
            name = f'{site}_{seed}'
            net, fits[name] = fit(key, site, seed, d, x)
            prob, support = net.estimate(x[site])
            with np.load(study.checked(parent['banks'][name]), allow_pickle=False) as old:
                np.testing.assert_array_equal(q, old['q'])
                data = {k: old[k] for k in old.files}
            cv = data['cv']
            stats = {}
            for k in ('additive', 'joint'):
                path, stats[k] = model.guard(cv, data[f'fixed_{k}'], data[f'pooled_{k}'], support, blocks)
                data[f'guarded_{k}'] = path
                fixed_peak, new_peak = (np.linalg.norm(data[f'{a}_{k}']-cv, axis=-1).max(1) for a in ('fixed', 'guarded'))
                for ix in blocks:
                    np.testing.assert_allclose(np.sort(fixed_peak[ix]), np.sort(new_peak[ix]), atol=1e-9, rtol=1e-9)
            files[name] = exp.previous.save_bank(LOCAL/f'{name}.npz', dict(**data, prob=prob, support=support))
            movement[name] = stats
            print(name, stats, flush=True)
    base.immutable(OUT/'freeze.json', dict(registration=key, fits=fits, banks=files, movement=movement,
        independent_confirmation=False, donor_labels_only_per_fold=True, uses_later_observation_prefix=True))


def score(key, cohorts, previous):
    p = OUT/'freeze.json'
    f = study.read(p)
    if f['registration'] != key or not f['donor_labels_only_per_fold'] or len(f['banks']) != 21:
        raise ValueError('Require held predictions before scoring')
    src, by_site = exp.loader(cohorts), {}
    for site in SITES:
        d = src[site]
        cut = np.quantile(np.concatenate([a['e'] for s, a in src.items() if s != site]), [.25, .75])
        eligible, _ = model.targets(d['cv'], d['y'], d['v'], cut[0])
        counts, diffs, probes = {k: Counter() for k in ARMS}, {'additive': [], 'joint': []}, []
        for seed in SEEDS:
            name = f'{site}_{seed}'
            if set(f['fits'][name]['donors']) != set(SITES)-{site}:
                raise ValueError('Scored source contributed training labels')
            with np.load(study.checked(f['banks'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(d['q'], z['q'])
                paths = {k: z[k] for k in ('cv', 'ols4', *ARMS)}
                if eligible.any() and not eligible.all():
                    probes.append(dict(auroc=roc_auc_score(eligible, z['prob']),
                                       average_precision=average_precision_score(eligible, z['prob'])))
            for rec in np.unique(d['rec']):
                ix = d['rec'] == rec
                names = ('pooled_additive', 'pooled_joint', 'guarded_additive', 'guarded_joint')
                pths = {k: paths[v][ix] for k, v in zip(prior.menu.ARMS, names, strict=True)}
                pths.update(cv=paths['cv'][ix], ols4=paths['ols4'][ix])
                q = np.unique(d['q'][ix], return_inverse=True)[1]
                m, pair = prior.menu.recording(pths, d['y'][ix], d['v'][ix], q, cut)
                for k, v in zip(prior.menu.ARMS, names, strict=True):
                    counts[v].update({j: int(m[k][j]) for j in ('edits', 'risk', 'dual')})
                    easy = eligible[ix] & (np.linalg.norm(paths[v][ix]-pths['cv'], axis=-1).max(1) > 0.)
                    counts[v]['defined_selected_easy'] += int(easy.any())
                for k in diffs:
                    if pair[f'expanded_{k}_vs_base_{k}'] is not None:
                        diffs[k].append(pair[f'expanded_{k}_vs_base_{k}'])
        for k in ('fixed_additive', 'fixed_joint', 'pooled_additive', 'pooled_joint'):
            before = previous['sites'][site]['counts'][k]
            if k.startswith('fixed'):
                counts[k].update(before)
            elif any(counts[k][j] != v for j, v in before.items()):
                raise ValueError('Original pooling scores changed')
        by_site[site] = dict(counts=counts, contrast={k: float(np.mean(v)) if v else None for k, v in diffs.items()},
                             probes=probes, positive_rate=float(eligible.mean()))
    def interval(k):
        x = np.array([v['contrast'][k] for v in by_site.values() if v['contrast'][k] is not None])
        draws = np.random.default_rng(3183).choice(x, size=(3000, len(x))).mean(1)
        return dict(mean_pp=float(x.mean()), ci95_pp=np.quantile(draws, [.025, .975]).tolist(), sites=len(x))
    result = dict(result_source='fresh_run', new_fits=21, updates=21*STEPS, freeze=study.artifact(p),
        totals={k: dict(sum((a['counts'][k] for a in by_site.values()), Counter())) for k in ARMS},
        contrast={k: interval(k) for k in ('additive', 'joint')}, sites=by_site,
        movement={k: dict(sum((Counter(a[k]) for a in f['movement'].values()), Counter())) for k in ('additive', 'joint')},
        original_2pct=True, offline_diagnostic=True, independent_confirmation=False, deployment_changed=False)
    base.immutable(OUT/'readout.json', result)
    print(study.json.dumps({k: result[k] for k in ('totals', 'contrast', 'movement')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, previous = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, parent)
        else:
            score(key, cohorts, previous)


if __name__ == '__main__':
    main()
