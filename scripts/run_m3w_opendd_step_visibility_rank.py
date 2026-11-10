"""Train causal step-observability heads and test fixed-proposal ranking."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys

if __name__ == '__main__' and sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_fixed_joint_budget as previous
from src.evaluation import m3w_step_visibility_rank as visibility

prior, np, budget, pair, sha, read, write = (getattr(previous, k) for k in
    ('prior', 'np', 'budget', 'pair', 'sha', 'read', 'write'))
torch, old, support = prior.torch, prior.old, prior.support
OUT = previous.OUT.parent/'opendd_step_visibility_rank_v1'
LOCAL = previous.LOCAL.parent/OUT.name
STEPS = 200
ARMS = ('support', 'profile', 'learned')


def register():
    key, cells, cohorts, f, _ = previous.register()
    mask_path = previous.OUT/'mask_freeze.json'
    masks = read(mask_path)
    if (masks['registration'] != key or masks['outcomes_read'] or masks['optimal_witnesses'] != 252):
        raise ValueError('Require original movement-matched support masks')
    files = (Path(__file__), Path(visibility.__file__), mask_path, previous.OUT/'readout.json',
             prior.OUT/'score_freeze.json', OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(inputs={str(k.relative_to(prior.ROOT)): sha(k) for k in files}, steps=STEPS,
                  checkpoint_every=old.SAVE, seeds=prior.base.SEEDS, threads=4, workers=0,
                  original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(p), cells, cohorts, f, masks


def head():
    return torch.nn.Sequential(torch.nn.Linear(17, 32), torch.nn.SiLU(), torch.nn.Linear(32, 12))


def features(d, cut):
    x, _ = support.features(d['history'], cut)
    return torch.from_numpy(np.c_[x, -np.log(d['weight'])].astype(np.float32))


def loss(net, batch):
    x, y = batch
    return torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)


def fit(key, name, c, donors, past, cohorts):
    d = {k: cohorts[k] for k in donors}
    labels = {s: (~v['valid']).astype(np.float32) for s, v in prior.cohort_rows(d, ('valid',))}
    batches = [(features(past[s], c['cut'][0]), torch.from_numpy(labels[s])) for s in donors]
    rates = np.mean([labels[s].mean(axis=0, dtype=np.float64) for s in donors], axis=0)
    identity = dict(name=f'{name}_visibility', registration=key, seed=c['seed'],
                    excluded=sorted(c['packets']), fit_sites=donors)
    before = old.LOCAL
    try:
        old.LOCAL = LOCAL
        torch.manual_seed(c['seed'])
        net, check = old.fit(identity, batches, loss, head(), STEPS, old.BATCH)
    finally:
        old.LOCAL = before
    return net, check, rates


def beat(phase, n):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=63))
    if n == 1 or n % 10 == 0 or n == 63:
        print(f'{phase}: {n}/63', flush=True)


@torch.inference_mode()
def infer(name, site, c, ref, net, rates, data, score_freeze, masks, seconds):
    prefix = previous.LOCAL/f'{name}_{site}.npz'
    if sha(prefix) != masks['packets'][f'{name}:{site}']:
        raise ValueError('Prior frozen mask changed')
    with np.load(prefix, allow_pickle=False) as m, prior.prior.packet(prior.learn.LOCAL, ref) as z:
        cv, rows, paths = (z[k] for k in ('cv', 'joint_rows', 'joint_paths'))
        dist = np.linalg.norm(paths-cv[rows], axis=-1)
        p = net(features(data, c['cut'][0])).sigmoid().numpy()[rows]
        q = np.broadcast_to(rates, p.shape)
        estimate = np.stack([visibility.normalized_debit(dist, x) for x in (q, p)])
        merit = prior.scores(name, site, score_freeze)[1]
        cost = budget.movement(cv, rows, paths, data['weight'])
        np.testing.assert_array_equal(m['costs'], cost)
        match = m['movement_matched']
        chosen = np.stack((match[1], *(budget.select(merit-v, cost, match[0], seconds) for v in estimate)))
        return dict(masks=chosen, original=match[0], cost=cost, p=p, profile=rates, predicted_debit=estimate)


def freeze(key, cells, cohorts, f, masks, seconds):
    past = dict(prior.cohort_rows(cohorts, ('history', 'weight')))
    done = {}
    for n, (name, c) in enumerate(cells.items(), 1):
        file = LOCAL/f'{name}.json'
        if not file.exists():
            donors = c['fit_sites']
            if set(donors) != set(cohorts)-set(c['packets']) or len(donors) != 5:
                raise ValueError('Both held and calibration locations must be excluded')
            net, check, rate = fit(key, name, c, donors, past, cohorts)
            packets = {}
            for site, ref in c['packets'].items():
                data = infer(name, site, c, ref, net, rate, past[site], f, masks, seconds)
                p = LOCAL/f'{name}_{site}.npz'
                packets[site] = prior.base.parent.indices.save_npz(p, data)
            write(file, dict(registration=key, weight=check, packets=packets), immutable=True)
        r = read(file)
        if (r['registration'] != key or r['weight']['step'] != STEPS
                or sha(LOCAL/f'{name}_visibility.pt') != r['weight']['sha256']
                or set(r['packets']) != set(c['packets'])
                or any(sha(LOCAL/f'{name}_{s}.npz') != v for s, v in r['packets'].items())):
            raise ValueError('Cached checkpoint or selection changed')
        done[name] = r
        beat('freeze_visibility', n)
    write(OUT/'freeze.json', dict(registration=key, cells=done, fits=len(done), optimizer_updates=len(done)*STEPS,
          held_outcomes_loaded=False, original_2pct=True), immutable=True)


def measure(z, p, d, cut):
    cv, rows, paths = (z[k] for k in ('cv', 'joint_rows', 'joint_paths'))
    cost = budget.movement(cv, rows, paths, d['weight'])
    np.testing.assert_array_equal(cost, p['cost'])
    missing = ~d['valid'][rows]
    actual = visibility.normalized_debit(np.linalg.norm(paths-cv[rows], axis=-1), missing)
    prob = (np.broadcast_to(p['profile'], missing.shape), p['p'])
    metrics = {a: dict(brier=float(np.mean((q-missing)**2)),
                       debit_mae=float(np.mean(abs(v-actual))))
               for a, q, v in zip(ARMS[1:], prob, p['predicted_debit'], strict=True)}
    y = pair.evaluate(pair.prediction(z, 'joint'), z, d, cut)
    k = int(p['original'].sum())
    mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(1))
    pred, counts = {}, {}
    for a, use in zip(ARMS, p['masks'], strict=True):
        ratio = cost[:, use].sum(axis=1)/cost[:, p['original']].sum(axis=1)
        if int(use.sum()) != k or np.any(abs(ratio-1.) > .01000001):
            raise ValueError('Movement or count changed')
        pred[a] = prior.policy.prediction(cv, rows, paths, use)
        counts[a] = dict(selected=k, dual=int((use & y['dual']).sum()), risk_pass=int((use & y['risk']).sum()),
                         undefined_easy=int((use & y['missing']).sum()), max_spend_error=float(abs(ratio-1.).max()))
    contrasts = {}
    for a, b in (('learned', 'profile'), ('learned', 'support'), ('profile', 'support')):
        q = pair.paired(pred[a], pred[b], d, z['bounds'], mass)
        q['dual_rate_pp'] = 100.*(counts[a]['dual']-counts[b]['dual'])/k
        if b == 'profile':
            for name in metrics[a]:
                q[name+'_delta'] = metrics[a][name]-metrics[b][name]
        contrasts[f'{a}_minus_{b}'] = q
    return dict(counts=counts, contrasts=contrasts, mask_metrics=metrics)


def summarize(views):
    draws = np.random.default_rng(12773).integers(7, size=(3000, 7))
    sites = prior.base.SITES
    contrasts = {}
    for a, row in views[0]['contrasts'].items():
        contrasts[a] = {}
        for k in row:
            x = np.array([np.mean([v['contrasts'][a][k] for v in views if v['site'] == s]) for s in sites])
            contrasts[a][k] = dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                                    location_values=x.tolist())
    counts = {a: {k: sum(v['counts'][a][k] for v in views) for k in
                  ('selected', 'dual', 'risk_pass', 'undefined_easy')} for a in ARMS}
    for a in ARMS:
        counts[a]['max_spend_error'] = max(v['counts'][a]['max_spend_error'] for v in views)
    means = {a: {k: float(np.mean([np.mean([v['mask_metrics'][a][k] for v in views if v['site'] == s])
                                 for s in sites])) for k in views[0]['mask_metrics'][a]} for a in ARMS[1:]}
    return dict(contrasts=contrasts, counts=counts, mask_metrics=means)


def score(key, cells, cohorts):
    file = OUT/'freeze.json'
    prior.require_published(file)
    f = read(file)
    if f['registration'] != key or f['held_outcomes_loaded'] or set(f['cells']) != set(cells):
        raise ValueError('Require all weights and masks before accessing held outcomes')
    data = dict(prior.cohort_rows(cohorts, ('future', 'valid', 'recording', 'weight')))
    views = []
    for n, (name, c) in enumerate(cells.items(), 1):
        for site, ref in c['packets'].items():
            file = LOCAL/f'{name}_{site}.npz'
            if sha(file) != f['cells'][name]['packets'][site]:
                raise ValueError('Frozen causal predictions changed')
            with prior.prior.packet(prior.learn.LOCAL, ref) as z, np.load(file, allow_pickle=False) as p:
                views.append(dict(site=site, **measure(z, p, data[site], c['cut'])))
        beat('score_visibility', n)
    r = summarize(views)
    anchor = read(previous.OUT/'readout.json')['summary']['movement_matched']['counts']['supported_merit']
    if any(anchor[k] != r['counts']['support'][k] for k in ('selected', 'dual', 'risk_pass', 'undefined_easy')):
        raise ValueError('Original support result did not replay')
    write(OUT/'readout.json', dict(summary=r, registration=key, freeze_sha256=sha(OUT/'freeze.json'),
          directions=len(views), locations=7, seeds=prior.base.SEEDS, training_fits=f['fits'],
          optimizer_updates=f['optimizer_updates'], result_source='fresh_run', original_2pct=True,
          external_outcomes_read=False, independent_confirmation=False, deployment_changed=False), immutable=True)
    print(r, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    p.add_argument('--solve-seconds', type=float, default=30.)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cells, cohorts, f, masks = register()
        if a.phase == 'freeze':
            freeze(key, cells, cohorts, f, masks, a.solve_seconds)
        else:
            score(key, cells, cohorts)
