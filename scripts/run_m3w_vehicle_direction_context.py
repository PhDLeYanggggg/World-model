"""Freeze matched context models and angle choices before OpenDD scoring."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_vehicle_single_rotation as prior
from scripts import build_m3w_opendd_vehicle_observed_geometry as geometry
from src.world_model import m3w_vehicle_direction_context as model

np, torch, base, study, pair = prior.np, prior.torch, prior.base, prior.study, prior.pair
sha, read, write = prior.sha, prior.old.read, prior.write
OUT = prior.OUT.parent/'opendd_vehicle_direction_context_v1'
LOCAL = prior.LOCAL.parent/OUT.name
ARMS = ('original', 'centered_proxy', 'spaced_proxy', 'own_model', 'neighbor_model', 'own_at_neighbor')
CONTRASTS = tuple((a, 'original') for a in ARMS[1:])+(
    ('spaced_proxy', 'centered_proxy'), ('neighbor_model', 'own_model'), ('neighbor_model', 'own_at_neighbor'))
STEPS, BATCH = 200, 256


def register():
    source, _, frozen, original, _ = prior.prior.ref.settings()
    previous = read(prior.OUT/'readout.json')
    paths = (Path(__file__), Path(model.__file__), Path(model.residual.__file__),
             Path(prior.__file__), Path(prior.model.__file__), Path(prior.kinematic.__file__),
             Path(prior.prior.model.__file__), geometry.OUT/'manifest.json', base.parent.OUT/'cohorts.json',
             study.OUT/'freeze.json', prior.OUT/'readout.json', OUT/'protocol.md')
    path = OUT/'registration.json'
    write(path, dict(files={str(f.resolve().relative_to(base.prior.ROOT)): sha(f) for f in paths},
        sites=base.SITES, seeds=base.SEEDS, steps=STEPS, batch=BATCH, checkpoint_every=50,
        workers=0, torch_threads=4, model_selection='fixed final update, no threshold tuning',
        original_2pct=True, independent_confirmation=False, external_outcomes_read=False), immutable=True)
    return sha(path), source, frozen, original, previous


def load_features(source):
    cohorts = read(base.parent.OUT/'cohorts.json')
    geo = read(geometry.OUT/'manifest.json')
    if cohorts['registration'] != source or cohorts['songdo_read'] or not (
            geo['past_coordinates_only'] and geo['no_labels_exported']):
        raise ValueError('Require registered cohorts and observed geometry')
    result = {}
    for site in base.SITES:
        c, g = cohorts['sites'][site], geo['sites'][site]
        file = base.parent.LOCAL/f'{site}_cohort.npz'
        gf = geometry.LOCAL/f'{site}.npz'
        if c['sha256'] != g['cohort_sha256'] or sha(file) != c['sha256'] or sha(gf) != g['sha256']:
            raise ValueError('Misaligned observed geometry')
        with np.load(file, allow_pickle=False) as z, np.load(gf, allow_pickle=False) as o:
            h, counts, offsets = z['history'], z['query_counts'], o['observed_current_offsets']
        x, cv, rotation, scale = model.design(h, offsets, counts)
        result[site] = dict(file=file, history=h, offsets=offsets, counts=counts,
                            x=x, cv=cv, rotation=rotation, scale=scale)
    return result


def training_data(data, donors, excluded):
    if set(donors) != set(data)-set(excluded):
        raise ValueError('Donor and held/calibration roles overlap')
    out = []
    for site in donors:
        d = data[site]
        with np.load(d['file'], allow_pickle=False) as z:
            y, valid = z['future'], z['valid']
        target = (y @ d['rotation']) / d['scale'][:, None, None]
        out.append(tuple(torch.from_numpy(v) for v in (d['x'], d['cv'], target.astype(np.float32), valid)))
    return out


def beat(phase, **fields):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, **fields))


def fit(ident, donor):
    torch.manual_seed(ident['seed'])
    rng = np.random.default_rng(ident['seed'])
    net = model.residual.motion.network(29)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    path = LOCAL/f"{ident['cell']}_{ident['arm']}.pt"
    begin = 0
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['identity'] != ident or not 0 <= state['step'] <= STEPS:
            raise ValueError('Checkpoint does not bind the registered fit')
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        rng.bit_generator.state = state['rng']
        begin = state['step']
    for step in range(begin+1, STEPS+1):
        # Equal source mass, including the same extra donor and row draws for both arms.
        counts = np.bincount(rng.permutation(len(donor))[:BATCH % len(donor)], minlength=len(donor))
        counts += BATCH//len(donor)
        batch = [tuple(v[ix] for v in rows) for rows, ix in (
            (rows, rng.integers(len(rows[0]), size=int(n))) for rows, n in zip(donor, counts, strict=True))]
        x, cv, y, valid = (torch.cat(v) for v in zip(*batch, strict=True))
        if ident['arm'] == 'own':
            x[:, 17:] = 0.
        loss = model.residual.loss(model.residual.predict(net, x, cv), cv, y, valid, completion=True)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite motion loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % 50 == 0:
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident, step=step, rng=rng.bit_generator.state,
                            model=net.state_dict(), opt=opt.state_dict(), loss=float(loss.detach())), tmp)
            tmp.replace(path)
            beat('training', cell=ident['cell'], arm=ident['arm'], step=step)
    return net.eval(), dict(file=path.name, sha256=sha(path))


@torch.inference_mode()
def motion(net, d, own):
    x = torch.from_numpy(model.ablate(d['x']) if own else d['x'])
    cv = torch.from_numpy(d['cv'])
    chunks = [model.residual.predict(net, a, b).numpy() for a, b in zip(x.split(4096), cv.split(4096), strict=True)]
    return (np.concatenate(chunks) @ d['rotation'].transpose(0, 2, 1))*d['scale'][:, None, None]


def choose(packet, d, forecasts):
    h, offsets = d['history'], d['offsets']
    cv, ols = packet['cv'], packet['ols4']
    ends = np.r_[0, np.cumsum(d['counts'])]
    allowed = set(zip(ends[:-1], ends[1:], strict=True))
    scale = np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1).clip(.05)
    b = prior.kinematic.paths(h, cv, scale)
    proxies = np.stack((ols, cv+b[..., 2]*scale[:, None, None], cv+b[..., 3]*scale[:, None, None]))
    anchor = pair.prediction(packet, 'additive')
    out = np.full((len(ARMS), len(packet['bounds']), 2), -1, dtype=np.int16)
    for k, (lo, hi) in enumerate(packet['bounds']):
        q = slice(lo, hi)
        rows = packet['additive_rows'][k]
        if (lo, hi) not in allowed or len(set(rows)) != 2 or not np.all((lo <= rows) & (rows < hi)):
            raise ValueError('Misaligned query or recipients')
        bank = prior.prior.model.candidates(cv[rows], anchor[rows])
        center = h[q]
        shared = center+offsets[q, None]

        def gains(ref, history):
            return prior.model.proxy_gains(bank, anchor[q], ref, history, rows-lo, scale[q])

        own, peer = (gains(forecasts[a][None, q], shared) for a in ('own', 'neighbor'))
        p = prior.model.choose(*peer)['interaction']
        choices = (prior.model.choose(*gains(proxies[:, q], center))['interaction'],
                   prior.model.choose(*gains(proxies[:, q], shared))['interaction'],
                   prior.model.choose(*own)['interaction'], p, model.matched(own[0], p))
        for a, value in enumerate(choices, start=1):
            if value is not None:
                out[a, k] = value
    return out


def freeze(key, data, frozen):
    cells = {}
    for name, c in frozen['cells'].items():
        receipt = LOCAL/f'{name}.json'
        excluded = sorted(c['packets'])
        if not receipt.exists():
            donor = training_data(data, c['fit_sites'], excluded)
            models, ckpt = {}, {}
            for a in ('own', 'neighbor'):
                models[a], ckpt[a] = fit(dict(registration=key, cell=name, seed=c['seed'],
                    arm=a, donors=c['fit_sites'], excluded=excluded), donor)
            files = {}
            for s, r in c['packets'].items():
                forecasts = {a: motion(net, data[s], a == 'own') for a, net in models.items()}
                with prior.prior.ref.prior.packet(study.LOCAL, r) as packet:
                    arrays = dict(choices=choose(packet, data[s], forecasts), **forecasts)
                path = LOCAL/f'{name}_{s}.npz'
                files[s] = dict(file=path.name, sha256=base.parent.indices.save_npz(path, arrays), original=r['sha256'])
            write(receipt, dict(registration=key, donors=c['fit_sites'], checkpoints=ckpt, sites=files), immutable=True)
        cell = read(receipt)
        if cell['registration'] != key or any(sha(LOCAL/z['file']) != z['sha256']
                for z in (*cell['sites'].values(), *cell['checkpoints'].values())):
            raise ValueError('Changed frozen choice or checkpoint')
        cells[name] = cell
        beat('freeze', cell=name, completed=len(cells), total=len(frozen['cells']))
        if len(cells) % 7 == 0:
            print('Frozen:', len(cells), '/', len(frozen['cells']), flush=True)
    write(OUT/'freeze.json', dict(registration=key, cells=cells, frozen_before_scoring=True), immutable=True)


def evaluate(packet, d, cut, z):
    cv, y, valid, weight = packet['cv'], d['future'], d['valid'], d['weight']
    anchor = pair.prediction(packet, 'additive')
    paths = {k: anchor.copy() for k in ARMS}
    cnt, rel = Counter(), {f'{a}_minus_{b}': np.zeros(2) for a, b in CONTRASTS}
    nrot = dict.fromkeys(ARMS, 0)
    for k, (lo, hi) in enumerate(packet['bounds']):
        q = slice(lo, hi)
        rows = packet['additive_rows'][k]
        bank = prior.prior.model.candidates(cv[rows], anchor[rows])
        choices = z['choices'][:, k]
        if choices[-1, 0] != choices[-2, 0]:
            raise ValueError('Matched comparison changed rotating actor')
        for arm, (i, angle) in zip(ARMS, choices, strict=True):
            if i >= 0:
                paths[arm][rows[i]] = bank[i, angle]
                nrot[arm] += 1
        cnt.update(views=1, known_pair_query=int((valid[q].sum(0) >= 2).any()))
        for a, b in CONTRASTS:
            rel[f'{a}_minus_{b}'] += prior.prior.pair_reduction(paths[a][None, q], paths[b][q], y[q], valid[q])[0]
    result = {}
    for a, p in paths.items():
        x = pair.evaluate(p, packet, d, cut)
        result[a] = dict(dual=int(x['dual'].sum()), risk=int(x['risk'].sum()), missing=int(x['missing'].sum()),
                         rotated_views=nrot[a])
    if len({v['missing'] for v in result.values()}) != 1:
        raise ValueError('Risk support changed')
    roster = np.isin(d['recording'], packet['records'])
    w, mask, truth = weight[roster], valid[roster], y[roster]
    mass = w @ np.where(mask, np.linalg.norm(cv[roster]-truth, axis=-1), 0.).sum(1)
    e = {}
    for a, b in CONTRASTS:
        key = f'{a}_minus_{b}'
        known, lower = base.parent.paired(paths[b][roster], paths[a][roster], truth, mask)
        e[key] = dict(observed_pp=float(100.*(w @ known.sum(1))/mass), lower_pp=float(100.*(w @ lower.sum(1))/mass),
                      pair_known=float(rel[key][0]/max(1, cnt['known_pair_query'])), pair_lower=float(rel[key][1]/cnt['views']))
    available = {a: float((w @ np.where(mask, np.linalg.norm(p[roster]-truth, axis=-1), 0.).sum(1))
                           / (w @ mask.sum(1))) for a, p in (('cv', cv), ('own', z['own']), ('neighbor', z['neighbor']))}
    return dict(counts=dict(cnt), arms=result, contrasts=e, available_ade=available)


def score(key, source, original, previous, frozen):
    path = OUT/'freeze.json'
    f = read(path)
    if f['registration'] != key or not f['frozen_before_scoring'] or set(f['cells']) != set(frozen['cells']):
        raise ValueError('Require a complete pre-score freeze')
    data, rows = base.parent.load(source), []
    for name, c in frozen['cells'].items():
        for s, r in c['packets'].items():
            item = f['cells'][name]['sites'][s]
            p = LOCAL/item['file']
            if sha(p) != item['sha256'] or r['sha256'] != item['original']:
                raise ValueError('Changed frozen inputs')
            dest = LOCAL/f'{name}_{s}_score.json'
            ident = dict(freeze=sha(path), packet=item['sha256'], site=s, seed=c['seed'])
            if not dest.exists():
                with prior.prior.ref.prior.packet(study.LOCAL, r) as packet, np.load(p, allow_pickle=False) as z:
                    record = evaluate(packet, data[s], c['cut'], {k: z[k] for k in z.files})
                write(dest, dict(identity=ident, **record), immutable=True)
            record = read(dest)
            if record['identity'] != ident:
                raise ValueError('Changed score inputs')
            rows.append(record)
        beat('score', cell=name, completed=len(rows), total=126)
    counts = {a: {k: sum(r['arms'][a][k] for r in rows) for k in rows[0]['arms'][a]} for a in ARMS}
    if (len(rows), sum(r['counts']['views'] for r in rows), counts['original']['dual']) != (
            126, 8874, original['summary']['additive']['matched_dual']):
        raise ValueError('Changed recording views')
    if counts['centered_proxy'] != previous['summary']['arms']['interaction']:
        raise ValueError('Centered proxy must reproduce the earlier run')
    groups = [[r for r in rows if r['identity']['site'] == s] for s in base.SITES]
    draws = np.random.default_rng(2917).integers(len(groups), size=(3000, len(groups)))

    def interval(get):
        x = np.array([np.mean([np.mean([get(r) for r in g if r['identity']['seed'] == seed])
                              for seed in base.SEEDS]) for g in groups])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(base.SITES, x.tolist(), strict=True)))

    paired = {k: {m: interval(lambda r: r['contrasts'][k][m]) for m in a}
              for k, a in rows[0]['contrasts'].items()}
    ade = {k: interval(lambda r: r['available_ade'][k]) for k in ('cv', 'own', 'neighbor')}
    ade['own_minus_neighbor'] = interval(lambda r: r['available_ade']['own']-r['available_ade']['neighbor'])
    result = dict(result_source='fresh_run', registration=key, freeze=sha(path), unique_recordings=493, views=8874,
        counts=counts, paired=paired, available_ade=ade, fit_models=2*len(f['cells']), updates=2*len(f['cells'])*STEPS,
        original_2pct=True, external_outcomes_read=False, independent_confirmation=False, deployment_changed=False)
    write(OUT/'readout.json', result, immutable=True)
    beat('complete', completed=len(rows), total=126)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    phase = parser.parse_args().phase
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use the native arm64 .venv-pytorch interpreter')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, frozen, original, previous = register()
        if phase == 'freeze':
            freeze(key, load_features(source), frozen)
        else:
            score(key, source, original, previous, frozen)
