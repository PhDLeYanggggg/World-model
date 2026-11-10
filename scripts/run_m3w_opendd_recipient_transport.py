"""Compare mean and worst-source swap learning with frozen vehicle paths."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import subprocess
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_direction_context as prior
from src.world_model import m3w_recipient_transport as model

np, torch, base, pair = prior.np, prior.torch, prior.base, prior.pair
sha, read, write = prior.sha, prior.read, prior.write
ref = prior.prior.prior.ref
OUT = prior.OUT.parent/'opendd_recipient_transport_v1'
LOCAL = prior.LOCAL.parent/OUT.name
ARMS = ('original', 'mean', 'robust')
CONTRASTS = (('mean', 'original'), ('robust', 'original'), ('robust', 'mean'))
STEPS, SAVE, BATCH_PER_DONOR = 300, 50, 32


def register():
    source, _, original, archived, _ = ref.settings()
    paths = (Path(__file__), Path(model.__file__), Path(prior.__file__),
             Path(prior.model.__file__), prior.geometry.OUT/'manifest.json',
             prior.study.OUT/'freeze.json', prior.study.OUT/'readout.json', OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(files={str(k.resolve().relative_to(base.prior.ROOT)): sha(k) for k in paths},
        sites=base.SITES, seeds=base.SEEDS, steps=STEPS, batch_per_donor=BATCH_PER_DONOR,
        workers=0, threads=4, checkpoint_every=SAVE, original_2pct=True,
        external_outcomes_read=False, confirmation=False), immutable=True)
    return sha(p), source, original, archived


def beat(phase, **fields):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, **fields))
    print(phase, fields, flush=True)


def check_cell(cell):
    excluded = set(cell['packets'])
    if len(excluded) != 2 or set(cell['fit_sites']) != set(base.SITES)-excluded:
        raise ValueError('Donor packet generator did not exclude both locations')


def geometry(packet, d):
    anchor = pair.prediction(packet, 'additive').astype(float)
    cv = packet['cv'].astype(float)
    bounds = np.r_[0, np.cumsum(d['counts'])]
    allowed = set(zip(bounds[:-1], bounds[1:], strict=True))
    return cv, anchor, allowed


def menus(packet, d, cv, anchor, allowed):
    for (lo, hi), recipients in zip(packet['bounds'], packet['additive_rows'], strict=True):
        if (lo, hi) not in allowed:
            raise ValueError('Forecast and geometry do not bind the same query')
        q = slice(lo, hi)
        drop, add, new, x = model.proposals(d['history'][q], d['offsets'][q], d['x'][q],
                                           cv[q], anchor[q], recipients-lo)
        yield drop+lo, add+lo, new, x


def donors_for(held, seed, original, features):
    data, lineage = {}, {}
    for name, cell in original['cells'].items():
        if cell['seed'] != seed or held not in cell['packets']:
            continue
        check_cell(cell)
        donor, = set(cell['packets'])-{held}
        if donor in data:
            raise ValueError('Duplicate donor packet')
        f = features[donor]
        with np.load(f['file'], allow_pickle=False) as cohort:
            y, valid = cohort['future'], cohort['valid']
        with ref.prior.packet(prior.study.LOCAL, cell['packets'][donor]) as p:
            cv, anchor, bounds = geometry(p, f)
            rows = [(x, model.targets(cv, anchor, y, valid, a, b, new))
                    for a, b, new, x in menus(p, f, cv, anchor, bounds) if len(x)]
        if not rows:
            raise ValueError(f'Empty train candidate set: {donor}')
        data[donor] = tuple(torch.from_numpy(np.concatenate(xs)) for xs in zip(*rows, strict=True))
        lineage[donor] = dict(pair_cell=name, packet=cell['packets'][donor]['sha256'],
                              cohort=sha(f['file']), rows=len(data[donor][0]))
    if set(data) != set(base.SITES)-{held}:
        raise ValueError('Wrong outer training sites')
    return [data[s] for s in sorted(data)], lineage


def fit(identity, data):
    torch.manual_seed(identity['seed'])
    rng = np.random.default_rng([identity['seed'], base.SITES.index(identity['held'])])
    net = model.network(data[0][0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f"{identity['held']}_{identity['seed']}_{identity['arm']}.pt"
    start = 0
    if p.exists():
        z = torch.load(p, map_location='cpu', weights_only=True)
        if z['identity'] != identity or not 0 <= z['step'] <= STEPS:
            raise ValueError('Changed training checkpoint')
        start = z['step']
        net.load_state_dict(z['model'])
        opt.load_state_dict(z['optimizer'])
        rng.bit_generator.state = z['rng']
    for step in range(start+1, STEPS+1):
        parts = [(x[i], y[i]) for x, y in data
                 for i in (rng.integers(len(x), size=BATCH_PER_DONOR),)]
        x, y = (torch.stack(v) for v in zip(*parts, strict=True))
        loss = model.objective(net(x.flatten(0, 1)).reshape(y.shape), y, identity['arm'] == 'robust')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            tmp = p.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, rng=rng.bit_generator.state,
                            model=net.state_dict(), optimizer=opt.state_dict()), tmp)
            tmp.replace(p)
            beat('fit', held=identity['held'], seed=identity['seed'], arm=identity['arm'],
                 step=step, loss=float(loss.detach()))
    return net.eval(), dict(file=p.name, sha256=sha(p), step=STEPS)


@torch.inference_mode()
def propose(packet, d, heads):
    cv, anchor, bounds = geometry(packet, d)
    out = {k: [] for k in heads}
    for _, _, _, x in menus(packet, d, cv, anchor, bounds):
        for k, net in heads.items():
            out[k].append(model.decision(net(torch.from_numpy(x)).numpy()))
    return {k: np.array(v, dtype=np.int32) for k, v in out.items()}


def freeze(key, source, original):
    features = prior.load_features(source)
    cells = {}
    for held in base.SITES:
        for seed in base.SEEDS:
            name = f'{held}_{seed}'
            dest = LOCAL/f'{name}_choices.json'
            if not dest.exists():
                data, lineage = donors_for(held, seed, original, features)
                heads, weights, packets = {}, {}, {}
                for arm in ARMS[1:]:
                    heads[arm], weights[arm] = fit(dict(registration=key, held=held, seed=seed,
                                                         arm=arm, donors=lineage), data)
                for k, c in original['cells'].items():
                    if c['seed'] != seed or held not in c['packets']:
                        continue
                    check_cell(c)
                    orig = c['packets'][held]
                    with ref.prior.packet(prior.study.LOCAL, orig) as packet:
                        out = propose(packet, features[held], heads)
                    p = LOCAL/f'{name}_{k}.npz'
                    packets[k] = dict(file=p.name, original=orig['sha256'],
                                      sha256=base.parent.indices.save_npz(p, out))
                write(dest, dict(registration=key, fit_sites=sorted(lineage), donors=lineage,
                                 weights=weights, packets=packets), immutable=True)
            c = read(dest)
            if (c['registration'] != key or set(c['fit_sites']) != set(base.SITES)-{held}
                    or len(c['packets']) != 6 or any(sha(LOCAL/v['file']) != v['sha256']
                    for v in (*c['weights'].values(), *c['packets'].values()))):
                raise ValueError('Changed frozen decisions')
            cells[name] = c
            beat('freeze', completed=len(cells), total=21)
    write(OUT/'freeze.json', dict(registration=key, cells=cells, external_outcomes_read=False,
                                 held_decisions_frozen=True), immutable=True)


def require_published(path):
    root = base.prior.ROOT
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=root).split()[0].decode()
    existing = subprocess.check_output(['git', 'show', f'{remote}:{path.relative_to(root)}'], cwd=root)
    if existing != path.read_bytes():
        raise ValueError('Publish the exact forecast freeze before scoring')


def evaluate(packet, f, d, cut, z):
    cv, anchor, bounds = geometry(packet, f)
    y, v, w = (d[k] for k in ('future', 'valid', 'weight'))
    pred = {k: anchor.copy() for k in ARMS}
    n = len(packet['bounds'])
    if any(z[k].shape != (n,) or not np.issubdtype(z[k].dtype, np.integer) for k in ARMS[1:]):
        raise ValueError('Invalid action roster')
    rel = {f'{a}_minus_{b}': np.zeros(2) for a, b in CONTRASTS}
    stats = Counter(views=n)
    for i, (drop, add, new, _) in enumerate(menus(packet, f, cv, anchor, bounds)):
        stats['has_swap'] += bool(len(drop))
        values = model.targets(cv, anchor, y, v, drop, add, new)
        stats['positive_hindsight'] += bool(len(values) and values.max() > 0.)
        for k in ARMS[1:]:
            j = z[k][i]
            if j < -1 or j >= len(drop):
                raise ValueError('Frozen choice no longer matches menu')
            if j >= 0:
                pred[k][drop[j]], pred[k][add[j]] = cv[drop[j]], new[j]
                stats[f'{k}_swaps'] += 1
                stats[f'{k}_positive'] += int(values[j] > 0.)
        lo, hi = packet['bounds'][i]
        q = slice(lo, hi)
        for a, b in CONTRASTS:
            if not np.array_equal(pred[a][q], pred[b][q]):
                rel[f'{a}_minus_{b}'] += prior.pair.pair_reduction(pred[a][None, q], pred[b][q], y[q], v[q])[0]
    arms = {}
    for k, p in pred.items():
        m = pair.evaluate(p, packet, d, cut)
        arms[k] = {a: int(m[a].sum()) for a in ('risk', 'dual', 'missing')}
        arms[k]['p95'] = m['p95']
    use = np.isin(d['recording'], packet['records'])
    mass = float(w[use] @ np.where(v[use], np.linalg.norm(cv[use]-y[use], axis=-1), 0.).sum(1))
    effects = {}
    for a, b in CONTRASTS:
        name = f'{a}_minus_{b}'
        seen, lower = base.parent.paired(pred[b][use], pred[a][use], y[use], v[use])
        effects[name] = dict(known_pp=float(100.*(w[use] @ seen.sum(1))/mass),
                             lower_pp=float(100.*(w[use] @ lower.sum(1))/mass),
                             relative_lower=float(rel[name][1]/n),
                             dual_pp=float(100.*(arms[a]['dual']-arms[b]['dual'])/n))
    return dict(counts=dict(stats), arms=arms, effects=effects)


def summarize(rows):
    draw = np.random.default_rng(195).integers(len(base.SITES), size=(3000, len(base.SITES)))
    groups = {s: [r for r in rows if r['identity']['held'] == s] for s in base.SITES}

    def ci(key, field):
        x = np.array([np.mean([np.mean([r['effects'][key][field] for r in group if r['identity']['seed'] == seed])
                               for seed in base.SEEDS]) for group in groups.values()])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(groups, x.tolist(), strict=True)))

    return dict(counts=dict(sum((Counter(r['counts']) for r in rows), Counter())),
        arms={a: {k: sum(r['arms'][a][k] for r in rows) for k in ('risk', 'dual', 'missing')} for a in ARMS},
        paired={k: {f: ci(k, f) for f in v} for k, v in rows[0]['effects'].items()})


def score(key, source, original, archived):
    p = OUT/'freeze.json'
    require_published(p)
    f = read(p)
    if f['registration'] != key or f['external_outcomes_read'] or not f['held_decisions_frozen']:
        raise ValueError('Require the checked pre-score freeze')
    features, data = prior.load_features(source), base.parent.load(source)
    rows = []
    for name, cell in original['cells'].items():
        check_cell(cell)
        for held, orig in cell['packets'].items():
            seed = cell['seed']
            c = f['cells'][f'{held}_{seed}']
            r = c['packets'][name]
            action = LOCAL/r['file']
            if r['original'] != orig['sha256'] or sha(action) != r['sha256'] or held in c['fit_sites']:
                raise ValueError('Changed frozen action or training sites')
            identity = dict(registration=key, held=held, seed=seed, original=orig['sha256'], action=r['sha256'])
            receipt = LOCAL/f'{name}_{held}_score.json'
            if not receipt.exists():
                with ref.prior.packet(prior.study.LOCAL, orig) as packet, np.load(action, allow_pickle=False) as z:
                    value = evaluate(packet, features[held], data[held], cell['cut'], z)
                write(receipt, dict(identity=identity, **value), immutable=True)
            row = read(receipt)
            if row['identity'] != identity:
                raise ValueError('Mismatched resumed score')
            rows.append(row)
        beat('score', completed=len(rows), total=126)
    s = summarize(rows)
    if (len(rows), s['counts']['views'], s['arms']['original']['dual']) != (
            126, 8874, archived['summary']['additive']['matched_dual']):
        raise ValueError('Original matched control changed')
    write(OUT/'readout.json', dict(registration=key, freeze=sha(p), summary=s, by_direction=rows,
        original_2pct=True, external_outcomes_read=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False), immutable=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg, source, original, archived = register()
        if args.phase == 'freeze':
            freeze(reg, source, original)
        else:
            score(reg, source, original, archived)
