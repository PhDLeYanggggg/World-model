"""Matched directional feature ablation on OpenDD's exposed development folds."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 Python')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_pair_effect as previous
from src.world_model import m3w_signed_pair_geometry as geometry

study, base, np, torch, exp, shared = (previous.study, previous.base, previous.np, previous.torch,
                                     previous.exp, previous.shared)
prior, model = previous.prior, previous.model
SITES, SEEDS = previous.SITES, previous.SEEDS
OUT = previous.OUT.parent/'opendd_signed_pair_geometry_v1'
LOCAL = previous.LOCAL.parent/OUT.name
ARMS = ('magnitude', 'signed')


def register():
    old_key, cohorts, parent, slots = previous.register()
    old_freeze = previous.OUT/'freeze.json'
    f = study.read(old_freeze)
    if f['registration'] != old_key or not f['fold_scoring_not_run']:
        raise ValueError('Require the completed pair-effect experiment')
    registration = OUT/'registration.json'
    files = (Path(__file__), Path(geometry.__file__), Path(previous.__file__), old_freeze,
             prior.OUT/'freeze.json', exp.OUT/'freeze.json')
    base.immutable(registration, dict(bindings=[study.artifact(p) for p in files], arms=ARMS, seeds=SEEDS,
        folds=SITES, target='identical normalized co-observed pair effect', signed_features=geometry.WIDTH,
        same_width=True, same_initial_weights=True, same_batches=True, steps=previous.STEPS, workers=0,
        original_2pct=True, no_test_tuning=True, checkpoint_every=50, threads=4))
    return base.sha(registration), cohorts, parent, slots, f


def menu(h, cv, ols, bank, radii):
    paths, slots = shared.slot_menu(cv, bank, radii)
    options = shared.legal_assignments(slots[0][0], slots[1][0] if len(slots) == 2 else None)
    row = np.concatenate([s[0] for s in slots])
    arm = np.concatenate([s[1]+k*bank.shape[1] for k, s in enumerate(slots)])
    x, scale, valid, _ = shared.costs.features(h, cv, ols, paths)
    if not len(options) or not valid[row, arm].all():
        raise ValueError('Frozen slots have no legal common actions')
    delta = paths[row, arm]-cv[row]
    return paths[row, arm], x[row, arm], scale[row], row, delta, options


def inputs(h, cv, x, scale, row, delta, options):
    a, g, on = exp.slate.pack(x, h[row, -1], delta, scale, options)
    both = on.all(1)
    d = np.zeros((len(options), geometry.WIDTH), np.float32)
    if both.any():
        opt = options[both]
        r = row[opt]
        d[both] = geometry.directional(h[r], cv[r], delta[opt], scale[opt].mean(1))
    return a, g, d, on


def examples(ref, index, expected):
    h, q, cv, ols, bank = prior.source_inputs(ref)
    with np.load(study.checked(ref), allow_pickle=False) as z:
        target = z['future']
    valid = np.isfinite(target).all(axis=-1)
    rng = np.random.default_rng(80131+index)
    chunks, sampled = [], 0
    bounds = np.r_[0, np.flatnonzero(np.diff(q))+1, len(q)]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b-a < 3:
            continue
        hist, c, o, p = h[a:b], cv[a:b], ols[a:b], bank[a:b]
        peaks = np.linalg.norm(p-c[:, None], axis=-1).max(axis=-1)
        r, k = np.nonzero(peaks > 0)
        if len(np.unique(r)) < 2:
            continue
        i = rng.integers(len(r))
        j = rng.choice(np.flatnonzero(r != r[i]))
        _, x, scale, row, delta, opt = menu(hist, c, o, p, peaks[r[[i, j]], k[[i, j]]])
        if len(opt) > previous.MAX_PAIRS:
            opt = opt[rng.choice(len(opt), previous.MAX_PAIRS, replace=False)]
        r = row[opt]
        effect, support = model.observed_effect(c[r], delta[opt], target[a+r], valid[a+r])
        keep = support >= previous.MIN_COMMON
        sampled += len(opt)
        if keep.any():
            feat = inputs(hist, c, x, scale, row, delta, opt[keep])[:3]
            chunks.append((*feat, (effect[keep]/scale[opt[keep]].mean(1)).astype(np.float32)))
    actual = dict(sampled=sampled, observed=sum(len(v[3]) for v in chunks), queries_with_support=len(chunks))
    if actual != expected:
        raise ValueError(f'Training sample support differs: {actual} vs {expected}')
    if not chunks:
        return None
    data = tuple(torch.from_numpy(np.concatenate(v)) for v in zip(*chunks, strict=True))
    orig, counts = previous.examples(ref, index)
    for new, old in zip((data[0], data[1], data[3]), orig, strict=True):
        torch.testing.assert_close(new, old, rtol=0., atol=0.)
    return data


class Matched(torch.nn.Module):
    def __init__(self, slate):
        super().__init__()
        self.encoder = model.PairEffect(slate).eval()
        self.encoder.requires_grad_(False)
        first = torch.nn.Sequential(torch.nn.Linear(52+geometry.WIDTH, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1))
        with torch.no_grad():
            first[0].weight[:, 52:].zero_()
        second = torch.nn.Sequential(torch.nn.Linear(52+geometry.WIDTH, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1))
        second.load_state_dict(first.state_dict())
        self.heads = torch.nn.ModuleDict(dict(magnitude=first, signed=second))

    def effects(self, z, direction):
        return [self.heads[k](torch.cat((z, d), dim=-1)).flatten()
                for k, d in zip(ARMS, (torch.zeros_like(direction), direction), strict=True)]

    def unary(self, x, on):
        slate = self.encoder.slate
        return (slate.unary(slate.action(x)).squeeze(-1)*on).sum(-1)


def load_model(site, seed, parent):
    torch.manual_seed(seed)
    return Matched(previous.slate_for(parent, site, seed))


def fit(key, site, seed, parent, data):
    name = f'{site}_{seed}'
    net = load_model(site, seed, parent)
    donors = previous.supported_donors(data, site)
    enc, weight = previous.donor_encoding(net.encoder, [(data[s][0], data[s][1], data[s][3]) for s in donors])
    groups = [(z, data[s][2], y) for s, (z, y) in zip(donors, enc, strict=True)]
    ident = dict(registration=key, held=site, seed=seed, donors=donors, weight=weight,
                 unary=parent['fits'][name]['checkpoint']['sha256'])
    path = LOCAL/f'{name}.pt'
    opt = torch.optim.AdamW(net.heads.parameters(), lr=.002)
    rng = np.random.default_rng(seed+1981)
    step, trace = 0, []
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['identity'] != ident or state['step'] not in range(50, previous.STEPS+1, 50):
            raise ValueError('Checkpoint identity or progress differs')
        net.heads.load_state_dict(state['heads'])
        opt.load_state_dict(state['opt'])
        step, trace, rng.bit_generator.state = state['step'], state['trace'], state['rng']
    while step < previous.STEPS:
        n = rng.multinomial(previous.BATCH, np.full(len(donors), 1./len(donors)))
        indexes = [rng.integers(len(g[2]), size=k) for g, k in zip(groups, n, strict=True)]
        z, d, y = [torch.cat([g[i][at] for g, at in zip(groups, indexes, strict=True)]) for i in range(3)]
        loss = [torch.nn.functional.smooth_l1_loss(p, y, beta=.1) for p in net.effects(z, d)]
        opt.zero_grad(set_to_none=True)
        sum(loss).backward()
        for head in net.heads.values():
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        step += 1
        if step % 50 == 0:
            trace.append([float(l.detach()) for l in loss])
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident, step=step, trace=trace, rng=rng.bit_generator.state,
                            heads=net.heads.state_dict(), opt=opt.state_dict()), tmp)
            tmp.replace(path)
            heartbeat(name, step)
    return net.eval(), weight, dict(identity=ident, checkpoint=study.artifact(path), final_loss=trace[-1])


def heartbeat(fold, step):
    tmp = LOCAL/'heartbeat.pending'
    tmp.write_text(study.json.dumps(dict(pid=os.getpid(), fold=fold, step=step, total=previous.STEPS))+'\n')
    tmp.replace(LOCAL/'heartbeat.json')


@torch.inference_mode()
def predict(net, weight, h, q, cv, ols, bank, native, control):
    paths = {k: cv.copy() for k in ('additive', *ARMS)}
    stats = Counter()
    peak = np.linalg.norm(native-cv, axis=-1).max(axis=-1)
    for j in np.unique(q[peak > 0]):
        ix = np.flatnonzero(q == j)
        slots = peak[ix][peak[ix] > 0]
        pool, x, scale, row, delta, opt = menu(h[ix], cv[ix], ols[ix], bank[ix], slots)
        best = {k: (-np.inf, None) for k in paths}
        for at in range(0, len(opt), 8192):
            batch = opt[at:at+8192]
            a, g, d, on = (torch.from_numpy(v) for v in inputs(h[ix], cv[ix], x, scale, row, delta, batch))
            single = net.unary(a, on)
            pair = [single + weight*p*on.all(-1) for p in net.effects(net.encoder.encode(a, g), d)]
            for k, p in zip(paths, (single, *pair), strict=True):
                i = int(p.argmax())
                if not torch.isfinite(p[i]):
                    raise ValueError('Non-finite pair score')
                if float(p[i]) > best[k][0]:
                    best[k] = (float(p[i]), batch[i])
        for k, (_, at) in best.items():
            at = at[at >= 0]
            paths[k][ix[row[at]]] = pool[at]
            p = np.linalg.norm(paths[k][ix]-cv[ix], axis=-1).max(axis=-1)
            np.testing.assert_allclose(np.sort(p[p > 0]), np.sort(slots), atol=1e-9, rtol=1e-9)
        change = not np.allclose(paths['signed'][ix], paths['magnitude'][ix], atol=1e-9, rtol=1e-9)
        if change and len(slots) < 2:
            raise ValueError('Signed context changed a singleton decision')
        stats[f'queries_{len(slots)}_slots'] += 1
        stats['signed_differs_magnitude'] += int(change)
    np.testing.assert_allclose(paths.pop('additive'), control, atol=1e-9, rtol=1e-9)
    return dict(q=q, cv=cv, ols4=ols, matched_independent=control, **paths), dict(stats)


def freeze(key, cohorts, parent, slots, old):
    data, fits, decisions, forecasts = {}, {}, {}, {}
    for site in SITES:
        for s in (s for s in SITES if s != site and s not in data):
            data[s] = examples(cohorts[s], SITES.index(s), old['examples'][s])
        donors = {s: data[s] for s in SITES if s != site}
        h, q, cv, ols, bank = prior.source_inputs(cohorts[site])
        for seed in SEEDS:
            name = f'{site}_{seed}'
            net, weight, fits[name] = fit(key, site, seed, parent, donors)
            with np.load(study.checked(slots['forecasts'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], q)
                control = z['matched_independent']
            with np.load(study.checked(parent['forecasts'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], q)
                np.testing.assert_array_equal(z['cv'], cv)
                native = z['joint']
            p, decisions[name] = predict(net, weight, h, q, cv, ols, bank, native, control)
            forecasts[name] = exp.previous.save_bank(LOCAL/f'{name}.npz', p)
            print(name, decisions[name], fits[name]['final_loss'], flush=True)
    base.immutable(OUT/'freeze.json', dict(registration=key, fits=fits, decisions=decisions,
        forecasts=forecasts, fresh_heads=2*len(fits), optimizer_steps_per_arm=len(fits)*previous.STEPS,
        held_policy_scoring_not_run=True))


@torch.inference_mode()
def probe(parent, receipt, site, seed, examples):
    if examples is None:
        return None
    net = load_model(site, seed, parent).eval()
    old = torch.load(study.checked(receipt['checkpoint']), weights_only=True, map_location='cpu')
    if old['identity'] != {**receipt['identity'], 'donors': tuple(receipt['identity']['donors'])}:
        raise ValueError('Unverified probe checkpoint')
    net.heads.load_state_dict(old['heads'])
    x, g, d, y = examples
    pred = net.effects(net.encoder.encode(x, g), d)
    null = float(y.square().mean())
    rmse = {k: float((p-y).square().mean())**.5 for k, p in zip(ARMS, pred, strict=True)}
    return dict(zero_rmse=null**.5, **rmse)


def score(key, cohorts, parent, old):
    frozen = OUT/'freeze.json'
    f = study.read(frozen)
    if f['registration'] != key or not f['held_policy_scoring_not_run'] or len(f['fits']) != 21:
        raise ValueError('Freeze all held policies before scoring')
    data = exp.loader(cohorts)
    result = {}
    for site, d in data.items():
        probe_rows = {}
        evidence = []
        sample = examples(cohorts[site], SITES.index(site), old['examples'][site])
        cut = np.quantile(np.concatenate([data[s]['e'] for s in SITES if s != site]), [.25, .75])
        for seed in SEEDS:
            name = f'{site}_{seed}'
            probe_rows[str(seed)] = probe(parent, f['fits'][name], site, seed, sample)
            with np.load(study.checked(f['forecasts'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], d['q'])
                p = {k: z[k] for k in ('cv', 'ols4', 'matched_independent', *ARMS)}
            for rec in np.unique(d['rec']):
                at = d['rec'] == rec
                q = np.unique(d['q'][at], return_inverse=True)[1]
                y, v = d['y'][at], d['v'][at]
                paths = {k: z[at] for k, z in p.items() if k != 'signed'}
                paths['joint'] = p['signed'][at]
                m = exp.score_record(paths, y, v, q, cut)
                if m['direct']['matched_independent'] is not None:
                    _, lower = study.paired(paths['magnitude'], paths['joint'], y, v)
                    m['signed_vs_magnitude'] = float(100.*((1./np.bincount(q)[q]) @ lower.sum(1))/m['reference_mass'])
                evidence.append(m)
        def counts(k):
            a = [m['arms'][k] for m in evidence]
            return dict(edits=sum(r['risk']['selected'] for r in a), risk=sum(r['risk']['supported'] for r in a),
                        dual=sum(r['dual_pass'] for r in a))
        contrasts = [m['signed_vs_magnitude'] for m in evidence if 'signed_vs_magnitude' in m]
        additive = [m['direct']['matched_independent'][1] for m in evidence if 'signed_vs_magnitude' in m]
        geometry_gain = [m['pair_relative_error_px']['magnitude']-m['pair_relative_error_px']['joint']
                         for m in evidence if 'signed_vs_magnitude' in m
                         and all(m['pair_relative_error_px'][k] is not None for k in ('joint', 'magnitude'))]
        mean = lambda x: float(np.mean(x)) if x else None
        result[site] = dict(views=len(evidence), estimable=len(contrasts), probes=probe_rows,
            signed=counts('joint'), magnitude=counts('magnitude'), additive=counts('matched_independent'),
            lower_signed_vs_magnitude_pp=mean(contrasts), lower_signed_vs_additive_pp=mean(additive),
            observed_pair_error_reduction_native=mean(geometry_gain))
    def interval(k):
        x = np.asarray([g[k] for g in result.values() if g[k] is not None])
        draw = np.random.default_rng(3183).choice(x, size=(3000, len(x))).mean(1)
        return dict(sites=len(x), mean=float(x.mean()), ci95=np.quantile(draw, [.025, .975]).tolist())
    metrics = dict(result_source='fresh_run', freeze=study.artifact(frozen), groups=result,
        totals=dict(sum(map(Counter, f['decisions'].values()), Counter())),
        contrasts={k: interval(k) for k in ('lower_signed_vs_magnitude_pp', 'lower_signed_vs_additive_pp',
                                            'observed_pair_error_reduction_native')},
        independent_confirmation=False, original_2pct=True, reserved_outcomes_opened=False, threshold_tuning=False)
    base.immutable(OUT/'readout.json', metrics)
    print(metrics['contrasts'], metrics['totals'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('freeze', 'score'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, slots, old = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, parent, slots, old)
        else:
            score(key, cohorts, parent, old)
