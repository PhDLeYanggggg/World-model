"""Train and freeze incremental angle value models on the exposed vehicle cohort."""

import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_direction_context as context
from src.world_model import m3w_vehicle_action_value as model

np, torch, base, study, pair = context.np, context.torch, context.base, context.study, context.pair
old = context.prior.old
sha, read, write = context.sha, context.read, context.write
OUT = context.OUT.parent/'opendd_vehicle_action_value_v1'
LOCAL = context.LOCAL.parent/OUT.name
ARMS = ('original', 'ego', 'context', 'joint', 'ego_at_context', 'constant_at_context', 'joint_at_context')
CAL = ('context', 'joint')
SCORE_ARMS = ARMS+tuple(f'{a}_calibrated' for a in CAL)
PAIRS = tuple((k, 'original') for k in SCORE_ARMS[1:])+(('context', 'ego'), ('context', 'ego_at_context'),
    ('context', 'constant_at_context'), ('joint_at_context', 'context'))
STEPS, DRAWS = 200, 3


def register():
    source, _, f, previous, _ = context.prior.prior.ref.settings()
    bank = base.OUT/'freeze.json'
    forecasts = read(bank)
    if not forecasts['held_choices_frozen'] or forecasts['songdo_read'] or len(forecasts['triple_checkpoints']) != 105:
        raise ValueError('Require the frozen three-site-excluded donor forecasts')
    prior = context.OUT/'readout.json'
    a = read(prior)
    if a['external_outcomes_read'] or a['freeze'] != sha(context.OUT/'freeze.json') or a['views'] != 8874:
        raise ValueError('Require the frozen, completed comparison cohort')
    files = (Path(__file__), Path(model.__file__), Path(context.__file__), Path(context.model.__file__),
             Path(old.__file__), Path(base.load_net.__code__.co_filename), Path(study.__file__),
             study.OUT/'freeze.json', bank, prior, OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(inputs={str(q.resolve().relative_to(base.prior.ROOT)): sha(q) for q in files},
        arms=SCORE_ARMS, train_steps=STEPS, samples_per_donor_recording=DRAWS, seeds=base.SEEDS,
        num_workers=0, torch_threads=4, original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(p), source, forecasts, f, previous


@lru_cache(maxsize=7)
def labels(file):
    with np.load(file, allow_pickle=False) as z:
        return {k: z[k] for k in ('future', 'valid', 'recording', 'weight')}


def inputs(cv, anchor, selected, d, q):
    return model.design(cv, anchor, selected, *(d[k][q] for k in ('x', 'rotation', 'scale', 'offsets')))


def donor_batch(d, donor, seed, excluded, bank):
    reject = sorted([*excluded, donor])
    if len(set(reject)) != 3:
        raise ValueError('Invalid donor forecast roles')
    tag = base.triple_id(reject, seed)
    net = base.load_net(base.LOCAL/f'{tag}.pt', bank['triple_checkpoints'][tag],
                        old.motion.motion.network(17), reject)
    y = labels(d['file'])
    data = dict(history=d['history'], query_counts=d['counts'], recording=y['recording'],
        tensors=tuple(torch.from_numpy(d[k][..., :17]) if k == 'x' else torch.from_numpy(d[k]) for k in ('x', 'cv')),
        rotation=d['rotation'], scale=d['scale'])
    m = study.menu(data, net)
    rng = np.random.default_rng([seed, *(base.SITES.index(t) for t in reject), base.SITES.index(donor)])
    blocks = []
    for lo, hi in m['slots'].values():
        q = slice(lo, hi)
        eligible = m['peak'][q] > 0.
        rows = np.flatnonzero(eligible.any(axis=1))
        cv = m['cv'][q]
        for _ in range(DRAWS):
            selected = rng.choice(rows, size=2, replace=False)
            family = [rng.choice(np.flatnonzero(eligible[i])) for i in selected]
            anchor = cv.copy()
            anchor[selected] = m['paths'][lo+selected, family]
            feat, angles = inputs(cv, anchor, selected, d, q)
            target = model.targets(anchor, selected, angles, y['future'][q], y['valid'][q])
            blocks.append((feat.reshape(-1, model.WIDTH), target.reshape(-1, 3)))
    if not blocks:
        raise ValueError('No donor action examples')
    return tuple(torch.from_numpy(np.concatenate(col)) for col in zip(*blocks, strict=True))


def fit(key, name, seed, donors, excluded, batches):
    saved = old.LOCAL
    result, refs = {}, {}
    old.LOCAL = LOCAL
    try:
        for a in ('ego', 'context'):
            torch.manual_seed(seed)
            ident = dict(name=f'{name}_{a}', registration=key, seed=seed, fit_sites=donors, excluded=excluded)
            result[a], refs[a] = old.fit(ident, batches, model.loss, model.AngleValue(a == 'context'), STEPS, 256)
    finally:
        old.LOCAL = saved
    return result, refs


@torch.inference_mode()
def propose(packet, d, nets, constant):
    cv, anchor = packet['cv'], pair.prediction(packet, 'additive')
    ends = np.r_[0, d['counts'].cumsum()]
    allowed = set(zip(ends[:-1], ends[1:], strict=True))
    x = []
    for i, (lo, hi) in enumerate(packet['bounds']):
        rows = packet['additive_rows'][i]
        if (lo, hi) not in allowed or len(set(rows)) != 2 or not np.all((lo <= rows) & (rows < hi)):
            raise ValueError('Unaligned original recipients')
        q = slice(lo, hi)
        x.append(inputs(cv[q], anchor[q], rows-lo, d, q)[0])
    t = torch.from_numpy(np.stack(x))
    pred = {a: net(t).numpy() for a, net in nets.items()}
    choices = np.full((len(ARMS), len(t), 2), -1, dtype=np.int16)
    scores = np.zeros((len(ARMS), len(t)))
    for i, (a, b) in enumerate(zip(pred['ego'], pred['context'], strict=True)):
        val = {'ego': model.choose(a), 'context': model.choose(b), 'joint': model.choose(b, joint=True)}
        row = val['context'][0][0]
        if row >= 0:
            val.update(ego_at_context=model.choose(a, same_actor=row),
                       constant_at_context=((row, constant), 0.),
                       joint_at_context=model.choose(b, joint=True, same_actor=row))
        for arm, (p, s) in val.items():
            choices[ARMS.index(arm), i] = p
            scores[ARMS.index(arm), i] = s
    return dict(choices=choices, scores=scores)


def paths(packet, selected):
    cv, original = packet['cv'], pair.prediction(packet, 'additive')
    out = original.copy()
    for rows, (i, angle) in zip(packet['additive_rows'], selected, strict=True):
        if i >= 0:
            out[rows[i]] = model.candidates(cv[rows], original[rows])[i, angle]
    return out


def calibrated_rule(packet, d, cut, choices, scores):
    on = choices[:, 0] >= 0
    out = paths(packet, choices)
    evaluated = pair.evaluate(out, packet, d, cut)
    original = pair.prediction(packet, 'additive')
    _, lower = base.parent.paired(original, out, d['future'], d['valid'])
    gain = np.asarray([d['weight'][lo:hi] @ lower[lo:hi].sum(1) for lo, hi in packet['bounds']])
    return base.calibration(scores[on], (evaluated['dual'] & (gain > 0.))[on])


def freeze(key, data, f, forecasts):
    cells = {}
    for name, c in f['cells'].items():
        receipt = LOCAL/f'{name}.json'
        excluded = sorted(c['packets'])
        donors, seed = c['fit_sites'], c['seed']
        if set(donors) != set(base.SITES)-set(excluded):
            raise ValueError('Overlapping train, calibration or held site')
        if not receipt.exists():
            batches = [donor_batch(data[s], s, seed, excluded, forecasts) for s in donors]
            mean = [((y[:, 0]-y[:, 1]).reshape(-1, 12)).mean(0).numpy() for _, y in batches]
            constant = int(np.mean(mean, axis=0).argmax())+1
            nets, refs = fit(key, name, seed, donors, excluded, batches)
            files, rules = {}, {}
            for s, r in c['packets'].items():
                with context.prior.prior.ref.prior.packet(study.LOCAL, r) as packet:
                    p = propose(packet, data[s], nets, constant)
                    d = labels(data[s]['file'])
                    rules[s] = {a: calibrated_rule(packet, d, c['cut'], p['choices'][ARMS.index(a)],
                             p['scores'][ARMS.index(a)]) for a in CAL}
                file = LOCAL/f'{name}_{s}.npz'
                files[s] = dict(file=file.name, sha256=base.parent.indices.save_npz(file, p), incumbent=r['sha256'])
            write(receipt, dict(registration=key, train_actions=[len(x) for x, _ in batches], fit_sites=donors,
                 donor_angle=constant, checkpoints=refs, packets=files, rules_from_calibration_source=rules), immutable=True)
        r = read(receipt)
        if (r['registration'] != key or any(sha(LOCAL/z['file']) != z['sha256'] for z in r['packets'].values())
                or any(sha(LOCAL/f"{v['identity']['name']}.pt") != v['sha256'] for v in r['checkpoints'].values())):
            raise ValueError('Cached choices or checkpoints changed')
        cells[name] = r
        context.write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase='freeze', cell=name, completed=len(cells), total=63))
        print('Frozen:', len(cells), '/ 63', flush=True)
    write(OUT/'freeze.json', dict(registration=key, cells=cells, freeze_before_held_score=True), immutable=True)


def evaluate(packet, d, cut, z, rules):
    chosen = dict(zip(ARMS, z['choices'], strict=True))
    for a in CAL:
        t = rules[a][0]
        chosen[f'{a}_calibrated'] = np.where((z['scores'][ARMS.index(a)] >= t)[:, None], chosen[a], -1) if t is not None else np.full_like(chosen[a], -1)
    for a in ('ego_at_context', 'constant_at_context', 'joint_at_context'):
        np.testing.assert_array_equal(chosen[a][:, 0], chosen['context'][:, 0])
    p = {a: paths(packet, choices) for a, choices in chosen.items()}
    y, valid, w = (d[k] for k in ('future', 'valid', 'weight'))
    values = {}
    for a, x in p.items():
        row = pair.evaluate(x, packet, d, cut)
        r = chosen[a][:, 0] >= 0
        values[a] = dict(dual=int(row['dual'].sum()), risk=int(row['risk'].sum()), missing=int(row['missing'].sum()),
                         rotations=int(r.sum()), rotated_dual=int(row['dual'][r].sum()), rotated_risk=int(row['risk'][r].sum()))
    at = np.isin(d['recording'], packet['records'])
    mass = w[at] @ np.where(valid[at], np.linalg.norm(packet['cv'][at]-y[at], axis=-1), 0.).sum(1)
    scores = {}
    for a, b in PAIRS:
        known, lower = base.parent.paired(p[b][at], p[a][at], y[at], valid[at])
        incident = [context.prior.prior.pair_reduction(p[a][None, lo:hi], p[b][lo:hi], y[lo:hi], valid[lo:hi])[0, 1]
                    for lo, hi in packet['bounds']]
        scores[f'{a}_minus_{b}'] = dict(known_pp=float(100.*(w[at] @ known.sum(1))/mass),
            lower_pp=float(100.*(w[at] @ lower.sum(1))/mass), relative_lower=float(np.mean(incident)))
    return dict(views=len(packet['bounds']), arms=values, contrasts=scores)


def score(key, data, original, f):
    fp = OUT/'freeze.json'
    frozen = read(fp)
    fr = sha(fp)
    if frozen['registration'] != key or not frozen['freeze_before_held_score'] or set(frozen['cells']) != set(f['cells']):
        raise ValueError('Incomplete pre-score freeze')
    rows = []
    for name, c in f['cells'].items():
        v = frozen['cells'][name]
        for s, r in c['packets'].items():
            cal, = set(c['packets'])-{s}
            if s in v['fit_sites'] or cal in v['fit_sites'] or r['sha256'] != v['packets'][s]['incumbent']:
                raise ValueError('Changed source roles')
            ident = dict(freeze=fr, site=s, calibration=cal, seed=c['seed'], packet=v['packets'][s]['sha256'])
            dest = LOCAL/f'{name}_{s}_score.json'
            if not dest.exists():
                with context.prior.prior.ref.prior.packet(study.LOCAL, r) as packet, context.prior.prior.ref.prior.packet(LOCAL, v['packets'][s]) as z:
                    x = evaluate(packet, labels(data[s]['file']), c['cut'], z, v['rules_from_calibration_source'][cal])
                write(dest, dict(identity=ident, **x), immutable=True)
            x = read(dest)
            if x['identity'] != ident:
                raise ValueError('Changed held score packet')
            rows.append(x)
        write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase='score', completed=len(rows), total=126))
    counts = {a: dict(sum((Counter(x['arms'][a]) for x in rows), Counter())) for a in SCORE_ARMS}
    views = sum(r['views'] for r in rows)
    if (len(rows), views, counts['original']['dual']) != (126, 8874, original['summary']['additive']['matched_dual']):
        raise ValueError('Original comparison or roster changed')
    groups = [[r for r in rows if r['identity']['site'] == s] for s in base.SITES]
    draws = np.random.default_rng(2917).integers(len(groups), size=(3000, len(groups)))

    def ci(name, field):
        v = np.asarray([np.mean([np.mean([r['contrasts'][name][field] for r in g if r['identity']['seed'] == seed])
                                 for seed in base.SEEDS]) for g in groups])
        return dict(point=float(v.mean()), ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(base.SITES, v.tolist(), strict=True)))

    paired = {a: {k: ci(a, k) for k in vals} for a, vals in rows[0]['contrasts'].items()}
    r = dict(result_source='fresh_run', registration=key, freeze=fr, counts=counts, paired=paired, views=views,
        recordings=493, fits=126, updates=126*STEPS, original_2pct=True, independent_confirmation=False,
        external_outcomes_read=False, deployment_changed=False)
    write(OUT/'readout.json', r, immutable=True)
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase='complete', completed=len(rows), total=126))
    print(counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    phase = parser.parse_args().phase
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, forecasts, parent, original = register()
        if phase == 'freeze':
            freeze(key, context.load_features(source), parent, forecasts)
        else:
            score(key, context.load_features(source), original, parent)
