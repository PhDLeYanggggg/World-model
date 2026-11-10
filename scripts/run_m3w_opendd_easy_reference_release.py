"""Learn selected-easy estimability without changing frozen swap nominations."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_nested_swap_release as prior
from sklearn.metrics import roc_auc_score

np, torch, base = prior.np, prior.torch, prior.base
read, write, sha = prior.read, prior.write, prior.sha
OUT = prior.OUT.parent/'opendd_easy_reference_release_v1'
LOCAL = prior.LOCAL.parent/OUT.name
ARMS = ('sign', 'sign_times_reference', 'reference_only')
STEPS, BATCH = 200, 160


def register():
    key, source, cells, cost = prior.register()
    archived = prior.phase_input(key)
    results = read(prior.OUT/'readout.json')
    if results['deployment_changed'] or results['independent_confirmation']:
        raise ValueError('Require the unchanged OpenDD development comparison')
    p = OUT/'registration.json'
    files = (Path(__file__), OUT/'protocol.md', prior.OUT/'freeze.json',
             prior.OUT/'calibration_freeze.json', prior.OUT/'readout.json')
    write(p, dict(files={str(v.relative_to(base.prior.ROOT)): sha(v) for v in files},
                  old_key=key, arms=ARMS, seeds=base.SEEDS, steps=STEPS, batch=BATCH,
                  threads=4, workers=0, risk_limit=.02, new_external_outcomes=False), immutable=True)
    return sha(p), source, cells, cost, archived


def easy_events(cv, y, valid, cut):
    """Same positive selected-easy denominator as the original risk check."""
    count = valid.sum(axis=1)
    err = np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(axis=1)/count.clip(1)
    return (count > 0) & (err > 0.) & (err <= cut)


def actor_features(history, offsets, cv, ols4):
    delta = np.diff(history, axis=1)
    speed = np.linalg.norm(delta, axis=-1)
    accel = np.diff(delta, axis=1)
    past = history+offsets[:, None]
    lo, hi = past.min((0, 1)), past.max((0, 1))
    projected = (cv+offsets[:, None]-lo)/(hi-lo).clip(.1)
    margin = np.minimum(projected, 1.-projected).min(-1)
    disagreement = np.linalg.norm(cv-ols4, axis=-1)
    # Rotation-invariant motion, plus the shared observed footprint.
    return np.column_stack((np.log1p(speed), np.log1p(np.linalg.norm(accel, axis=-1)),
                            np.log1p(disagreement[:, [0, -1]]), (margin >= 0.).mean(axis=1),
                            margin[:, -1].clip(-8., 1.), margin.min(axis=1).clip(-8., 1.)))


def pair_features(actor, kept, added):
    a, b = actor[kept], actor[added]
    return np.column_stack((np.minimum(a, b), np.maximum(a, b),
                            np.full(len(a), np.log1p(len(actor))))).astype(np.float32)


def proposals(packet, feature):
    cv, anchor, allowed = prior.run.old.geometry(packet, feature)
    for i, (q, drop, add, _, _) in enumerate(prior.run.menus(packet, feature, cv, anchor, allowed)):
        if not len(drop):
            yield None
            continue
        left, right = packet['additive_rows'][i]-q.start
        if not np.isin(drop, [left, right]).all():
            raise ValueError('Swap removed an actor outside the anchor')
        kept = np.where(drop == left, right, left)
        x = actor_features(feature['history'][q], feature['offsets'][q], cv[q], packet['ols4'][q])
        yield q, kept, add, pair_features(x, kept, add)


def donor(key, c, site, data, feature, triples):
    excluded = sorted((*c['packets'], site))
    tag = base.triple_id(excluded, c['seed'])
    ck = triples[tag]
    ident = dict(key=key, excluded=excluded, cohort=sha(feature['file']), forecaster=ck['sha256'], cut=c['cut'][0])
    path = LOCAL/f'{tag}_{site}_examples.npz'
    receipt = path.with_suffix('.json')
    if not receipt.exists():
        net = base.load_net(base.LOCAL/f'{tag}.pt', ck, prior.old.motion.motion.network(17), excluded)
        packet = prior.donor_packet(data, net)
        easy = easy_events(packet['cv'], data['future'], data['valid'], c['cut'][0])
        chunks = [(x, easy[q][a] | easy[q][b]) for item in proposals(packet, feature)
                  if item is not None for q, a, b, x in [item]]
        x, y = (np.concatenate(v) for v in zip(*chunks, strict=True))
        write(receipt, dict(identity=ident, sha256=base.parent.indices.save_npz(path, dict(x=x, y=y)),
                            rows=len(y), positive=int(y.sum())), immutable=True)
    r = read(receipt)
    if r['identity'] != ident or sha(path) != r['sha256']:
        raise ValueError('Changed support training examples')
    with np.load(path, allow_pickle=False) as z:
        return (torch.from_numpy(z['x']), torch.from_numpy(z['y'].astype(np.float32))), r


def fit(identity, samples):
    torch.manual_seed(identity['seed'])
    nn = torch.nn
    net = nn.Sequential(nn.Linear(samples[0][0].shape[1], 32), nn.SiLU(), nn.Linear(32, 1), nn.Flatten(0))
    home = prior.old.LOCAL
    try:
        prior.old.LOCAL = LOCAL
        return prior.old.fit(identity, samples,
                             lambda m, z: nn.functional.binary_cross_entropy_with_logits(m(z[0]), z[1]),
                             net, STEPS, BATCH)
    finally:
        prior.old.LOCAL = home


@torch.inference_mode()
def forecast(packet, f, actions, net):
    indices = actions['sign'][:, 0]
    probs = np.zeros(len(indices))
    for i, (j, item) in enumerate(zip(indices, proposals(packet, f), strict=True)):
        if (j < 0.) != (item is None):
            raise ValueError('Frozen sign nomination differs from reconstructed menu')
        if item is not None:
            x = item[-1][[int(j)]]
            probs[i] = net(torch.from_numpy(x)).sigmoid().item()
    return probs


def freeze(key, source, cells, costs, archived):
    data, features = base.parent.load(source), prior.run.old.prior.load_features(source)
    output = {}
    for name, c in cells.items():
        receipt = LOCAL/f'{name}_forecast.json'
        if not receipt.exists():
            donors, proof = [], {}
            for site in c['fit_sites']:
                batch, proof[site] = donor(key, c, site, data[site], features[site], costs['triple_checkpoints'])
                donors.append(batch)
            net, ck = fit(dict(key=key, name=name, seed=c['seed'], excluded=sorted(c['packets']),
                              donors=proof), donors)
            forecasts = {}
            for site in c['packets']:
                packet, actions = prior.opened(site, name, c, archived)
                with packet as p, actions as a:
                    values = forecast(p, features[site], a, net)
                dest = LOCAL/f'{name}_{site}.npz'
                forecasts[site] = dict(file=dest.name, sha256=base.parent.indices.save_npz(dest, dict(p=values)),
                                       original=archived['cells'][name]['packets'][site]['sha256'])
            write(receipt, dict(key=key, donors=proof, checkpoint=ck, forecasts=forecasts), immutable=True)
        r = read(receipt)
        if (r['key'] != key or set(r['donors']) != set(c['fit_sites'])
                or sha(LOCAL/f'{name}.pt') != r['checkpoint']['sha256']):
            raise ValueError('Changed selected-easy fit')
        output[name] = r
        beat('freeze', len(output), 63)
    write(OUT/'freeze.json', dict(key=key, cells=output), immutable=True)


def beat(phase, completed, total):
    info = dict(pid=os.getpid(), phase=phase, completed=completed, total=total)
    write(LOCAL/'heartbeat.json', info)
    print(info, flush=True)


def policies(sign, reference):
    if sign.shape != reference.shape or not np.all((reference >= 0.) & (reference <= 1.)):
        raise ValueError('Misaligned or invalid reference probabilities')
    return dict(sign=sign, sign_times_reference=sign*reference, reference_only=reference)


def inspect(site, name, c, archived, frozen, data, features):
    if site in c['fit_sites'] or site in frozen['cells'][name]['donors']:
        raise ValueError('Source roles overlap')
    pck, act = prior.opened(site, name, c, archived)
    ref = frozen['cells'][name]['forecasts'][site]
    fp = prior.support.checked_local(ref, LOCAL, archived['cells'][name]['packets'][site]['sha256'])
    with pck as p, act as actions, np.load(fp, allow_pickle=False) as z:
        cv, pred, scores = prior.inspect(p, features[site], data[site], c['cut'], actions)
        sign = scores['sign']
        scores = {arm: dict(sign, score=s) for arm, s in policies(sign['score'], z['p']).items()}
        pos = sign['enabled']
        event = ~sign['missing'][pos]
        prob = z['p'][pos]
        quality = dict(rows=len(event), positives=int(event.sum()), brier=float(np.mean((event-prob)**2)),
                       auc=float(roc_auc_score(event, prob)) if 0 < event.sum() < len(event) else None)
        packet = {k: p[k] for k in ('records', 'bounds', 'ols4')}
    return packet, cv, dict.fromkeys(ARMS, pred['sign']), scores, quality


def phases(key, source, cells, archived, score):
    f, t = OUT/'freeze.json', OUT/'calibration_freeze.json'
    prior.run.old.require_published(f)
    frozen = read(f)
    if frozen['key'] != key:
        raise ValueError('Changed support score registration')
    previous = read(prior.OUT/'calibration_freeze.json')['rules']
    if score:
        prior.run.old.require_published(t)
        rule = read(t)
        if rule['key'] != key or rule['freeze'] != sha(f):
            raise ValueError('Changed support calibration freeze')
        rules = rule['rules']
    data, features = base.parent.load(source), prior.run.old.prior.load_features(source)
    out = {}
    for name, c in cells.items():
        for held, old in c['rules'].items():
            cal = old['calibration_site']
            if cal == held or set((cal, held)) != set(c['packets']):
                raise ValueError('Calibration must exclude the held location')
            tag = f'{name}/{held}'
            site = held if score else cal
            p, cv, pred, xs, q = inspect(site, name, c, archived, frozen, data, features)
            if score:
                arms = prior.held_metrics(p, data[site], cv, pred, xs, rules[tag]['arms'])
            else:
                arms = {}
                for a, x in xs.items():
                    use = x['enabled']
                    threshold, count = base.calibration(x['score'][use], x['strict'][use])
                    arms[a] = dict(threshold=threshold, calibration_recordings=count)
                if any(previous[tag]['arms']['sign'][k] != v for k, v in arms['sign'].items()):
                    raise ValueError('Frozen sign release comparator must reproduce')
            out[tag] = dict(held=held, calibration=cal, seed=c['seed'], arms=arms, quality=q)
        beat('score' if score else 'calibrate', len(out), 126)
    if not score:
        write(t, dict(key=key, freeze=sha(f), rules=out), immutable=True)
        return
    rows = list(out.values())
    summary = summarize(rows, rules)
    old = read(prior.OUT/'readout.json')['summary']['sign']
    if any(summary['sign'][k] != old[k] for k in ('released', 'risk', 'dual', 'strict', 'missing', 'effects')):
        raise ValueError('Held sign comparator must reproduce')
    write(OUT/'readout.json', dict(key=key, calibration=sha(t), summary=summary, rows=out,
          paired=paired(summary), development_only=True, original_2pct=True, deployment_changed=False,
          external_outcomes_read=False, submission_ready=False), immutable=True)


def summarize(rows, thresholds):
    count_keys = ('released', 'risk', 'dual', 'strict', 'missing', 'eligible')
    metrics = ('observed_cv_pp', 'lower_cv_pp', 'lower_ols_pp', 'relative_observed', 'relative_lower')
    draw = np.random.default_rng(211).integers(7, size=(3000, 7))
    groups = {s: [r for r in rows if r['held'] == s] for s in base.SITES}
    output = {}
    for arm in ARMS:
        x = [r['arms'][arm] for r in rows]
        c = {k: sum(v[k] for v in x) for k in count_keys}
        c['calibration_active_directions'] = sum(v['arms'][arm]['threshold'] is not None for v in thresholds.values())
        c['held_active_directions'] = sum(v['released'] > 0 for v in x)
        c['active_directions_all_strict'] = sum(0 < v['released'] == v['strict'] for v in x)
        c['effects'] = {}
        for k in metrics:
            v = np.array([np.mean([np.mean([r['arms'][arm][k] for r in g if r['seed'] == seed])
                                   for seed in base.SEEDS]) for g in groups.values()])
            c['effects'][k] = dict(point=float(v.mean()), ci95=np.quantile(v[draw].mean(1), [.025, .975]).tolist(),
                                   by_site=dict(zip(groups, v.tolist(), strict=True)))
        output[arm] = c
    return output


def paired(summary):
    draw = np.random.default_rng(211).integers(7, size=(3000, 7))
    sign = summary['sign']['effects']['lower_cv_pp']['by_site']
    result = {}
    for a in ARMS[1:]:
        other = summary[a]['effects']['lower_cv_pp']['by_site']
        d = np.array([other[s]-sign[s] for s in sign])
        result[a] = dict(point_pp=float(d.mean()), ci95_pp=np.quantile(d[draw].mean(1), [.025, .975]).tolist())
    return result


if __name__ == '__main__':
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('phase', choices=('freeze', 'calibrate', 'score'))
    args = arg.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        k, s, c, b, old = register()
        if args.phase == 'freeze':
            freeze(k, s, c, b, old)
        else:
            phases(k, s, c, old, args.phase == 'score')
