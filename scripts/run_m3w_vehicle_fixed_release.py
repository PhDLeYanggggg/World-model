"""Refit release scores without changing any frozen OpenDD candidate paths."""

import argparse
from collections import Counter, defaultdict
import fcntl
import json
from math import fsum
import os
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import package_m3w_vehicle_recording_replay as replay
from scripts import run_m3w_vehicle_anchor_rank as rank
from src.world_model import m3w_vehicle_release_score as model

base, old, np, torch, study = rank.base, rank.old, rank.np, rank.torch, rank.study
verify = replay.verify
OUT = rank.OUT.parent/'opendd_vehicle_fixed_release_v1'
LOCAL = rank.LOCAL.parent/OUT.name
FIT = ('direct', 'nested')
ARMS = ('archived', *FIT)


def register():
    k, source, parent, cost, forecasts = rank.register()
    prior = rank.OUT/'freeze.json'
    r = old.read(prior)
    scalar = old.read(replay.OUT/'receipt.json')
    if (r['registration'] != k or set(r['cells']) != set(parent['cells']) or not r['held_choices_frozen']
            or scalar['freeze_sha256'] != rank.audit.sha(prior) or scalar['external_outcomes_read']):
        raise ValueError('Require the unchanged frozen relative proposal and replay')
    files = (Path(__file__), Path(model.__file__), prior, OUT/'protocol.md', replay.OUT/'receipt.json',
             Path(verify.__file__))
    p = OUT/'registration.json'
    old.write(p, dict(files={str(f.relative_to(base.prior.ROOT)): rank.audit.sha(f) for f in files},
                      arms=ARMS, seeds=base.SEEDS, steps=study.STEPS, workers=0, threads=4,
                      fixed_proposals=True, original_2pct=True, external_outcomes_read=False))
    return rank.audit.sha(p), source, parent, r, cost, forecasts


def beat(**fields):
    rank.audit.write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), **fields))
    print(fields, flush=True)


def fit(key, name, seed, sites, donors, batches):
    result, weights = {}, {}
    saved = old.LOCAL
    try:
        old.LOCAL = LOCAL
        for a in FIT:
            torch.manual_seed(seed)
            identity = dict(name=f'{name}_{a}', registration=key, seed=seed, excluded=sites, fit_sites=donors)
            loss = lambda net, batch: model.loss(net, batch, a == 'nested')
            result[a], weights[a] = old.fit(identity, batches, loss, rank.model.AnchorValue(),
                                            study.STEPS, old.BATCH)
    finally:
        old.LOCAL = saved
    return result, weights


@torch.inference_mode()
def fixed_scores(m, z, models):
    for k in ('cv', 'ols4'):
        np.testing.assert_array_equal(m[k], z[k])
    np.testing.assert_array_equal(list(m['slots']), z['records'])
    np.testing.assert_array_equal(list(m['slots'].values()), z['bounds'])
    chosen, anchor = ({f'additive_{k}': z[f'{arm}_{k}'] for k in ('rows', 'paths')}
                      for arm in ('relative', 'additive'))
    rows = []
    for n, (lo, hi) in enumerate(z['bounds']):
        r, f = np.nonzero(m['peak'][lo:hi] > 0.)
        r += lo
        p = m['paths'][r, f]
        i, j = rank.anchor_actions(chosen, n, r, p)
        a = rank.anchor_actions(anchor, n, r, p)
        rows.append(rank.features(m, r, f, np.array([i]), np.array([j]), a))
    features = [torch.from_numpy(np.concatenate(x)) for x in zip(*rows, strict=True)]
    scores = {a: model.log_score(net(*features), nested=(a == 'nested')).numpy().tolist()
              for a, net in models.items()}
    if not all(np.isfinite(s).all() for s in scores.values()):
        raise ValueError('Nonfinite release score')
    return dict(recordings=z['records'].tolist(), scores=scores)


def freeze(key, source, parent, fixed, cost, forecasts):
    data, cells = base.parent.load(source), {}
    for name, c in fixed['cells'].items():
        sites, cut, seed = list(c['packets']), c['cut'], c['seed']
        donors = parent['cells'][name]['fit_sites']
        if set(donors) != set(base.SITES)-set(sites):
            raise ValueError('Donor site overlaps calibration or held site')
        receipt = LOCAL/f'{name}.json'
        if receipt.exists():
            r = old.read(receipt)
            if (r['registration'] != key
                    or any(rank.audit.sha(LOCAL/f"{w['identity']['name']}.pt") != w['sha256']
                           for w in r['weights'].values())):
                raise ValueError('Frozen checkpoint changed')
        else:
            batches = []
            for donor in donors:
                excluded = (*sites, donor)
                tag = base.triple_id(excluded, seed)
                n = base.load_net(base.LOCAL/f'{tag}.pt', cost['triple_checkpoints'][tag],
                                  old.motion.motion.network(17), excluded)
                d = data[donor]
                rng = np.random.default_rng([2917, seed, base.SITES.index(donor), *map(base.SITES.index, sites)])
                batches.append(rank.donor_batch(d, study.menu(d, n), cut, rng))
            if [len(b[0]) for b in batches] != c['donor_pairs']:
                raise ValueError('Refit changed the donor sample')
            nets, weights = fit(key, name, seed, sites, donors, batches)
            tag = old.double_name(sites, seed)
            n = base.load_net(old.LOCAL/f'{tag}.pt', forecasts['double_excluded'][tag],
                              old.motion.motion.network(17), sites)
            packets = {}
            for s, p in c['packets'].items():
                with rank.audit.prior.packet(rank.LOCAL, p) as z:
                    packets[s] = fixed_scores(study.menu(data[s], n), z, nets)
            r = dict(registration=key, seed=seed, weights=weights, donor_pairs=c['donor_pairs'],
                     packets=packets)
            old.write(receipt, r)
        cells[name] = rank.audit.sha(receipt)
        beat(phase='freeze', cell=name, complete=len(cells))
    old.write(OUT/'freeze.json', dict(registration=key, cells=cells, scores_frozen=True,
                                     thresholds_selected=False, fixed_proposals=True))


def offline():
    scalar = old.read(replay.OUT/'receipt.json')
    if rank.audit.sha(replay.DEFAULT) != scalar['archive_sha256']:
        raise ValueError('Frozen scalar archive has changed')
    with ZipFile(replay.DEFAULT) as z:
        data = json.loads(z.read('evidence.json'))
    return data, scalar


def selected(score, labels, threshold):
    t, use = labels, [i for i, x in enumerate(score) if threshold is not None and x is not None and x >= threshold]
    return dict(directions=int(bool(use)), admitted=len(use), risk_pass=sum(t[i][0] for i in use),
                dual=sum(t[i][1] for i in use), full_win=sum(t[i][2] for i in use),
                missing=sum(t[i][3] for i in use), all_full_win=int(bool(use) and all(t[i][2] for i in use)))


def score(key, fixed):
    path = OUT/'freeze.json'
    f = old.read(path)
    if f['registration'] != key or not f['scores_frozen'] or set(f['cells']) != set(fixed['cells']):
        raise ValueError('Scores must freeze before calibration or held scoring')
    data, prior = offline()
    original = {(c['site'], c['calibration'], c['seed']): c for c in data['directions']}
    scores, ids = {}, {}
    for name, c in fixed['cells'].items():
        receipt = LOCAL/f'{name}.json'
        if rank.audit.sha(receipt) != f['cells'][name]:
            raise ValueError('Refit score receipt changed')
        x = old.read(receipt)
        for site, p in x['packets'].items():
            direction = (base.SITES.index(site), base.SITES.index(c['rules'][site]['calibration_site']), c['seed'])
            ref = original[direction]['recordings']
            if [ids.setdefault(n, len(ids)) for n in p['recordings']] != [r['id'] for r in ref]:
                raise ValueError('Frozen scores do not align to the original recording aliases')
            scores[direction] = dict(archived=[r['arms']['relative']['score'] for r in ref], **p['scores'])
    if set(scores) != set(original) or len(ids) != 493:
        raise ValueError('Incomplete recording or direction roster')
    labels = {k: [verify.qualification(r['arms']['relative']) for r in c['recordings']] for k, c in original.items()}
    result = {a: Counter() for a in ARMS}
    metrics = {a: defaultdict(lambda: defaultdict(list)) for a in ARMS}
    by_seed = {a: {seed: Counter() for seed in base.SEEDS} for a in ARMS}
    rules = []
    for k, cell in original.items():
        s, cal, seed = k
        y = labels[k]
        other = (cal, s, seed)
        val = [v[2] for v in labels[other]]
        rules.append(dict(site=s, calibration=cal, seed=seed, arms={}))
        for a in ARMS:
            threshold, n = verify.prefix(zip(scores[other][a], val, strict=True))
            if a == 'archived' and [threshold, n] != cell['thresholds']['relative']:
                raise ValueError('Archived calibration result did not replay')
            x = scores[k][a]
            c = selected(x, y, threshold)
            c['calibration_prefix_directions'] = int(threshold is not None)
            result[a].update(c)
            by_seed[a][seed].update(c)
            rules[-1]['arms'][a] = dict(threshold=threshold, calibration_count=n, held=c)
            for tag, j in (('risk', 0), ('dual', 1), ('full', 2)):
                eligible = [(score, label[j]) for score, label in zip(x, y, strict=True) if score is not None]
                hits, k5 = verify.top_mass(eligible, 5)
                lift = 100.*(hits/k5-fsum(q for _, q in eligible)/len(eligible))
                metrics[a][f'top5_{tag}_lift_pp'][s].append(lift)
    prior_fields = set(prior['active']['relative'])
    if {k: result['archived'][k] for k in prior_fields} != prior['active']['relative']:
        raise ValueError('Archived release counts changed')
    intervals = {a: {k: verify.interval(group, data['source_resamples']) for k, group in v.items()}
                 for a, v in metrics.items()}
    old.write(OUT/'readout.json', dict(registration=key, freeze=rank.audit.sha(path), summary=result, by_seed=by_seed,
        intervals=intervals, directions=len(scores), recordings=len(ids), views=sum(map(len, labels.values())),
        same_proposals=True, original_2pct_unchanged=True, external_outcomes_read=False, held_tuning=False,
        deployment_changed=False, independent_confirmation=False, rules=rules))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, parent, fixed, cost, forecasts = register()
        if args.phase == 'freeze':
            freeze(key, source, parent, fixed, cost, forecasts)
        else:
            score(key, fixed)
