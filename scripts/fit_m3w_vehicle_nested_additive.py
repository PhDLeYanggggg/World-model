"""Fit source-excluded copies of the original additive recipient chooser."""

from itertools import combinations
import fcntl
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_family_alignment as family

nested, ctx, action, base, np, torch = family.nested, family.ctx, family.action, family.base, family.np, family.torch
study, old, sha, read, write = action.study, nested.old, family.sha, family.read, family.write
OUT = family.OUT.parent/'opendd_vehicle_policy_anchor_v1'
LOCAL = family.LOCAL.parent/OUT.name


def register():
    parent, source, bank, upstream, fixed, teachers = family.register()
    f, r = (family.OUT/k for k in ('freeze.json', 'readout.json'))
    if read(r)['registration'] != parent or read(r)['freeze'] != sha(f) or read(r)['external_outcomes_read']:
        raise ValueError('Require completed frozen path-family evidence')
    paths = (Path(__file__), Path(study.__file__), Path(study.model.__file__),
             Path(old.__file__), Path(nested.__file__), family.prior.OUT/'teacher_freeze.json',
             f, r, OUT/'protocol.md')
    p = OUT/'additive_registration.json'
    write(p, dict(files={str(x.resolve().relative_to(base.prior.ROOT)): sha(x) for x in paths},
        seeds=base.SEEDS, steps=study.STEPS, pair_cap=study.CAP, excluded_sources=3, workers=0,
        torch_threads=4, original_2pct=True, external_outcomes_read=False), immutable=True)
    return sha(p), source, bank, upstream, fixed, teachers


def batches(data, excluded, donors, seed, teachers, cut):
    out = []
    for site in donors:
        ex = sorted([*excluded, site])
        if len(ex) != 4 or len(set(ex)) != 4:
            raise ValueError('Inner forecast must exclude its own candidate source')
        tag = nested.tag(ex, seed)
        ref = teachers['checkpoints']['forecast'][tag]
        net = base.load_net(family.prior.LOCAL/f'{tag}_forecast.pt', ref,
                            old.motion.motion.network(17), ex)
        d = data[site]
        rng = np.random.default_rng([seed, base.SITES.index(site), *(base.SITES.index(s) for s in excluded)])
        out.append(study.donor_batch(d, study.menu(d, net), cut, rng))
    return out


def fit_all(key, data, teachers):
    cache = {}
    for ex in combinations(base.SITES, 3):
        excluded = list(ex)
        donors = [s for s in base.SITES if s not in ex]
        cut = old.calibration_cut(data, donors)
        for seed in base.SEEDS:
            tag = nested.tag(ex, seed)
            name = f'{tag}_additive'
            identity = dict(registration=key, name=name, seed=seed, excluded=excluded, fit_sites=donors)
            p = LOCAL/f'{name}.json'
            if not p.exists():
                train = batches(data, excluded, donors, seed, teachers, cut)
                saved = old.LOCAL
                try:
                    old.LOCAL = LOCAL
                    torch.manual_seed(seed)
                    _, ref = old.fit(identity, train, study.model.loss, study.model.PairValue(False),
                                      study.STEPS, old.BATCH)
                finally:
                    old.LOCAL = saved
                write(p, dict(checkpoint=ref, train_examples=[len(x[0]) for x in train],
                              cut=cut.tolist()), immutable=True)
            result = read(p)
            ref = result['checkpoint']
            if (ref['identity'] != identity or ref['step'] != study.STEPS
                    or ref['sha256'] != sha(LOCAL/f'{name}.pt') or result['cut'] != cut.tolist()):
                raise ValueError('Changed nested additive checkpoint')
            cache[tag] = result
            write(LOCAL/'additive_heartbeat.json', dict(pid=os.getpid(), completed=len(cache), total=105))
            if len(cache) % 7 == 0:
                print('Additive:', len(cache), '/ 105', flush=True)
    write(OUT/'additive_freeze.json', dict(registration=key, fits=cache, updates=len(cache)*study.STEPS,
        external_outcomes_read=False), immutable=True)


if __name__ == '__main__':
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'additive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, _, _, _, teachers = register()
        fit_all(key, base.parent.load(source), teachers)
