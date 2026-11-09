"""Refit causal action costs on the expanded OpenDD menu with pair-excluded donors."""

import argparse
from collections import Counter
from itertools import combinations
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_opendd_self_menu as menu
from src.world_model import m3w_exact_slot_costs as model

base, study, np, torch, exp = menu.base, menu.study, menu.np, menu.torch, menu.exp
SITES, SEEDS, CAL = menu.SITES, menu.SEEDS, menu.calibration
OUT = menu.OUT.parent/'opendd_expanded_costs_v1'
LOCAL = menu.LOCAL.parent/OUT.name
STEPS, BATCH, CHECKPOINT = 400, 128, 50
ARMS = ('frozen_additive', 'frozen_joint', 'trained_additive', 'trained_joint')


def donor_path(pair, seed):
    return CAL.PRIVATE/f'{"+".join(pair)}_{seed}_self.pt'


def register():
    key, cohorts, _, _, slate, weights, producer = menu.register()
    fpath, rpath = (menu.OUT/f'{k}.json' for k in ('freeze', 'readout'))
    frozen, result = map(study.read, (fpath, rpath))
    calibration = CAL.PUBLIC/'registration.json'
    inner = CAL.PUBLIC/'analysis.json'
    saved = study.read(inner)
    expected = {(f'{a}+{b}', seed) for a, b in combinations(SITES, 2) for seed in SEEDS}
    if (frozen['registration'] != key or frozen['outcomes_accessed'] or result['freeze'] != study.artifact(fpath)
            or saved['registration'] != base.sha(calibration) or saved['official_confirmation']
            or {(r['held'], r['seed']) for r in saved['fresh_inner_fits']} != expected):
        raise ValueError('Require frozen expanded menu and all pair-excluded motion fits')
    refs = {f'{a}+{b}_{seed}': study.artifact(donor_path((a, b), seed))
            for a, b in combinations(SITES, 2) for seed in SEEDS}
    files = (Path(__file__), Path(model.__file__), Path(menu.__file__), Path(menu.self_model.__file__),
             Path(model.shared.__file__), Path(model.shared.costs.__file__))
    p = OUT/'registration.json'
    base.immutable(p, dict(parents=[study.artifact(q) for q in (fpath, rpath, calibration, inner)],
        implementation=[study.artifact(q) for q in files], donor_producers=refs, arms=ARMS,
        sites=SITES, seeds=SEEDS, updates=STEPS, batch=BATCH, checkpoint_every=CHECKPOINT,
        training_amplitudes=model.AMPLITUDES, donor_cut='other five sites; excludes donor and outer',
        objective='Huber beta .05 on two signed lower gains and three signed risk terms; equal weights',
        decoder='min two gains minus positive excesses; additive per action versus joint after aggregation',
        original_2pct=True, thresholds_tuned=False, cpu_threads=4, workers=0,
        development_only=True, independent_confirmation=False))
    return base.sha(p), cohorts, slate, frozen, result, weights, producer, refs, base.sha(calibration)


@torch.inference_mode()
def pair_forecast(h, pair, seed, refs, calibration):
    label = '+'.join(pair)
    state = torch.load(study.checked(refs[f'{label}_{seed}']), weights_only=True, map_location='cpu')
    identity = dict(registration=calibration, train=[s for s in SITES if s not in pair],
                    held=label, seed=seed, arm='self')
    if state['identity'] != identity or state['step'] != CAL.velocity.STEPS:
        raise ValueError('Donor producer may have seen the donor or outer site')
    x, rot, scale = menu.self_model.design(h, np.full((len(h), 2, 2, 8, 2), np.nan))
    net = menu.self_model.network(x.shape[1]).eval()
    net.load_state_dict(state['model'])
    decoded = menu.self_model.decode(net(torch.from_numpy(x))).numpy().astype(float)
    return (decoded*scale[:, None, None]) @ rot.transpose(0, 2, 1)


def donor_examples(data, held, seed, refs, calibration):
    out = {}
    for donor in SITES:
        if donor == held:
            continue
        pair = tuple(sorted((held, donor)))
        src = data[donor]
        forecast = pair_forecast(src['h'], pair, seed, refs, calibration)
        bank = np.concatenate((src['bank'], forecast[:, None]), axis=1)
        easy = float(np.quantile(np.concatenate([data[s]['e'] for s in SITES if s not in pair]), .25))
        out[donor] = model.training_examples(src['h'], src['cv'], src['ols'], bank, src['y'], src['v'], easy)
        if not len(out[donor][0]):
            raise ValueError('Donor lacks legal action training examples')
    return out


def beat(**row):
    path = LOCAL/'heartbeat.pending'
    path.write_text(study.json.dumps(dict(pid=os.getpid(), **row))+'\n')
    path.replace(LOCAL/'heartbeat.json')


def fit(key, site, seed, train):
    name = f'{site}_{seed}'
    path = LOCAL/f'{name}.pt'
    ident = dict(registration=key, held=site, seed=seed, donors={s: len(v[0]) for s, v in train.items()})
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+7091)
    net = model.CostHead()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location='cpu')
        if ck['identity'] != ident or ck['step'] not in range(CHECKPOINT, STEPS+1, CHECKPOINT):
            raise ValueError('Checkpoint identity or progress changed')
        net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['optimizer'])
        step, trace, rng.bit_generator.state = ck['step'], ck['trace'], ck['rng']
    donors = tuple(train.values())
    while step < STEPS:
        counts = rng.multinomial(BATCH, np.full(len(donors), 1./len(donors)))
        indices = [rng.integers(len(t[0]), size=n) for n, t in zip(counts, donors, strict=True)]
        x, y = [torch.cat([a[k][i] for a, i in zip(donors, indices, strict=True)]) for k in range(2)]
        loss = torch.nn.functional.smooth_l1_loss(net(x), y, beta=.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        step += 1
        if step % CHECKPOINT == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident, net=net.state_dict(), optimizer=opt.state_dict(), step=step,
                            trace=trace, rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat(phase='training', fold=name, step=step, total=STEPS)
    return net.eval(), dict(identity=ident, checkpoint=study.artifact(path), loss=trace)


def predict(net, h, q, cv, ols, bank, origin):
    peak = np.linalg.norm(origin-cv, axis=-1).max(-1)
    output = {k: cv.copy() for k in ARMS[2:]}
    counts = Counter()
    for j in np.unique(q[peak > 0.]):
        ix = np.flatnonzero(q == j)
        radii = peak[ix][peak[ix] > 0.]
        a, b = model.choose(net, h[ix], cv[ix], ols[ix], bank[ix], radii)
        output[ARMS[2]][ix], output[ARMS[3]][ix] = a, b
        counts[f'queries_{len(radii)}_slots'] += 1
        counts['two_arm_disagreements'] += int(not np.allclose(a, b, atol=1e-9, rtol=1e-9))
    return output, dict(counts)


def freeze(key, cohorts, slate, parent, weights, producer, refs, calibration):
    data = exp.loader(cohorts)
    fits, banks, decisions = {}, {}, {}
    for site in SITES:
        h, q, cv, ols, bank = menu.parent.source_inputs(cohorts[site])
        for seed in SEEDS:
            name = f'{site}_{seed}'
            train = donor_examples(data, site, seed, refs, calibration)
            net, fits[name] = fit(key, site, seed, train)
            self_path = menu.self_forecast(h, site, seed, weights[name], producer)
            with (np.load(study.checked(slate['forecasts'][name]), allow_pickle=False) as original,
                  np.load(study.checked(parent['banks'][name]), allow_pickle=False) as previous):
                for k, x in zip(('q', 'cv', 'ols4'), (q, cv, ols), strict=True):
                    np.testing.assert_array_equal(previous[k], x)
                np.testing.assert_array_equal(original['q'], q)
                np.testing.assert_array_equal(original['cv'], cv)
                pred, decisions[name] = predict(net, h, q, cv, ols, np.concatenate((bank, self_path[:, None]), axis=1),
                                                original['joint'])
                values = dict(q=q, cv=cv, ols4=ols, frozen_additive=previous['expanded_additive'],
                              frozen_joint=previous['expanded_joint'], **pred)
            banks[name] = exp.previous.save_bank(LOCAL/f'{name}.npz', values)
            beat(phase='inference', last=name, completed=len(banks), total=21)
            print(name, decisions[name], flush=True)
    base.immutable(OUT/'freeze.json', dict(registration=key, fits=fits, banks=banks, decisions=decisions,
        loss_uses_donor_only_for_each_fold=True, inference_future_inputs=False, held_scoring_not_run=True))


def score(key, cohorts, parent):
    p = OUT/'freeze.json'
    frozen = study.read(p)
    if frozen['registration'] != key or not frozen['held_scoring_not_run'] or len(frozen['fits']) != 21:
        raise ValueError('Require all matched forecasts frozen before held scoring')
    data, by_site = exp.loader(cohorts), {}
    names = dict(zip(menu.ARMS, ARMS, strict=True))
    for site in SITES:
        d = data[site]
        cut = np.quantile(np.concatenate([v['e'] for s, v in data.items() if s != site]), [.25, .75])
        rows, effects = [], []
        for seed in SEEDS:
            name = f'{site}_{seed}'
            ident = frozen['fits'][name]['identity']
            if ident['held'] != site or set(ident['donors']) != set(SITES)-{site}:
                raise ValueError('Held location appeared in cost fit')
            with np.load(study.checked(frozen['banks'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], d['q'])
                paths = {k: z[v] for k, v in names.items()}
                paths.update(cv=z['cv'], ols4=z['ols4'])
            for record in np.unique(d['rec']):
                ix = d['rec'] == record
                a, b = menu.recording({k: v[ix] for k, v in paths.items()}, d['y'][ix], d['v'][ix],
                                      np.unique(d['q'][ix], return_inverse=True)[1], cut)
                rows.append({names[k]: v for k, v in a.items()})
                effects.append({f'{names[a]}_vs_{names[b]}': v for (a, b), v in zip(menu.CONTRASTS, b.values(), strict=True)})
        aggregate = {k: {v: sum(r[k][v] for r in rows) for v in ('edits', 'risk', 'dual')} for k in ARMS}
        for a, b in zip(ARMS[:2], menu.ARMS[2:], strict=True):
            if aggregate[a] != parent['by_site'][site]['counts'][b]:
                raise ValueError('Frozen expanded-menu control did not replay')
        means = {k: [p[k] for p in effects if p[k] is not None] for k in effects[0]}
        by_site[site] = dict(views=len(rows), counts=aggregate,
            contrasts_pp={k: float(np.mean(v)) if v else None for k, v in means.items()},
            estimable={k: len(v) for k, v in means.items()},
            outcomes={k: {m: float(np.mean(v)) if (v := [r[k][m] for r in rows if r[k][m] is not None]) else None
                          for m in ('ade', 'fde', 'lower_vs_cv_pp', 'lower_vs_ols4_pp', 'easy_degradation')}
                      for k in ARMS})
    def interval(k):
        values = np.array([s['contrasts_pp'][k] for s in by_site.values() if s['contrasts_pp'][k] is not None])
        draws = np.random.default_rng(3183).choice(values, size=(3000, len(values))).mean(1)
        return dict(mean_pp=float(values.mean()), ci95_pp=np.quantile(draws, [.025, .975]).tolist(), sites=len(values))
    result = dict(result_source='fresh_run', new_fits=21, optimizer_updates=21*STEPS, seeds=SEEDS,
        counts={k: dict(sum((Counter(s['counts'][k]) for s in by_site.values()), Counter())) for k in ARMS},
        contrasts={k: interval(k) for k in next(iter(by_site.values()))['contrasts_pp']},
        assignments=dict(sum(map(Counter, frozen['decisions'].values()), Counter())), by_site=by_site,
        original_2pct=True, training_and_calibration_exclude_held=True, independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False, freeze=study.artifact(p))
    base.immutable(OUT/'readout.json', result)
    print(study.json.dumps({k: result[k] for k in ('counts', 'contrasts', 'assignments')}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, slate, frozen, result, weights, producer, refs, cal = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, slate, frozen, weights, producer, refs, cal)
        else:
            score(key, cohorts, result)


if __name__ == '__main__':
    main()
