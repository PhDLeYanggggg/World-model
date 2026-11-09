"""Add a source-held self-history forecast to both frozen exact-budget selectors."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_slate_shared_menu as parent
from scripts import run_m3w_opendd_velocity_calibration as calibration
from scripts.run_m3w_pair_effect import slate_for
from src.evaluation.m3w_exact_slot_headroom import lower
from src.world_model import m3w_opendd_velocity_candidate as self_model

base, study, exp, np, torch = parent.base, parent.study, parent.exp, parent.np, parent.torch
SITES, SEEDS = parent.SITES, parent.SEEDS
OUT = parent.OUT.parent/'opendd_self_menu_v1'
LOCAL = parent.LOCAL.parent/OUT.name
ARMS = ('base_additive', 'base_joint', 'expanded_additive', 'expanded_joint')
CONTRASTS = (('base_joint', 'base_additive'), ('expanded_joint', 'expanded_additive'),
             ('expanded_additive', 'base_additive'), ('expanded_joint', 'base_joint'))


def register():
    old = [parent.OUT/name for name in ('registration.json', 'freeze.json', 'readout.json')]
    frozen = [calibration.PUBLIC/'registration.json', calibration.velocity.PUBLIC/'registration.json']
    reg, prior, readout, calibrated, producer = map(study.read, old+frozen)
    if (prior['registration'] != base.sha(old[0]) or prior['outcomes_accessed']
            or readout['freeze'] != study.artifact(old[1]) or not reg['development_only'] or not reg['original_2pct']
            or calibrated['files'][str(frozen[1].relative_to(ROOT))] != base.sha(frozen[1])
            or calibrated['dronecrowd_confirmation_access'] or producer['dronecrowd_confirmation_access']
            or calibrated['official_risk_budget'] != .02 or tuple(calibrated['seeds']) != SEEDS):
        raise ValueError('Require the frozen shared budget and source-held motion checkpoints')
    slate_registration = study.read(study.checked(reg['parent']))
    slate_freeze = study.read(study.checked(reg['freeze']))
    cohorts = slate_registration['cohorts']
    if set(cohorts) != set(SITES):
        raise ValueError('Unexpected development source roster')
    files = [Path(__file__), Path(parent.__file__), Path(self_model.__file__), Path(parent.shared.__file__),
             ROOT/'src/evaluation/m3w_opendd_location.py', ROOT/'src/evaluation/m3w_selection_forecast_readout.py']
    path = OUT/'registration.json'
    base.immutable(path, dict(parent=[study.artifact(p) for p in old+frozen], code=[study.artifact(p) for p in files],
        arms=ARMS, sites=SITES, seeds=SEEDS, original_2pct=True, development_only=True,
        shared='same frozen selector weights, eligible agents, exact per-query peak slots and inference features',
        menu='add the source-held self-history path, shortened only to the exact slot as for the analytic paths',
        tuning=False, new_training=False, bootstrap_draws=3000, bootstrap_seed=3183))
    return base.sha(path), cohorts, prior, readout, slate_freeze, calibrated['checkpoint_sha256'], base.sha(frozen[1])


@torch.inference_mode()
def self_forecast(history, site, seed, checkpoint, producer):
    path = calibration.FROZEN/f'{site}_{seed}_self.pt'
    if base.sha(path) != checkpoint:
        raise ValueError('Self-history checkpoint changed')
    state = torch.load(path, weights_only=True, map_location='cpu')
    identity = dict(registration=producer, held=site, train=[s for s in SITES if s != site], seed=seed, arm='self')
    if state['identity'] != identity or state['step'] != calibration.velocity.STEPS:
        raise ValueError('Self-history checkpoint did not exclude the scored location')
    x, rot, scale = self_model.design(history, np.full((len(history), 2, 2, 8, 2), np.nan))
    net = self_model.network(x.shape[1]).eval()
    net.load_state_dict(state['model'])
    disp = self_model.decode(net(torch.from_numpy(x))).numpy().astype(float)
    return (disp*scale[:, None, None]) @ rot.transpose(0, 2, 1)


def scaled_to_path(cv, source, path):
    radius = np.linalg.norm(source-cv, axis=-1).max(-1)
    peak = np.linalg.norm(path-cv, axis=-1).max(-1)
    ratio = np.divide(peak, radius, out=np.zeros_like(peak), where=radius > 0.)
    return (peak > 0.) & (radius >= peak-1e-10) & np.isclose(
        cv + np.minimum(ratio, 1.)[:, None, None]*(source-cv), path, atol=1e-9, rtol=1e-9).all((1, 2))


def freeze(key, cohorts, previous, slate, hashes, producer):
    banks, stats = {}, {}
    for site in SITES:
        h, q, cv, ols, bank = parent.source_inputs(cohorts[site])
        for seed in SEEDS:
            name = f'{site}_{seed}'
            model = slate_for(slate, site, seed)
            nn = self_forecast(h, site, seed, hashes[name], producer)
            with (np.load(study.checked(previous['forecasts'][name]), allow_pickle=False) as old,
                  np.load(study.checked(slate['forecasts'][name]), allow_pickle=False) as origin):
                for k, x in zip(('q', 'cv', 'ols4'), (q, cv, ols), strict=True):
                    np.testing.assert_array_equal(old[k], x)
                np.testing.assert_array_equal(origin['q'], q)
                np.testing.assert_array_equal(origin['cv'], cv)
                native, original_additive = origin['joint'], origin['matched_independent']
                before, replay = parent.replay(model, h, q, cv, ols, bank, native, original_additive)
                for k in ('joint', 'matched_independent'):
                    np.testing.assert_array_equal(before[k], old[k])
                after, changes = parent.replay(model, h, q, cv, ols, np.concatenate((bank, nn[:, None]), axis=1),
                                               native, original_additive)
            data = dict(q=q, cv=cv, ols4=ols)
            info = dict(base=replay, expanded=changes)
            for arm, k in zip(ARMS[:2], ('matched_independent', 'joint'), strict=True):
                data[arm] = before[k]
                expanded = f'expanded_{arm.removeprefix("base_")}'
                data[expanded] = after[k]
                neural = scaled_to_path(cv, nn, after[k])
                analytic = np.column_stack([scaled_to_path(cv, a, after[k]) for a in bank.swapaxes(0, 1)]).any(1)
                info[expanded] = dict(neural_only_edits=int((neural & ~analytic).sum()),
                                      changed_agents=int((np.abs(before[k]-after[k]) > 1e-9).any((1, 2)).sum()))
            banks[name] = exp.previous.save_bank(LOCAL/f'{name}.npz', data)
            stats[name] = info
            tmp = LOCAL/'heartbeat.pending'
            tmp.write_text(study.json.dumps(dict(pid=os.getpid(), phase='inference', last=name,
                                               completed=len(banks), total=21))+'\n')
            tmp.replace(LOCAL/'heartbeat.json')
            print(name, {k: info[k] for k in ARMS[2:]}, flush=True)
    base.immutable(OUT/'freeze.json', dict(registration=key, banks=banks, assignment_stats=stats,
                                          outcomes_accessed=False, new_fits=0))


def recording(paths, y, valid, q, cut):
    cv, ols = paths['cv'], paths['ols4']
    weight = 1./np.bincount(q)[q]
    mass = float(weight @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    def effect(ref, pred):
        return float(100.*(weight @ lower(ref, pred, y, valid))/mass) if mass > 0. else None
    arms, direct = {}, {}
    for k in ARMS:
        p = paths[k]
        m = study.policy(p, cv, cv, y, valid, np.repeat('record', len(cv)), easy_cut=cut[0], hard_cut=cut[1])
        risk = m['recordings']['record']
        both = (effect(cv, p), effect(ols, p))
        arms[k] = dict(edits=risk['selected'], risk=risk['supported'],
            dual=bool(risk['supported'] and None not in both and min(both) > 0.),
            lower_vs_cv_pp=both[0], lower_vs_ols4_pp=both[1],
            easy_degradation=risk['known_easy_net_degradation'], ade=m['known_ADE_pixel'],
            fde=m['available_endpoint_FDE_diagnostic_pixel'])
    if len({v['edits'] for v in arms.values()}) != 1:
        raise ValueError('Expanded menu changed realized edit count')
    for a, b in CONTRASTS:
        direct[f'{a}_vs_{b}'] = effect(paths[b], paths[a]) if arms[a]['edits'] else None
    return arms, direct


def score(key, cohorts, baseline):
    fp = OUT/'freeze.json'
    frozen = study.read(fp)
    if frozen['registration'] != key or frozen['outcomes_accessed'] or len(frozen['banks']) != 21:
        raise ValueError('Require all predictions before loading held outcomes')
    data = exp.loader(cohorts)
    sites = {}
    for site in SITES:
        d = data[site]
        cut = np.quantile(np.concatenate([v['e'] for s, v in data.items() if s != site]), [.25, .75])
        rows, pairs = [], []
        for seed in SEEDS:
            with np.load(study.checked(frozen['banks'][f'{site}_{seed}']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], d['q'])
                paths = {k: z[k] for k in ('cv', 'ols4', *ARMS)}
            for r in np.unique(d['rec']):
                at = d['rec'] == r
                m, p = recording({k: x[at] for k, x in paths.items()}, d['y'][at], d['v'][at],
                                 np.unique(d['q'][at], return_inverse=True)[1], cut)
                rows.append(m)
                pairs.append(p)
        values = {k: [r[k] for r in rows] for k in ARMS}
        counts = {k: {f: sum(v[f] for v in x) for f in ('edits', 'risk', 'dual')} for k, x in values.items()}
        for a, old in zip(ARMS[:2], ('matched_independent', 'joint'), strict=True):
            expected = baseline['groups'][site]['arms'][old]
            if counts[a] != dict(edits=expected['edits'], risk=expected['risk_pass'], dual=expected['dual_pass']):
                raise ValueError('Base-menu outcomes do not replay')
        means = {k: [p[k] for p in pairs if p[k] is not None] for k in pairs[0]}
        sites[site] = dict(views=len(rows), counts=counts, estimable={k: len(v) for k, v in means.items()},
            contrast_pp={k: float(np.mean(v)) if v else None for k, v in means.items()},
            worst_signed_easy={k: max((v['easy_degradation'] for v in x if v['easy_degradation'] is not None), default=None)
                               for k, x in values.items()})
        if means['base_joint_vs_base_additive']:
            np.testing.assert_allclose(sites[site]['contrast_pp']['base_joint_vs_base_additive'],
                                       baseline['groups'][site]['lower_pp'], atol=1e-9, rtol=1e-9)
    def interval(key):
        x = np.array([v['contrast_pp'][key] for v in sites.values() if v['contrast_pp'][key] is not None])
        d = np.random.default_rng(3183).choice(x, size=(3000, len(x))).mean(1)
        return dict(mean_pp=float(x.mean()), ci95_pp=np.quantile(d, [.025, .975]).tolist(), locations=len(x))
    results = dict(result_source='fresh_run', fits='cached_verified', freeze=study.artifact(fp), by_site=sites,
        counts={k: dict(sum((Counter(v['counts'][k]) for v in sites.values()), Counter())) for k in ARMS},
        contrasts={k: interval(k) for k in next(iter(sites.values()))['contrast_pp']},
        neural_only_edits={k: sum(v[k]['neural_only_edits'] for v in frozen['assignment_stats'].values()) for k in ARMS[2:]},
        recordings=sum(v['recordings'] for v in baseline['groups'].values()), seeds=SEEDS,
        independent_confirmation=False, reserved_outcomes_opened=False, new_training=False, deployment_changed=False)
    base.immutable(OUT/'readout.json', results)
    print(study.json.dumps({k: results[k] for k in ('counts', 'contrasts', 'neural_only_edits')}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, prior, old, slate, hashes, producer = register()
        if a.phase == 'freeze':
            freeze(key, cohorts, prior, slate, hashes, producer)
        else:
            score(key, cohorts, old)


if __name__ == '__main__':
    main()
