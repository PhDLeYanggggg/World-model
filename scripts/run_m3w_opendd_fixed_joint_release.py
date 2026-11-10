"""Freeze, calibrate and score fixed-proposal support ranking on OpenDD."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import subprocess
import sys

if __name__ == '__main__' and sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_support_rerank as prior
from src.evaluation import m3w_fixed_joint_release as policy

learn, support, pair, base, old, np, torch = (getattr(prior, k) for k in
    ('learn', 'support', 'pair', 'base', 'old', 'np', 'torch'))
ROOT = base.prior.ROOT
OUT = prior.OUT.parent/'opendd_fixed_joint_release_v1'
LOCAL = prior.LOCAL.parent/OUT.name
read, sha, write = old.read, prior.sha, prior.write
ARMS = policy.ARMS


def register():
    key, source, _, _, _ = learn.register()
    skey, other, _, _ = support.register()
    f, s = learn.OUT/'freeze.json', support.OUT/'freeze.json'
    proposals, heads = read(f), read(s)
    cohort = base.parent.OUT/'cohorts.json'
    report = learn.OUT/'readout.json'
    if (source != other or proposals['registration'] != key or heads['registration'] != skey
            or not proposals['held_choices_frozen'] or not heads['held_choices_frozen'] or heads['songdo_read']
            or len(proposals['cells']) != 63 or set(proposals['cells']) != set(heads['cells'])
            or read(report)['freeze'] != sha(f) or read(cohort)['registration'] != source):
        raise ValueError('Require the complete frozen, source-excluded OpenDD banks')
    refs = (Path(__file__), Path(policy.__file__), OUT/'protocol.md', f, s, cohort, report)
    binding = dict(files={str(k.relative_to(ROOT)): sha(k) for k in refs}, seeds=base.SEEDS,
                   arms=ARMS, original_2pct=True, external_outcomes_read=False, fit=False)
    write(OUT/'registration.json', binding, immutable=True)
    return sha(OUT/'registration.json'), proposals['cells'], heads['cells'], read(cohort)['sites']


def require_published(path):
    ref = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, timeout=25).split()
    if (len(ref) != 2 or subprocess.check_output(['git', 'show', f'{ref[0].decode()}:{path.relative_to(ROOT)}'],
            cwd=ROOT, timeout=10) != path.read_bytes()):
        raise ValueError('Publish the frozen phase receipt before proceeding')


def heartbeat(phase, n, total=63):
    if n == 1 or n % 10 == 0 or n == total:
        write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=total))
        print(f'{phase}: {n}/{total}', flush=True)


def cohort_rows(cohorts, fields):
    for site, r in cohorts.items():
        path = base.parent.LOCAL/f'{site}_cohort.npz'
        if sha(path) != r['sha256']:
            raise ValueError('Cohort changed')
        with np.load(path, allow_pickle=False) as z:
            yield site, {k: z[k] for k in fields}


@torch.inference_mode()
def freeze_scores(key, cells, heads, cohorts):
    history = {site: d['history'] for site, d in cohort_rows(cohorts, ('history',))}
    packets = {}
    for name, c in cells.items():
        h = heads[name]
        fit, excluded = set(c['fit_sites']), set(c['packets'])
        ident = h['checkpoint']['identity']
        if (fit & excluded or fit | excluded != set(base.SITES) or h['cut'] != c['cut']
                or ident != h['identity'] or set(ident['fit_sites']) != fit or set(ident['excluded']) != excluded
                or ident['seed'] != c['seed'] or any(set(c['weights'][a]['identity']['fit_sites']) != fit
                    or set(c['weights'][a]['identity']['excluded']) != excluded for a in ('additive', 'joint'))):
            raise ValueError('Training sites must exclude both calibration and held sites')
        net = base.load_net(support.LOCAL/f'{name}_support.pt', h['checkpoint'], support.support_model(), sorted(excluded))
        for site, ref in c['packets'].items():
            with prior.packet(learn.LOCAL, ref) as z:
                features, _ = support.features(history[site], c['cut'][0])
                p = net(torch.from_numpy(features)).squeeze(1).sigmoid().numpy()
                score = policy.scores(z['joint_scores'], p, z['joint_rows'])
                if not np.array_equal(score[0], z['joint_scores']):
                    raise ValueError('Original ordering changed')
            file = LOCAL/f'{name}_{site}.npz'
            packets[f'{name}:{site}'] = base.parent.indices.save_npz(file, dict(score=score))
        heartbeat('freeze_scores', len(packets)//2)
    write(OUT/'score_freeze.json', dict(registration=key, packets=packets,
          outcome_fields_loaded=False, new_training=False, independent_confirmation=False), immutable=True)


def scores(name, site, frozen):
    path = LOCAL/f'{name}_{site}.npz'
    if sha(path) != frozen['packets'][f'{name}:{site}']:
        raise ValueError('Frozen release scores changed')
    with np.load(path, allow_pickle=False) as z:
        return z['score']


def labels(key, cells, cohorts, frozen, *, evaluate):
    if (frozen['registration'] != key or frozen['outcome_fields_loaded'] or len(frozen['packets']) != 126):
        raise ValueError('Require all scores frozen without outcomes')
    thresholds = None
    if evaluate:
        p = OUT/'calibration.json'
        require_published(p)
        thresholds = read(p)
        if thresholds['score_freeze_sha256'] != sha(OUT/'score_freeze.json'):
            raise ValueError('Calibration references other scores')
    data = dict(cohort_rows(cohorts, ('future', 'valid', 'recording', 'weight')))
    old_rows = {(v['site'], v['seed'], v['calibration_site']): v['arms']['joint']
                for v in read(learn.OUT/'readout.json')['views']}
    result = {}
    for n, (name, c) in enumerate(cells.items(), 1):
        for site, ref in c['packets'].items():
            other = c['rules'][site]['calibration_site']
            if site == other or set((site, other)) != set(c['packets']):
                raise ValueError('Calibration and held roles overlap')
            mode = site if evaluate else other
            rank, d = scores(name, mode, frozen), data[mode]
            ref = c['packets'][mode]
            with prior.packet(learn.LOCAL, ref) as z:
                joint = pair.prediction(z, 'joint')
                y = pair.evaluate(joint, z, d, c['cut'])
                if evaluate:
                    r = old_rows[site, c['seed'], other]
                    if [len(y['dual']), y['risk'].sum(), y['dual'].sum(), y['missing'].sum()] != [
                            r[k] for k in ('matched', 'matched_risk', 'matched_dual', 'matched_missing')]:
                        raise ValueError('Original joint matched results changed')
                    cuts = thresholds['directions'][f'{name}:{site}']
                    if cuts['calibration_site'] != other or cuts['seed'] != c['seed']:
                        raise ValueError('Calibrated threshold has changed roles')
                    entry = score_direction(rank, y, d, z, cuts['arms'])
                else:
                    entry = dict(arms={a: dict(zip(('threshold', 'selected'), base.calibration(s, y['risk'])))
                                      for a, s in zip(ARMS, rank, strict=True)})
            result[f'{name}:{site}'] = dict(**entry, held_site=site, calibration_site=other, seed=c['seed'])
        heartbeat('held_score' if evaluate else 'calibrate', n)
    return result


def score_direction(rank, y, d, z, rules):
    cv, rows, paths = (z[k] for k in ('cv', 'joint_rows', 'joint_paths'))
    mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(1))
    result = {}
    for method in ('fixed_quarter', 'calibrated'):
        summaries, preds = {}, {}
        for arm, s in zip(ARMS, rank, strict=True):
            t = rules[arm]['threshold']
            used = policy.quota(s) if method == 'fixed_quarter' else (
                s >= t if t is not None else np.zeros(s.shape, dtype=bool))
            preds[arm] = policy.prediction(cv, rows, paths, used)
            summaries[arm] = dict(selected=int(used.sum()), risk_pass=int((used & y['risk']).sum()),
                                  dual=int((used & y['dual']).sum()), undefined_easy=int((used & y['missing']).sum()))
        paired = {a: pair.paired(preds[a], preds['original'], d, z['bounds'], mass) for a in ARMS[1:]}
        if method == 'fixed_quarter':
            if len({v['selected'] for v in summaries.values()}) != 1:
                raise ValueError('Different edit budget in fixed-count contrast')
            for a, v in paired.items():
                v.update({k+'_rate_pp': 100.*(summaries[a][k]-summaries['original'][k])/summaries[a]['selected']
                          for k in ('risk_pass', 'dual', 'undefined_easy')})
        result[method] = dict(arms=summaries, paired=paired)
    return result


def summarize(views):
    draws = np.random.default_rng(12773).integers(7, size=(3000, 7))
    result = {}
    for method in ('fixed_quarter', 'calibrated'):
        tally = {a: {k: sum(v[method]['arms'][a][k] for v in views) for k in
                    ('selected', 'risk_pass', 'dual', 'undefined_easy')} for a in ARMS}
        ci = {}
        for a in ARMS[1:]:
            ci[a] = {}
            for k in views[0][method]['paired'][a]:
                x = np.array([np.mean([v[method]['paired'][a][k] for v in views if v['held_site'] == site])
                              for site in base.SITES])
                ci[a][k] = dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                                location_values=x.tolist())
        result[method] = dict(counts=tally, contrasts=ci)
    return result


def main(phase):
    key, cells, heads, cohorts = register()
    if phase == 'freeze':
        return freeze_scores(key, cells, heads, cohorts)
    p = OUT/'score_freeze.json'
    require_published(p)
    rows = labels(key, cells, cohorts, read(p), evaluate=phase == 'score')
    if phase == 'calibrate':
        write(OUT/'calibration.json', dict(score_freeze_sha256=sha(p), directions=rows), immutable=True)
        return
    summary = summarize(list(rows.values()))
    write(OUT/'readout.json', dict(result_source='fresh_run', registration=key,
        score_freeze_sha256=sha(p), calibration_sha256=sha(OUT/'calibration.json'), summary=summary,
        directions=len(rows), locations=len(base.SITES), seeds=base.SEEDS, paired_location_bootstrap=3000,
        original_2pct=True, fit=False, external_outcomes_read=False,
        independent_confirmation=False, deployment_changed=False), immutable=True)
    print(summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'calibrate', 'score'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.phase)
