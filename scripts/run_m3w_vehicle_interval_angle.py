"""Fit and evaluate an interval-target angle selector on exposed OpenDD data."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_pair_angle as prior
from src.world_model import m3w_vehicle_interval_angle as interval

old, short, netmodel = prior.old, prior.short, prior.model
base, np, torch, screen = prior.base, prior.np, prior.torch, prior.screen
action, nested, chooser_parent = prior.action, prior.nested, prior.prior
read, write, sha = prior.read, prior.write, prior.sha
ROOT = base.prior.ROOT
OUT = prior.OUT.parent/'opendd_vehicle_interval_angle_v1'
LOCAL = prior.LOCAL.parent/OUT.name


def register():
    key, data, bank, parent, teachers, chooser, _, _, _ = prior.register()
    f, r = (prior.OUT/f'{k}.json' for k in ('freeze', 'readout'))
    direct, result = read(f), read(r)
    if (direct['registration'] != key or result['freeze'] != sha(f)
            or not result['original_2pct'] or result['external_outcomes_read'] or result['directions'] != 126):
        raise ValueError('Require completed direct-target control with original rules')
    paths = (Path(__file__), Path(interval.__file__), f, r, OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(inputs={str(x.resolve().relative_to(ROOT)): sha(x) for x in paths}, seeds=base.SEEDS,
                  steps=prior.STEPS, batch=256, torch_threads=4, loader_workers=0, development_only=True),
          immutable=True)
    return sha(p), data, bank, parent, teachers, chooser, direct, result


@torch.inference_mode()
def donor(d, site, excluded, seed, bank, teachers, chooser, cut):
    ex = sorted({*excluded, site})
    if len(ex) != 3 or site in excluded:
        raise ValueError('The donor and both outer sites must be excluded from upstream fitting')
    tag = nested.tag(ex, seed)
    ego = base.load_net(chooser_parent.prior.LOCAL/f'{tag}_ego.pt', teachers['checkpoints']['ego'][tag],
                        action.model.AngleValue(False), ex)
    additive = base.load_net(chooser_parent.LOCAL/f'{tag}_additive.pt', chooser['fits'][tag]['checkpoint'],
                             action.study.model.PairValue(False), ex)
    t = base.triple_id(ex, seed)
    forecast = base.load_net(base.LOCAL/f'{t}.pt', bank['triple_checkpoints'][t],
                             nested.old.motion.motion.network(17), ex)
    d_y = action.labels(d['file'])
    m = action.study.menu(nested.data_slice(d, d_y), forecast)
    _, rec, off, cap = screen.offsets(m['cv'], m['ols4'], d_y, cut)
    x, labels = [], []
    for lo, hi in m['slots'].values():
        q = slice(lo, hi)
        rows, families, i, j = action.study.choices(m['peak'], lo, hi)
        h = additive.head(additive.actor(torch.from_numpy(m['x'][rows, families]))).numpy()
        k = int(action.study.model.merit(.5*(h[i]+h[j])).argmax())
        pick = [i[k], j[k]]
        sel = rows[pick]-lo
        cv, ols = m['cv'][q], m['ols4'][q]
        anchor = cv.copy()
        anchor[sel] = m['paths'][rows[pick], families[pick]]
        feat, _ = action.inputs(cv, anchor, sel, d, q)
        indices, margin = short.model.order(ego(torch.from_numpy(feat)).numpy())
        paths = short.options(cv, anchor, sel, indices)
        y, valid, w = (d_y[k][q] for k in ('future', 'valid', 'weight'))
        if not np.all(w == w[0]):
            raise ValueError('The original comparison assumes equal weights within a query')
        _, dual, _, _, _ = short.outcomes(cv, ols, anchor, paths, sel, y, valid, w, cut, off[rec[lo]], cap[rec[lo]])
        scales = prior.scale(paths)
        unit = scales[1:, 1].clip(1e-12)
        target = prior.targets(paths, y, valid, scales, dual)
        bounds = (interval.relative_bounds(paths[2:], paths[1], y, valid)/unit[:, None]).clip(-1., 1.)
        np.testing.assert_allclose(bounds[:, 0], target[:, 3], rtol=1e-6, atol=1e-7)
        x.append(netmodel.features(short.feature_rows(feat, indices, margin, hi-lo), scales)[0])
        labels.append(np.column_stack((target[:, 2:5], bounds[:, 1])).astype(np.float32))
    return tuple(torch.from_numpy(np.concatenate(v)) for v in (x, labels))


def heartbeat(phase, n, total):
    write(LOCAL/'progress.json', dict(pid=os.getpid(), phase=phase, completed=n, total=total))
    if n % 7 == 0 or n == total:
        print(f'{phase}: {n}/{total}', flush=True)


def fit(ident, train):
    saved = nested.old.LOCAL
    try:
        nested.old.LOCAL = LOCAL
        torch.manual_seed(ident['seed'])
        return nested.old.fit(ident, train, interval.loss, netmodel.network(), prior.STEPS, 256)
    finally:
        nested.old.LOCAL = saved


def fit_all(key, data, bank, parent, teachers, chooser, direct):
    shortlist = read(short.OUT/'freeze.json')
    anchors = read(action.OUT/'freeze.json')
    cells = {}
    for name, cell in parent['cells'].items():
        seed, excluded = cell['seed'], sorted(cell['packets'])
        donors = direct['cells'][name]['donors']
        if set(donors) != set(base.SITES)-set(excluded):
            raise ValueError('Changed fit site roles')
        ident = dict(name=name, registration=key, excluded=excluded, fit_sites=donors, seed=seed)
        path = LOCAL/f'{name}.json'
        if not path.exists():
            train = [donor(data[s], s, excluded, seed, bank, teachers, chooser, cell['cut']) for s in donors]
            labels = torch.cat([y for _, y in train])
            if len(labels) != direct['cells'][name]['training']['rows'] or torch.any(labels[:, 1] > labels[:, 3]+1e-7):
                raise ValueError('Training row roster or interval endpoints changed')
            status = dict(rows=len(labels), point=int((labels[:, 3] == labels[:, 1]).sum()),
                          crosses_zero=int(((labels[:, 1] < -1e-10) & (labels[:, 3] > 1e-10)).sum()))
            net, fit_ref = fit(ident, train)
            e = anchors['cells'][name]['checkpoints']['ego']
            ego = base.load_net(action.LOCAL/f"{e['identity']['name']}.pt", e, action.model.AngleValue(False), excluded)
            control, = [r for r in direct['cells'][name]['checkpoints']
                        if r['identity']['name'].endswith('_direct')]
            other = base.load_net(prior.LOCAL/f"{control['identity']['name']}.pt",
                                  control, netmodel.network(), excluded)
            packets = {}
            for site, upstream in cell['packets'].items():
                zref = shortlist['cells'][name]['packets'][site]
                dref = direct['cells'][name]['packets'][site]
                with (screen.packet_file(short.ctx.study.LOCAL, upstream) as p, screen.packet_file(short.LOCAL, zref) as z,
                      screen.packet_file(prior.LOCAL, dref) as d):
                    a = prior.propose(p, z, data[site], ego, [net, other])
                    for k in ('trajectory', 'agreement', 'score'):
                        np.testing.assert_array_equal(a[f'direct_{k}'], d[f'direct_{k}'])
                    output = {k: a[f'anchor_difference_{k}'] for k in ('trajectory', 'agreement', 'score')}
                dest = LOCAL/f'{name}_{site}.npz'
                packets[site] = dict(file=dest.name, sha256=base.parent.indices.save_npz(dest, output),
                                     shortlist=zref['sha256'])
            write(path, dict(identity=ident, fit=fit_ref, packets=packets, training=status), immutable=True)
        row = read(path)
        if (row['identity'] != ident or sha(LOCAL/f'{name}.pt') != row['fit']['sha256']
                or any(sha(LOCAL/v['file']) != v['sha256'] for v in row['packets'].values())):
            raise ValueError('Changed interval fit or frozen forecast')
        cells[name] = row
        heartbeat('fit', len(cells), len(parent['cells']))
    write(OUT/'freeze.json', dict(registration=key, cells=cells, before_evaluation=True,
                                external_outcomes_read=False), immutable=True)


def require_remote(path):
    commit, ref = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, timeout=25).split()
    if ref != b'refs/heads/main' or subprocess.check_output(
            ['git', 'show', f'{commit.decode()}:{path.relative_to(ROOT)}'], cwd=ROOT, timeout=10) != path.read_bytes():
        raise ValueError('Publish the exact development forecast freeze before evaluation')


def cutoff(p, label, cut, z, a):
    m = old.masks(z)
    flag, gain = old.compare(p, label, cut, z, m['ego_count'] | m['fixed_15pct'])
    ix = (np.arange(len(a['agreement'])), a['agreement'])
    good = flag[ix][:, 1] & (gain[ix][:, :2] > 1e-10).all(1)
    return {q: old.model.threshold(a['score'][at], good[at]) for q, at in m.items()}


def score_all(key, data, parent, direct, baseline):
    f = OUT/'freeze.json'
    freeze = read(f)
    if (freeze['registration'] != key or not freeze['before_evaluation']
            or set(freeze['cells']) != set(parent['cells'])):
        raise ValueError('Require complete frozen interval predictions')
    require_remote(f)
    freeze_sha = sha(f)
    shortlist = read(short.OUT/'freeze.json')
    rows = []
    for name, cell in parent['cells'].items():
        cut = cell['cut']
        choices = {}
        for s, upstream in cell['packets'].items():
            a = freeze['cells'][name]['packets'][s]
            z = shortlist['cells'][name]['packets'][s]
            if a['shortlist'] != z['sha256']:
                raise ValueError('Changed fixed angle menu')
            choices[s] = (upstream, z, a)
        rules = {}
        for s, (pref, zref, aref) in choices.items():
            dest = LOCAL/f'{name}_{s}_cutoff.json'
            ident = dict(freeze=freeze_sha, site=s, packet=aref['sha256'])
            if not dest.exists():
                with (screen.packet_file(short.ctx.study.LOCAL, pref) as p,
                      screen.packet_file(short.LOCAL, zref) as z, screen.packet_file(LOCAL, aref) as a):
                    value = cutoff(p, action.labels(data[s]['file']), cut, z, a)
                write(dest, dict(identity=ident, value=value), immutable=True)
            row = read(dest)
            if row['identity'] != ident:
                raise ValueError('Changed opposite-site cutoff')
            rules[s] = row['value']
        for s, (pref, zref, aref) in choices.items():
            cal, = set(choices)-{s}
            dest = LOCAL/f'{name}_{s}_score.json'
            ident = dict(freeze=freeze_sha, site=s, calibration=cal, seed=cell['seed'], packet=aref['sha256'])
            if not dest.exists():
                dref = direct['cells'][name]['packets'][s]
                with (screen.packet_file(short.ctx.study.LOCAL, pref) as p, screen.packet_file(short.LOCAL, zref) as z,
                      screen.packet_file(LOCAL, aref) as a, screen.packet_file(prior.LOCAL, dref) as d):
                    y = action.labels(data[s]['file'])
                    # The unused utility column is set to the same first-angle control.
                    arm = {k: a[k] for k in ('trajectory', 'agreement', 'score')}
                    ctrl = {k: d[f'direct_{k}'] for k in arm}
                    for packet in (arm, ctrl):
                        packet['utility'] = np.zeros(len(packet['agreement']), dtype=int)
                    drule = direct['cells'][name]['cutoffs'][cal]['direct']
                    effects = old.metrics(p, y, cut, z, arm, rules[cal])
                    replay = old.metrics(p, y, cut, z, ctrl, drule)
                    # Existing paired routine compares its first argument called "direct"
                    # against each of two controls. Both control entries here are identical.
                    paired = prior.pair_contrasts(p, y, cut, z, dict(direct=arm, anchor_difference=ctrl, pointwise=ctrl),
                        dict(direct=rules[cal], anchor_difference=drule, pointwise=drule))['anchor_difference']
                write(dest, dict(identity=ident, interval=effects, direct=replay, paired=paired), immutable=True)
            row = read(dest)
            if row['identity'] != ident:
                raise ValueError('Changed development evaluation direction')
            rows.append(row)
        heartbeat('evaluate', len(rows), 126)
    groups = {s: [r for r in rows if r['identity']['site'] == s] for s in base.SITES}
    draws = np.random.default_rng(2917).integers(len(groups), size=(3000, len(groups)))

    def paired_ci(q, arm, metric):
        sites = np.array([np.mean([np.mean([r['paired'][q][arm][metric] for r in rr if r['identity']['seed'] == seed])
                                   for seed in base.SEEDS]) for rr in groups.values()])
        return dict(point=float(sites.mean()), ci95=np.quantile(sites[draws].mean(1), [.025, .975]).tolist(),
                    site_means=sites.tolist())

    modes = ('trajectory', 'agreement', 'calibrated_agreement')
    counts = {t: {q: {a: dict(sum((Counter(r[t]['counts'][q][a]) for r in rows), Counter()))
                      for a in modes} for q in old.QUOTAS} for t in ('interval', 'direct')}
    if counts['direct'] != baseline['counts']['direct']:
        raise ValueError('Original direct-target control counts did not replay')
    paired = {q: {a: {k: paired_ci(q, a, k) for k in old.FIELDS} for a in modes} for q in old.QUOTAS}
    status = dict(sum((Counter(c['training']) for c in freeze['cells'].values()), Counter()))
    row = dict(result_source='fresh_development_fit_and_evaluation', fits=len(freeze['cells']), directions=len(rows),
               seeds=base.SEEDS, freeze=freeze_sha, counts=counts, interval_minus_direct=paired, training=status,
               original_2pct=True, bootstrap_draws=3000, bootstrap_sites=len(groups), deployment_changed=False,
               external_outcomes_read=False, independent_confirmation=False, submission_ready=False)
    write(OUT/'readout.json', row, immutable=True)
    print(status, counts, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'score'))
    phase = cli.parse_args().phase
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use the native arm64 .venv-pytorch interpreter')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, bank, parent, teachers, chooser, direct, result = register()
        if phase == 'freeze':
            fit_all(key, data, bank, parent, teachers, chooser, direct)
        else:
            score_all(key, data, parent, direct, result)
