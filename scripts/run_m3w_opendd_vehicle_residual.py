"""Train and assess matched vehicle residuals on existing OpenDD held-site folds."""

import argparse
from collections import Counter
from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_location as prior
from scripts import build_m3w_songdo_prefix_pack as indices
from src.data_unification.m3w_topdown_source_support import grid_indices
from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_vehicle_residual as model
from src.world_model.m3w_wepdtof_matched_policy import analytic
import numpy as np
import torch

OUT = prior.PUBLIC.parent / 'opendd_vehicle_residual_v1'
LOCAL = prior.PRIVATE.parent / OUT.name
SITES = tuple(f'rdb{i}' for i in range(1, 8))
CLASSES = ('Car', 'Medium Vehicle', 'Heavy Vehicle', 'Motorcycle', 'Bus')
ARMS = ('observed', 'completion')
STEPS, BATCH, SAVE = 400, 128, 100


def beat(phase, **kw):
    msg = dict(pid=os.getpid(), phase=phase, **kw)
    prior.write_json(LOCAL / 'heartbeat.json', msg)
    print(json.dumps(msg), flush=True)


def register():
    rows = json.loads(prior.SOURCE.read_text())['opendd']['sites']
    if [r['site'] for r in rows] != list(SITES) or any(r['issues'] for r in rows):
        raise ValueError('Require the existing seven OpenDD groups')
    files = (Path(__file__), Path(model.__file__), Path(model.motion.__file__),
             Path(indices.__file__), Path(indices.audit.__file__), prior.SOURCE, OUT / 'protocol.md')
    reg = OUT / 'registration.json'
    prior.write_json(reg, dict(files={str(p.relative_to(prior.ROOT)): prior.sha(p) for p in files},
        sites=SITES, classes=CLASSES, seeds=prior.SEEDS, steps=STEPS, batch=BATCH, checkpoint_every=SAVE,
        observed_steps=8, future_steps=12, stride=12, arms=ARMS, original_2pct=True,
        optimizer='AdamW 0.001', role='OpenDD vehicle-only development, original location-held groups',
        songdo_read=False, confirmation_read=False, workers=0, threads=4), immutable=True)
    return prior.sha(reg), rows


def extract_flight(db, name):
    if not re.fullmatch(r'rdb[1-7]_\d+', name):
        raise ValueError('Unexpected recording name')
    start, end = grid_indices(db.execute(f'SELECT MIN(TIMESTAMP),MAX(TIMESTAMP) FROM "{name}"').fetchone())[0]
    classes = dict(db.execute(f'SELECT CLASS,COUNT(*) FROM "{name}" GROUP BY CLASS'))
    sql = (f'SELECT OBJID,TIMESTAMP,UTM_X,UTM_Y FROM "{name}" WHERE CLASS IN (?,?,?,?,?) '
           'ORDER BY OBJID,TIMESTAMP')
    data = np.asarray(db.execute(sql, CLASSES).fetchall(), dtype=float).reshape(-1, 4)
    if not len(data):
        return None, classes
    if not np.isfinite(data).all() or not np.all(data[:, 0] == data[:, 0].astype(np.int64)):
        raise ValueError('Invalid coordinate or identity')
    frames = grid_indices(data[:, 1])[0] - start
    ids = data[:, 0].astype(np.int64)
    agent, begin, _, take = indices.exact_indices(ids, frames, np.arange(len(ids)), int(end-start))
    if not len(agent):
        return None, classes
    h = data[take[:, :8], 2:]
    y_idx = take[:, 8:]
    valid = y_idx >= 0
    y = np.where(valid[..., None], data[np.maximum(y_idx, 0), 2:] - h[:, -1:], 0.)
    _, inv, count = np.unique(begin, return_inverse=True, return_counts=True)
    return dict(history=h-h[:, -1:], future=y, valid=valid, weight=1./count[inv],
                agent=agent, query_counts=count), classes


def extract(key, rows):
    files = {}
    for row in rows:
        site = row['site']
        p = LOCAL / f'{site}_cohort.npz'
        receipt = p.with_suffix('.json')
        zfile = prior.ARCHIVES / row['asset']['archive']
        if not receipt.exists():
            if zfile.stat().st_size != row['asset']['bytes'] or prior.sha(zfile) != row['asset']['sha256']:
                raise ValueError(f'Unexpected archive {site}')
            chunks, classes = [], Counter()
            with ZipFile(zfile) as z, TemporaryDirectory(dir=LOCAL) as tmp:
                member, = [n for n in z.namelist() if Path(n).name == f'trajectories_{site}_v3.sqlite']
                db = Path(tmp) / 'tracks.sqlite'
                with z.open(member) as src, db.open('wb') as out:
                    shutil.copyfileobj(src, out, 1024*1024)
                with closing(sqlite3.connect(f'{db.as_uri()}?mode=ro&immutable=1', uri=True)) as conn:
                    conn.execute('PRAGMA trusted_schema=OFF')
                    conn.execute('PRAGMA query_only=ON')
                    names = [t for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                             if re.fullmatch(rf'{site}_\d+', t)]
                    if len(names) != row['recordings']:
                        raise ValueError(f'Unexpected flight roster: {site}')
                    for i, name in enumerate(names, start=1):
                        d, class_counts = extract_flight(conn, name)
                        classes.update(class_counts)
                        if d is not None:
                            chunks.append(dict(**d, recording=np.full(len(d['agent']), name)))
                        if i % 20 == 0:
                            beat('extract', site=site, flights=i, total=len(names))
            data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
            for k in ('history', 'future'):
                data[k] = data[k].astype(np.float32)
            sha = indices.save_npz(p, data)
            prior.write_json(receipt, dict(registration=key, sha256=sha, rows=len(data['agent']),
                complete=int(data['valid'].all(axis=1).sum()), queries=len(data['query_counts']),
                recordings=len(chunks), archive=row['asset']['sha256'], observed_publisher_class_rows=dict(classes)), immutable=True)
        info = json.loads(receipt.read_text())
        if info['registration'] != key or info['archive'] != row['asset']['sha256'] or info['sha256'] != prior.sha(p):
            raise ValueError(f'Changed cached cohort: {site}')
        files[site] = info
        beat('cohort_ready', site=site, rows=info['rows'], recordings=info['recordings'])
    prior.write_json(OUT / 'cohorts.json', dict(registration=key, sites=files, songdo_read=False), immutable=True)


def load(key):
    receipt = json.loads((OUT / 'cohorts.json').read_text())
    if receipt['registration'] != key or receipt['songdo_read'] or set(receipt['sites']) != set(SITES):
        raise ValueError('Require the registered vehicle cohorts')
    result, identities = {}, set()
    for site, meta in receipt['sites'].items():
        p = LOCAL / f'{site}_cohort.npz'
        if prior.sha(p) != meta['sha256']:
            raise ValueError(f'Cohort changed: {site}')
        with np.load(p, allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        ids = set(d['agent'].tolist())
        if identities & ids:
            raise ValueError('Publisher object IDs repeat across location folds')
        identities.update(ids)
        x, cv, rotation, scale = model.encode(d['history'])
        y = ((d['future'] @ rotation) / scale[:, None, None]).astype(np.float32)
        result[site] = dict(**d, rotation=rotation, scale=scale, tensors=tuple(map(torch.from_numpy, (x, cv, y, d['valid']))))
    return result


def fit(key, sites, held, seed, arm):
    donors = [s for s in SITES if s != held]
    identity = dict(registration=key, held=held, donors=donors, seed=seed, arm=arm)
    name = f'{held}_{seed}_{arm}'
    path = LOCAL / f'{name}.pt'
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, SITES.index(held)])
    net = model.motion.network(17)
    optim = torch.optim.AdamW(net.parameters(), lr=.001)
    begin = 0
    if path.exists():
        state = torch.load(path, weights_only=True, map_location='cpu')
        if state['identity'] != identity:
            raise ValueError(f'Cannot resume a different fold: {name}')
        begin = state['step']
        net.load_state_dict(state['weights'])
        optim.load_state_dict(state['optim'])
        rng.bit_generator.state = state['rng']
    for step in range(begin+1, STEPS+1):
        pick = rng.integers(len(donors), size=BATCH)
        groups = []
        for i, s in enumerate(donors):
            t = sites[s]['tensors']
            row = rng.integers(len(t[0]), size=int((pick == i).sum()))
            groups.append(tuple(z[row] for z in t))
        x, cv, y, valid = (torch.cat(z) for z in zip(*groups, strict=True))
        optim.zero_grad()
        value = model.loss(model.predict(net, x, cv), cv, y, valid, completion=arm=='completion')
        if not torch.isfinite(value):
            raise ValueError(f'Nonfinite training loss: {name}/{step}')
        value.backward()
        optim.step()
        if step % SAVE == 0 or step == STEPS:
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, weights=net.state_dict(), optim=optim.state_dict(),
                            rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat('fit', fold=name, step=step, loss=float(value.detach()))
    return dict(identity=identity, sha256=prior.sha(path), step=STEPS)


def train(key):
    sites = load(key)
    cells = {}
    for held in SITES:
        for seed in prior.SEEDS:
            for arm in ARMS:
                name = f'{held}_{seed}_{arm}'
                cells[name] = fit(key, sites, held, seed, arm)
    prior.write_json(OUT / 'freeze.json', dict(registration=key, checkpoints=cells, songdo_read=False,
        selection='all fixed-step checkpoints, no selection on held-location labels', updates=len(cells)*STEPS), immutable=True)


def metric(pred, cv, ols, d, cut):
    y, v, w = d['future'], d['valid'], d['weight']
    mass = float(w @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(axis=1))
    if mass <= 0:
        raise ValueError('No nonzero CV error mass in the held group')
    def gain(ref):
        known, lower = paired(ref, pred, y, v)
        return [float(100. * (w @ z.sum(axis=1)) / mass) for z in (known, lower)]
    p = policy(pred, cv, cv, y, v, d['recording'], easy_cut=cut[0], hard_cut=cut[1])
    return dict(cv_gain_known_lower_pp=gain(cv), ols_gain_known_lower_pp=gain(ols),
                known_ade_native=p['known_ADE_pixel'], fde_available_native=p['available_endpoint_FDE_diagnostic_pixel'],
                easy_gain_percent=p['subsets']['easy']['gain_percent'], p95_ade_native=p['p95_known_ADE_pixel'],
                selected_rate=p['switch_rate'], risk_passes=sum(x['supported'] for x in p['recordings'].values()),
                recordings=len(p['recordings']))


def score(key):
    p = OUT / 'freeze.json'
    f = json.loads(p.read_text())
    if f['registration'] != key or len(f['checkpoints']) != 42 or f['songdo_read']:
        raise ValueError('Require completed frozen fits')
    sites = load(key)
    references = {s: analytic(d['history']) for s, d in sites.items()}
    def complete_errors(s):
        d = sites[s]
        at = d['valid'].all(axis=1)
        e = np.linalg.norm(references[s][0][at]-d['future'][at], axis=-1).mean(axis=1)
        return e[e > 0.]
    complete = {s: complete_errors(s) for s in sites}
    results = {}
    for s, d in sites.items():
        cut = np.quantile(np.concatenate([complete[k] for k in sites if k != s]), [.25, .75])
        cv, ols = references[s]
        r = dict(rows=len(cv), complete=int(d['valid'].all(axis=1).sum()), controls={
            'cv': metric(cv, cv, ols, d, cut), 'ols4': metric(ols, cv, ols, d, cut)}, seeds={})
        x, cv_local = d['tensors'][:2]
        for seed in prior.SEEDS:
            arm_result, paths = {}, {}
            for arm in ARMS:
                name = f'{s}_{seed}_{arm}'
                chk = LOCAL / f'{name}.pt'
                receipt = f['checkpoints'][name]
                if receipt['sha256'] != prior.sha(chk) or receipt['identity']['donors'] != [k for k in SITES if k != s]:
                    raise ValueError('Held site entered a fit or a checkpoint changed')
                state = torch.load(chk, weights_only=True, map_location='cpu')
                net = model.motion.network(17).eval()
                net.load_state_dict(state['weights'])
                with torch.inference_mode():
                    local = torch.cat([model.predict(net, a, b) for a, b in zip(x.split(4096), cv_local.split(4096), strict=True)]).numpy()
                pred = (local * d['scale'][:, None, None]) @ d['rotation'].transpose(0, 2, 1)
                arm_result[arm] = metric(pred, cv, ols, d, cut)
                paths[arm] = pred
            _, lo = paired(paths['observed'], paths['completion'], d['future'], d['valid'])
            mass = d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(axis=1)
            r['seeds'][str(seed)] = dict(arms=arm_result, completion_vs_observed_lower_pp=float(100. * (d['weight'] @ lo.sum(axis=1)) / mass))
        results[s] = r
        beat('score', site=s, complete_sites=len(results))
    def interval(values):
        x = np.array(values)
        z = np.random.default_rng(3183).choice(x, (3000, len(x))).mean(axis=1)
        return dict(mean=float(x.mean()), ci95=np.quantile(z, [.025, .975]).tolist())
    summary = {}
    for arm in ARMS:
        summary[arm] = {ref: interval([np.mean([z['arms'][arm][ref][1] for z in r['seeds'].values()])
                                      for r in results.values()]) for ref in ('cv_gain_known_lower_pp', 'ols_gain_known_lower_pp')}
    summary['completion_vs_observed'] = interval([np.mean([v['completion_vs_observed_lower_pp'] for v in r['seeds'].values()])
                                                  for r in results.values()])
    prior.write_json(OUT / 'readout.json', dict(registration=key, freeze=prior.sha(p), sites=results, summary=summary,
        original_2pct_unchanged=True, songdo_read=False, reserved_outcomes_opened=False,
        deployment_changed=False, independent_confirmation=False, submission_ready=False), immutable=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('extract', 'fit', 'score'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, rows = register()
        if args.phase == 'extract':
            extract(key, rows)
        elif args.phase == 'fit':
            train(key)
        else:
            score(key)
