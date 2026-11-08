"""Hash-bound OpenDD location-held intervention diagnostic, with local resume."""
import argparse
from contextlib import closing
import fcntl
from hashlib import sha256
from itertools import groupby
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/opendd_location_diagnostic_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/opendd_location_diagnostic_v1'
ARCHIVES = ROOT / 'external_data/OpenDD_source_audit'
SOURCE = PUBLIC.parent / 'topdown_source_intake_v1/opendd.json'
AUTH = PUBLIC.parent / 'delegated_research_authorization_20260923.json'
SEEDS = (17, 29, 43)
ARMS = ('self', 'neighbor')
STEPS, BATCH, STRIDE = 300, 128, 12


def sha(path):
    with path.open('rb') as f:
        from hashlib import file_digest
        return file_digest(f, 'sha256').hexdigest()


def write_json(path, value, *, immutable=False):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if immutable and path.exists():
        if path.read_text() != encoded:
            raise ValueError(f'Cannot overwrite different registered artifact: {path}')
        return
    tmp = path.with_suffix('.pending')
    tmp.write_text(encoded)
    tmp.replace(path)


def beat(phase, **kwargs):
    state = dict(phase=phase, pid=os.getpid(), **kwargs)
    write_json(PRIVATE / 'heartbeat.json', state)
    print(json.dumps(state), flush=True)


def register():
    ledger = json.loads(AUTH.read_text())
    if not ledger['evidence_based_observation_protocol_and_data_role_decisions_delegated']:
        raise ValueError('Diagnostic role requires research delegation')
    sites = json.loads(SOURCE.read_text())['opendd']['sites']
    if [s['site'] for s in sites] != [f'rdb{i}' for i in range(1, 8)] or any(s['issues'] for s in sites):
        raise ValueError('Require seven verified OpenDD sources')
    for row in sites:
        path = ARCHIVES / row['asset']['archive']
        if path.stat().st_size != row['asset']['bytes'] or sha(path) != row['asset']['sha256']:
            raise ValueError(f'Unexpected OpenDD archive: {path.name}')
    files = (Path(__file__).resolve(), PUBLIC / 'protocol.md', SOURCE, AUTH,
             ROOT / 'src/evaluation/m3w_opendd_location.py',
             ROOT / 'src/data_unification/m3w_topdown_source_support.py',
             ROOT / 'src/world_model/m3w_ols_replacement.py')
    path = PUBLIC / 'registration.json'
    write_json(path, dict(files={str(f.relative_to(ROOT)): sha(f) for f in files},
        sites=[s['site'] for s in sites], seeds=SEEDS, stride=STRIDE, steps=STEPS,
        batch=BATCH, arms=ARMS, role='exploratory_location_held_trajectory_diagnostic',
        original_risk_budget=0.02, dronecrowd_confirmation_access=False,
        stage5c_executed=False, smc_enabled=False), immutable=True)
    return sites, sha(path)


def flight(conn, name):
    start, stop = conn.execute(f'SELECT MIN(TIMESTAMP), MAX(TIMESTAMP) FROM "{name}"').fetchone()
    ticks, _ = grid_indices([start, stop])
    count = int(ticks[1] - ticks[0] + 1)
    query = (f'SELECT OBJID,TIMESTAMP,UTM_X,UTM_Y FROM "{name}" '
             "WHERE CLASS='Pedestrian' ORDER BY OBJID,TIMESTAMP")
    tracks = []
    for ident, group in groupby(conn.execute(query), key=lambda row: row[0]):
        values = list(group)
        index, _ = grid_indices([row[1] for row in values])
        index -= ticks[0]
        xy = np.asarray([row[2:] for row in values], dtype=np.float64)
        if not np.isfinite(xy).all() or (np.diff(index) <= 0).any():
            raise ValueError(f'Malformed track: {name}/{ident}')
        tracks.append((ident, index, xy))
    if len(tracks) < 2:
        return [t[0] for t in tracks], []
    values = np.full((len(tracks), count, 2), np.nan)
    visible = np.zeros(values.shape[:2], dtype=bool)
    for i, (_, index, xy) in enumerate(tracks):
        values[i, index] = xy
        visible[i, index] = True
    keys = np.asarray([t[0] for t in tracks])
    result = []
    grid = STRIDE * np.arange(20)
    for start, at in model.cohorts(visible, STRIDE):
        block = values[at[:, None], start + grid]
        past, future = block[:, :8], block[:, 8:] - block[:, 7:8]
        result.append((start + int(ticks[0]), keys[at], past, future))
    return keys.tolist(), result


def extract(row, registration):
    site = row['site']
    path = PRIVATE / f'{site}_cohort.npz'
    receipt = path.with_suffix('.json')
    if path.exists() and receipt.exists():
        got = json.loads(receipt.read_text())
        if got['registration'] != registration or got['sha256'] != sha(path):
            raise ValueError(f'Cohort hash mismatch: {site}')
        beat('cohort_cached_verified', site=site, rows=got['rows'])
    else:
        beat('extract', site=site)
        names, keys, histories, labels, starts, ids = [], [], [], [], [], []
        source = ARCHIVES / row['asset']['archive']
        with ZipFile(source) as z, TemporaryDirectory(dir=PRIVATE) as tmp:
            members = [n for n in z.namelist() if Path(n).name == f'trajectories_{site}_v3.sqlite']
            if len(members) != 1:
                raise ValueError(f'Expected exactly one v3 export for {site}')
            db = Path(tmp) / 'tracks.sqlite'
            with z.open(members[0]) as src, db.open('wb') as dest:
                shutil.copyfileobj(src, dest, 1024 * 1024)
            with closing(sqlite3.connect(f'{db.as_uri()}?mode=ro&immutable=1', uri=True)) as conn:
                conn.execute('PRAGMA trusted_schema=OFF')
                conn.execute('PRAGMA query_only=ON')
                tables = [t for (t,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                    if re.fullmatch(rf'{site}_\d+', t)]
                if len(tables) != row['recordings']:
                    raise ValueError(f'Recording roster changed: {site}')
                for name in tables:
                    tracks, windows = flight(conn, name)
                    ids.extend(tracks)
                    for start, a, past, target in windows:
                        keys.append(a)
                        histories.append(past)
                        labels.append(target)
                        starts.append(start)
                        names.append(name)
        if len(ids) != row['classes']['Pedestrian']['tracks'] or len(ids) != len(set(ids)):
            raise ValueError(f'Expected unique publisher pedestrian IDs in {site}')
        tmp = path.with_suffix('.pending')
        with tmp.open('wb') as f:
            np.savez(f, id=np.concatenate(keys), history=np.concatenate(histories), future=np.concatenate(labels),
                     all_ids=np.asarray(ids), recording=np.asarray(names), start=np.asarray(starts),
                     offsets=np.r_[0, np.cumsum([len(a) for a in keys])])
        tmp.replace(path)
        write_json(receipt, dict(sha256=sha(path), registration=registration, rows=sum(map(len, keys))), immutable=True)
        beat('cohort_extracted', site=site, queries=len(keys), rows=sum(map(len, keys)))
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def design(raw):
    h, y, offsets = raw['history'], raw['future'], raw['offsets']
    x, cv, ols, bank, scale = motion.features(h)
    peer = model.geometry(h, offsets, scale)
    peer = np.broadcast_to(peer[:, None], (*x.shape[:2], peer.shape[1]))
    valid = np.isfinite(y).all(2)
    labels = motion.targets(ols, bank, y, valid, scale)
    dist = np.linalg.norm(bank - ols[:, None], axis=-1)
    return dict(x=np.concatenate((x, peer), axis=-1), cv=cv, ols=ols, bank=bank,
        labels=labels, y=y, valid=valid, dist=dist, legal=(dist > 1e-9).any(2),
        hindcast=x[..., -4] - x[..., -2], offsets=offsets, flight=raw['recording'])


def inputs(d, arm):
    x = torch.from_numpy(d['x'].copy())
    if arm == 'self':
        x[..., -6:] = 0
    return x


def fit(sources, held, seed, arm, registration):
    names = sorted(s for s in sources if s != held)
    train = [sources[s] for s in names]
    x = [inputs(d, arm) for d in train]
    y = [torch.from_numpy(d['labels']) for d in train]
    pools = [np.flatnonzero(d['valid'].any(1)) for d in train]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1776)
    net = motion.network(x[0].shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    path = PRIVATE / f'{held}_{seed}_{arm}.pt'
    ident = dict(registration=registration, held=held, train=names, seed=seed, arm=arm)
    step, trace = 0, []
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != ident:
            raise ValueError('Checkpoint binding mismatch')
        step, trace = saved['step'], saved['trace']
        net.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['rng']
    for i in range(step, STEPS):
        site_draw = rng.integers(len(names), size=BATCH)
        selections = [(k, rng.choice(p, size=int((site_draw == k).sum()))) for k, p in enumerate(pools)]
        xb = torch.cat([x[k][at] for k, at in selections])
        yb = torch.cat([y[k][at] for k, at in selections])
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(motion.prediction(net, xb), yb, beta=.1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        opt.step()
        if (i + 1) % 100 == 0:
            trace.append(float(loss))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident, step=i + 1, trace=trace, model=net.state_dict(),
                            optimizer=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat('checkpoint', held=held, seed=seed, arm=arm, step=i + 1, loss=trace[-1])
    return net.eval(), dict(**ident, losses=trace)


def edited(d, score):
    k = model.choose(score, d['legal'], d['offsets'])
    at = np.flatnonzero(k >= 0)
    pred = d['ols'].copy()
    pred[at] = d['bank'][at, k[at]]
    return pred, len(at)


def error(pred, d):
    return np.linalg.norm(pred - d['y'], axis=-1)


def comparisons(d, a, b):
    gain, lower = model.paired(a, b, d['y'], d['valid'])
    den = np.nansum(error(d['ols'], d))
    return dict(observed_gain_pp=float(100 * gain.sum() / den), lower_pp=float(100 * lower.sum() / den))


def evaluate(d, training, nets):
    y, valid = d['y'], d['valid']
    counts = valid.sum(1)
    full = counts == 12
    train_err = np.concatenate([error(t['cv'], t)[t['valid'].all(1)].mean(1) for t in training])
    easy_cut = float(np.quantile(train_err[train_err > 0], .25))
    e_cv = error(d['cv'], d)
    row_cv = np.divide(np.nansum(e_cv, axis=1), counts, out=np.zeros(len(y)), where=counts > 0)
    easy = full & (row_cv > 0) & (row_cv <= easy_cut)
    den = row_cv[easy].sum()
    bank_error = np.linalg.norm(d['bank'] - y[:, None], axis=-1)
    oracle_gain = np.where(valid[:, None], error(d['ols'], d)[:, None] - bank_error, 0.).sum(2)
    oracle, n = edited(d, oracle_gain)
    arms = dict(cv=(d['cv'], 0), ols4=(d['ols'], 0), oracle_observed=(oracle, n),
                minimum=edited(d, -d['dist'].mean(2)), hindcast=edited(d, d['hindcast']))
    for arm in ARMS:
        with torch.no_grad():
            value = motion.prediction(nets[arm], inputs(d, arm)).numpy()
        arms[arm] = edited(d, value[..., 0] - value[..., 1])
    if len({arms[a][1] for a in arms if a not in ('cv', 'ols4')}) != 1:
        raise ValueError('Matched decision counts differ')
    result = dict(rows=len(y), queries=len(d['offsets']) - 1, flights=len(set(d['flight'])),
        complete=int(full.sum()), no_labels=int((counts == 0).sum()),
        partial=int(((counts > 0) & ~full).sum()), easy=int(easy.sum()), train_easy_cut=easy_cut,
        arms={})
    for arm, (pred, n) in arms.items():
        err = error(pred, d)
        result['arms'][arm] = dict(switches=n, ade=float(np.nanmean(err)), fde=float(np.nanmean(err[:, -1])),
            complete_ade=float(err[full].mean()),
            easy_cv_change_pct=float(100 * (err[easy].mean(1).sum() - den) / den) if den else None,
            **comparisons(d, d['ols'], pred))
    result['neighbor_vs_self'] = comparisons(d, arms['self'][0], arms['neighbor'][0])
    result['self_vs_hindcast'] = comparisons(d, arms['hindcast'][0], arms['self'][0])
    result['neighbor_vs_hindcast'] = comparisons(d, arms['hindcast'][0], arms['neighbor'][0])
    return result


def summarize(sites):
    def stat(values):
        v = np.asarray(values)
        index = np.random.default_rng(817).integers(len(v), size=(3000, len(v)))
        return dict(equal_site_mean_pp=float(v.mean()), site_bootstrap_ci95_pp=np.quantile(v[index].mean(1), [.025, .975]).tolist(),
                    positive_sites=int((v > 0).sum()), worst_site_pp=float(v.min()))

    arms = ('cv', 'self', 'neighbor', 'oracle_observed', 'hindcast', 'minimum')
    out = {}
    for arm in arms:
        rows = [[s['arms'][arm] for s in results] for results in sites.values()]
        out[f'{arm}_vs_ols4'] = {key: stat([np.mean([r[key] for r in site]) for site in rows])
                                for key in ('lower_pp', 'observed_gain_pp')}
    for arm in ('neighbor_vs_self', 'self_vs_hindcast', 'neighbor_vs_hindcast'):
        out[arm] = {key: stat([np.mean([s[arm][key] for s in results]) for results in sites.values()])
                    for key in ('lower_pp', 'observed_gain_pp')}
    return out


def run():
    start = monotonic()
    records, registration = register()
    raw = {s['site']: extract(s, registration) for s in records}
    all_ids = np.concatenate([r['all_ids'] for r in raw.values()])
    if len(all_ids) != len(np.unique(all_ids)):
        raise ValueError('Publisher object IDs overlap across held sites')
    sources = {k: design(r) for k, r in raw.items()}
    fits, weights = [], {}
    for held in sources:
        for seed in SEEDS:
            weights[held, seed] = {}
            for arm in ARMS:
                net, fit_meta = fit(sources, held, seed, arm, registration)
                weights[held, seed][arm] = net
                fits.append(fit_meta)
    scores = {held: [evaluate(d, [v for k, v in sources.items() if k != held], weights[held, seed]) for seed in SEEDS]
              for held, d in sources.items()}
    output = PUBLIC / 'analysis.json'
    replay = output.exists()
    write_json(output, dict(registration=registration, scope='exploratory', seeds=SEEDS,
        runtime=dict(torch=str(torch.__version__), arch=platform.machine(), torch_threads=4, workers=0),
        disjoint_pedestrian_ids=int(len(all_ids)), fits=fits, sites=scores,
        summary=summarize(scores), bootstrap_sites=7, bootstrap_draws=3000,
        no_confirmation_access=True, official_safety_pass=False), immutable=True)
    beat('complete', analysis_sha256=sha(output), exact_replay=replay, elapsed_s=round(monotonic() - start, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Run native arm64 .venv-pytorch/bin/python, not Rosetta')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.register_only:
            _, digest = register()
            print(digest)
        else:
            import numpy as np
            import torch
            from src.data_unification.m3w_topdown_source_support import grid_indices
            from src.evaluation import m3w_opendd_location as model
            from src.world_model import m3w_ols_replacement as motion

            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            run()
