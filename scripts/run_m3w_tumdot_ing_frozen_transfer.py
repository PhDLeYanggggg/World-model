"""Score fixed OpenDD forecasts on the public TUMDOT-ING diagnostic sample."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_tumdot_ing_support import FILES, frame_mapping
from scripts import run_m3w_opendd_location as prior
from src.evaluation.m3w_opendd_location import cohorts

PUBLIC = prior.PUBLIC.parent / 'tumdot_ing_frozen_transfer_v1'
PRIVATE = prior.PRIVATE.parent / PUBLIC.name
SOURCE = ROOT / 'external_data/TUMDOT_ING_source_audit'
VELOCITY = prior.PUBLIC.parent / 'opendd_velocity_candidate_v1'
CALIBRATION = prior.PUBLIC.parent / 'opendd_velocity_calibration_v1'
SITES = tuple(f'rdb{i}' for i in range(1, 8))
ROSTER = {
    'P1-d1': FILES['P1'],
    'P1-d2': ('Intersection_P1_-_d2_-__16_01_39_16_08_52__-_A.csv', 43400570),
    'P1-d3': ('Intersection_P1_-_d3_-__15_59_15_16_05_13__-_A.csv', 31256735),
    'P2-d1': FILES['P2'],
    'P2-d2': ('Intersection_2-d2-_16_01_42_16_10_0.csv', 70210185),
    'P2-d3': ('Intersection_2-d3-_15_59_33_16_07_5.csv', 77221427),
    'P3-d1': FILES['P3'],
    'P3-d2': ('Intersection_3-d2-_16_01_39_16_12_4.csv', 59852172),
    'P3-d3': ('Intersection_3-d3-_15_59_35_16_09_4.csv', 61114285),
}
POSITIONS = ('x_local', 'y_local', 'x_glob_UTM32N', 'y_glob_UTM32N')


def register():
    paths = [
        Path(__file__).resolve(), PUBLIC / 'protocol.md', Path(prior.__file__),
        ROOT / 'scripts/audit_m3w_tumdot_ing_support.py',
        VELOCITY / 'registration.json', CALIBRATION / 'registration.json',
    ]
    for name in ('evaluation/m3w_opendd_location', 'evaluation/m3w_native_metrics',
                 'evaluation/m3w_selection_forecast_readout', 'world_model/m3w_ols_replacement',
                 'world_model/m3w_opendd_velocity_candidate'):
        paths.append(ROOT / f'src/{name}.py')
    files = {}
    day_one = json.loads((PUBLIC.parent / 'station_sharedspace_intake_v1/tumdot_ing.json').read_text())
    for key, (name, size) in ROSTER.items():
        p = SOURCE / name
        if p.stat().st_size != size:
            raise ValueError(f'Publisher byte count mismatch: {name}')
        digest = prior.sha(p)
        if key.endswith('d1') and digest != day_one['sites'][key[:2]]['asset']['sha256']:
            raise ValueError(f'Previously verified file changed: {name}')
        files[key] = dict(name=name, bytes=size, sha256=digest)
    donor_registration = prior.sha(prior.PUBLIC / 'registration.json')
    donor_cohorts = {}
    for site in SITES:
        path = prior.PRIVATE / f'{site}_cohort.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['registration'] != donor_registration or meta['sha256'] != prior.sha(path):
            raise ValueError(f'Unexpected OpenDD cohort identity: {site}')
        donor_cohorts[site] = meta['sha256']
    freeze = dict(kind='prospective_cross_source_diagnostic', files=files,
                  code={str(p.relative_to(ROOT)): prior.sha(p) for p in paths},
                  open_dd_donor_cohorts=donor_cohorts, seeds=list(prior.SEEDS),
                  primary_alpha=.05, diagnostic_alpha=.25, stride=12, steps=[8, 12],
                  risk_budget=.02, dronecrowd_confirmation_access=False)
    p = PUBLIC / 'registration.json'
    prior.write_json(p, freeze, immutable=True)
    return prior.sha(p)


def read_tracks(path):
    times, kinds, tracks = set(), {}, defaultdict(list)
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = csv.reader(handle)
        head = next(rows)
        if head[:3] != ['timestep', 'track_id', 'category'] or len(set(head)) != len(head):
            raise ValueError('Unexpected TUMDOT-ING schema')
        at = [head.index(k) for k in POSITIONS]
        for row in rows:
            ts, ident, kind = row[:3]
            t = float(ts)
            if not np.isfinite(t) or not ident or not kind or kinds.setdefault(ident, kind) != kind:
                raise ValueError('Malformed time, track identity or class')
            times.add(t)
            if kind.lower() in ('pedestrian', 'pedestrians'):
                x, y, gx, gy = [float(row[i]) for i in at]
                if not np.isfinite([x, y, gx, gy]).all():
                    raise ValueError('Malformed pedestrian position')
                offset = [gx-x, gy-y]
                lo, hi = np.minimum(lo, offset), np.maximum(hi, offset)
                tracks[ident].append((t, x, y))
    ticks, error, gaps = frame_mapping(times)
    if len(tracks) < 2 or (hi-lo > .01).any():
        raise ValueError('No supported pedestrian cohort or local axes not translated UTM')
    data = np.full((len(tracks), max(ticks.values())+1, 2), np.nan)
    for i, ident in enumerate(sorted(tracks)):
        tr = tracks[ident]
        ix = [ticks[t] for t, _, _ in tr]
        if len(set(ix)) != len(ix):
            raise ValueError('Duplicate pedestrian time within track')
        data[i, ix] = [(x, y) for _, x, y in tr]
    return data, dict(pedestrian_tracks=len(tracks), grid_residual_seconds=error,
                      grid_gaps=gaps, local_translation_range=(hi-lo).tolist()), set(kinds)


def windows(data):
    vis = np.isfinite(data).all(2)
    past, future, previous, starts = [], [], [], []
    for start, at in cohorts(vis, 12):
        past.append(data[at[:, None], start+12*np.arange(8)])
        future.append(data[at[:, None], start+12*np.arange(8, 20)])
        previous.append(data[at, start+84])
        starts.append(start)
    if not starts:
        raise ValueError('No history-eligible query in fixed roster')
    history = np.concatenate(past)
    label = np.concatenate(future) - np.concatenate(previous)[:, None]
    return history, label, starts


def donor_cuts(motion):
    donor = []
    for site in SITES:
        with np.load(prior.PRIVATE / f'{site}_cohort.npz', allow_pickle=False) as f:
            h, y = f['history'], f['future']
        complete = np.isfinite(y).all(axis=(1, 2))
        cv, _, _, _ = motion.rollout(h[complete], 12)
        ade = np.linalg.norm(cv-y[complete], axis=-1).mean(1)
        donor.append(ade[ade > 0])
    return np.quantile(np.concatenate(donor), [.25, .75])


def load_networks(torch, model):
    binding = json.loads((CALIBRATION / 'registration.json').read_text())['checkpoint_sha256']
    reg = prior.sha(VELOCITY / 'registration.json')
    result = {}
    for seed in prior.SEEDS:
        result[seed] = []
        for held in SITES:
            path = prior.PRIVATE.parent / VELOCITY.name / f'{held}_{seed}_self.pt'
            if prior.sha(path) != binding[f'{held}_{seed}']:
                raise ValueError('Frozen OpenDD checkpoint changed')
            ck = torch.load(path, weights_only=True, map_location='cpu')
            expected = dict(registration=reg, held=held, train=[s for s in SITES if s != held],
                            seed=seed, arm='self')
            if ck['identity'] != expected or ck['step'] != 600:
                raise ValueError('Unexpected OpenDD checkpoint provenance')
            net = model.network(113)
            net.load_state_dict(ck['model'])
            result[seed].append(net.eval())
    return result


def motion_bank(h, motion):
    x, cv, ols, bank, _ = motion.features(h)
    ols_probe = []
    for prefix in (4, 5, 6):
        c, o, _, s = motion.rollout(h[:, :prefix], 8-prefix)
        target = h[:, prefix:]-h[:, prefix-1:prefix]
        ols_probe.append((np.linalg.norm(c-target, axis=-1).mean(1)
                          - np.linalg.norm(o-target, axis=-1).mean(1)) / s)
    bank = np.concatenate((ols[:, None], bank), axis=1)
    merit = np.column_stack((np.mean(ols_probe, axis=0).clip(-16., 16.).astype(np.float32), x[:, :, 17]))
    legal = (np.linalg.norm(bank-cv[:, None], axis=-1)>1e-9).any(2)
    at = np.where(legal, merit, -np.inf).argmax(1)
    return cv, ols, bank[np.arange(len(h)), at], legal.any(1)


def blend(cv, forecast, matched, eligible, alpha):
    common = eligible & (np.linalg.norm(forecast-cv, axis=-1).max(1)>1e-9)
    m = alpha * common[:, None, None]
    result, old = cv + m*(forecast-cv), cv + m*(matched-cv)
    for v in (result, old):
        if not np.array_equal((v != cv).any(axis=(1, 2)), common):
            raise ValueError('Matched intervention counts are not exact')
    return result, old


def score(p, cv, ols, y, valid, key, cuts, matched=None):
    from src.evaluation.m3w_opendd_location import paired
    from src.evaluation.m3w_selection_forecast_readout import policy

    err = np.linalg.norm(p-y, axis=-1)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    if not valid[:, -1].any() or mass <= 0:
        raise ValueError('Missing fixed CV scoring support')
    risk = policy(p, cv, cv, y, valid, np.full(len(y), key),
                  easy_cut=cuts[0], hard_cut=cuts[1])['recordings'][key]
    refs = dict(cv=cv, ols4=ols)
    if matched is not None:
        refs['matched_hindcast'] = matched
    gain = {name: float(100*paired(v, p, y, valid)[1].sum()/mass) for name, v in refs.items()}
    return dict(lower_pp=gain, risk=risk, available_ade=float(err[valid].mean()),
                available_fde=float(err[valid[:, -1], -1].mean()))


def mean_summary(records):
    out = {}
    for arm in records[0]['arms']:
        values = [row['arms'][arm] for row in records]
        out[arm] = dict(
            lower_pp={ref: float(np.mean([v['lower_pp'][ref] for v in values]))
                      for ref in values[0]['lower_pp']},
            risk_passes=sum(v['risk']['supported'] for v in values),
            undefined_selected_easy=sum(v['risk']['completion_easy_risk_upper'] is None for v in values),
            worst_defined_net_easy_change_pct=max(
                (100*v['risk']['known_easy_net_degradation'] for v in values
                 if v['risk']['known_easy_net_degradation'] is not None), default=None),
            available_ade=float(np.mean([v['available_ade'] for v in values])),
            available_fde=float(np.mean([v['available_fde'] for v in values])),
        )
    return out


def evaluate(reg, replay):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_ols_replacement as motion
    from src.world_model import m3w_opendd_velocity_candidate as model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0 = monotonic()
    cuts = donor_cuts(motion)
    networks = load_networks(torch, model)
    sites, ids = {}, Counter()
    for key, (name, _) in ROSTER.items():
        state = dict(recording=key, pid=os.getpid(), replay=replay)
        prior.write_json(PRIVATE / 'heartbeat.json', state)
        print(json.dumps(state), flush=True)
        p = PRIVATE / f'{key}.json'
        id_path = PRIVATE / f'{key}.ids.json'
        if p.exists() and not replay:
            checkpoint = json.loads(p.read_text())
            if checkpoint['registration'] != reg or checkpoint['id_sha256'] != prior.sha(id_path):
                raise ValueError('TUMDOT per-file checkpoint changed')
            sites[key] = checkpoint['evidence']
            ids.update(json.loads(id_path.read_text()))
            continue
        data, audit, keys = read_tracks(SOURCE / name)
        ids.update(keys)
        h, y, starts = windows(data)
        del data
        valid = np.isfinite(y).all(2)
        cv, ols, old, eligible = motion_bank(h, motion)
        x, rot, scale = model.design(h, np.full((len(h), 2, 2, 8, 2), np.nan))
        samples = torch.from_numpy(x)
        controls = {k: score(v, cv, ols, y, valid, key, cuts) for k, v in dict(cv=cv, ols4=ols).items()}
        views = []
        for seed, nets in networks.items():
            with torch.inference_mode():
                local = np.mean([model.decode(net(samples)).numpy().astype(float) for net in nets], axis=0)
            forecast = (local*scale[:, None, None]) @ rot.transpose(0, 2, 1)
            arms = {**controls, 'direct': score(forecast, cv, ols, y, valid, key, cuts)}
            for arm, alpha in (('blend05', .05), ('blend25', .25)):
                candidate, matched = blend(cv, forecast, old, eligible, alpha)
                arms[arm] = score(candidate, cv, ols, y, valid, key, cuts, matched)
            views.append(dict(seed=seed, arms=arms))
        sites[key] = dict(**audit, queries=len(starts), targets=len(h), complete=int(valid.all(1).sum()),
                          unknown=int((~valid.any(1)).sum()), endpoint=int(valid[:, -1].sum()), views=views)
        prior.write_json(id_path, sorted(keys), immutable=True)
        prior.write_json(p, dict(registration=reg, id_sha256=prior.sha(id_path), evidence=sites[key]), immutable=True)
    views = [v for evidence in sites.values() for v in evidence['views']]
    by_day = {f'd{i}': mean_summary([v for k, row in sites.items() if k.endswith(f'd{i}') for v in row['views']])
              for i in (1, 2, 3)}
    out = dict(registration=reg, provenance='fresh_run', recorded_files=sites,
               mean=mean_summary(views), by_day=by_day, views=len(views), campaigns=1,
               open_dd_positive_complete_cv_cuts=cuts.tolist(),
               cross_file_repeated_track_ids=sum(v>1 for v in ids.values()),
               fitted_models=0, cached_verified_checkpoints=sum(map(len, networks.values())),
               dronecrowd_confirmation_access=False, formal_confirmation=False, deployment=False)
    result = PUBLIC / 'analysis.json'
    if replay and not result.exists():
        raise ValueError('Replay requires a completed reference')
    prior.write_json(result, out, immutable=True)
    receipt = PUBLIC / ('replay_execution.json' if replay else 'execution.json')
    if not receipt.exists():
        prior.write_json(receipt, dict(registration=reg, analysis_sha256=prior.sha(result),
                         fresh_exact_replay=replay, wall_seconds=round(monotonic()-t0, 3),
                         pid=os.getpid(), torch_threads=4, dataloader_workers=0, device='arm64_cpu'))
    print(f'{"Exact replay matched" if replay else "Readout saved"}: {prior.sha(result)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'replay'))
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg = register()
        if args.phase == 'register':
            print(reg)
        else:
            evaluate(reg, replay=args.phase == 'replay')


if __name__ == '__main__':
    main()
