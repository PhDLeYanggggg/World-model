"""Generate unranked Songdo forecasts and count legal equal-radius assignments."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_prefix_pack as pack
from scripts import run_m3w_tumdot_ing_frozen_transfer as frozen
from src.evaluation.m3w_camera_reference_capacity import linear_menu
from src.world_model import m3w_opendd_velocity_candidate as velocity
from src.world_model.m3w_wepdtof_matched_policy import FRACTION, analytic
import torch

np = pack.np
PUBLIC = pack.PUBLIC.parent / 'songdo_candidate_bank_v1'
CACHE = pack.CACHE.parent / PUBLIC.name
SEEDS = (17, 29, 43)
FAMILIES = ('ols4', 'median4', 'opendd_pedestrian_self')
PEAK_EPS = 1e-8


def slot_counts(peak):
    """Count ordered, distinct-recipient, equal-radius slots without enumerating slates."""
    if peak.ndim != 2 or not np.isfinite(peak).all() or (peak < 0.).any():
        raise ValueError('Require finite nonnegative peak matrix')
    common = peak.min(axis=1)
    positive = common[common > PEAK_EPS]
    if len(peak) < 3 or len(positive) < 2:
        return (0, 0, 0, 0, 0.)
    radius = float(FRACTION * np.median(positive))
    legal = (peak > PEAK_EPS) & (peak >= radius)
    by_actor, by_family = legal.sum(axis=1), legal.sum(axis=0)
    singles = int(by_actor.sum())
    joint = singles**2 - int(by_actor @ by_actor)
    uniform = int(by_family @ (by_family - 1))
    return singles, joint, uniform, int((by_actor == peak.shape[1]).sum()), radius


@torch.inference_mode()
def forecasts(h, networks):
    cv, ols = analytic(h)
    median = linear_menu(h, 12)[:, 2]
    result = dict(cv=cv, ols4=ols, median4=median)
    if len(h):
        x, rot, scale = velocity.design(h, np.full((len(h), 2, 2, 8, 2), np.nan))
        samples = torch.from_numpy(x)
        for seed, nets in networks.items():
            local = np.mean([velocity.decode(net(samples)).numpy().astype(float) for net in nets], axis=0)
            result[f'self_{seed}'] = (local * scale[:, None, None]) @ rot.transpose(0, 2, 1)
    else:
        result.update({f'self_{seed}': cv.copy() for seed in networks})
    if any(a.shape != (len(h), 12, 2) or not np.isfinite(a).all() for a in result.values()):
        raise ValueError('Unexpected nonfinite forecast or shape')
    return result


def count_queries(paths, offsets, seed):
    cv = paths['cv']
    raw = np.stack([paths[k] for k in ('ols4', 'median4', f'self_{seed}')], axis=1)
    peak = np.linalg.norm(raw - cv[:, None], axis=-1).max(axis=-1)
    stats = [slot_counts(peak[lo:hi]) for lo, hi in zip(offsets[:-1], offsets[1:], strict=True)]
    return np.asarray(stats, dtype=np.float64).reshape(-1, 5)


def aggregate(m):
    singles, pair, uniform, all_families, radius = m.T
    return dict(queries=len(m), joint_feasible=int((pair > 0).sum()), uniform_feasible=int((uniform > 0).sum()),
                all_families_two_recipients=int((all_families >= 2).sum()),
                one_slot_candidates=int(singles.sum()), ordered_two_slot_candidates=int(pair.sum()),
                ordered_same_family_two_slot_candidates=int(uniform.sum()),
                max_ordered_pairs_in_query=int(pair.max(initial=0)),
                positive_radius_median_native=float(np.median(radius[radius > 0.])) if (radius > 0.).any() else None)


def setup(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    if receipt['split_assigned'] or receipt['future_coordinate_fields_parsed'] or receipt['forecasts_scored']:
        raise ValueError('Require split-neutral and unscored prefix inputs')
    producer = frozen.VELOCITY / 'registration.json'
    calibration = frozen.CALIBRATION / 'registration.json'
    metadata = json.loads(calibration.read_text())
    weights = metadata['checkpoint_sha256']
    if set(weights) != {f'{s}_{seed}' for s in frozen.SITES for seed in SEEDS}:
        raise ValueError('Require all three donor ensembles')
    if metadata['dronecrowd_confirmation_access'] or metadata['official_risk_budget'] != .02:
        raise ValueError('Unapproved donor checkpoint provenance')
    nets = frozen.load_networks(torch, velocity)
    if set(nets) != set(SEEDS) or any(len(n) != 7 for n in nets.values()):
        raise ValueError('Incomplete checkpoint ensemble')
    binding = dict(prefix_receipt=pack.sha(receipt_path), implementation=pack.sha(Path(__file__)),
                   producer_registration=pack.sha(producer), calibration_registration=pack.sha(calibration),
                   loader=pack.sha(Path(frozen.__file__)), neural_source=pack.sha(Path(velocity.__file__)),
                   donor_weights=weights, seeds=list(SEEDS), radius_fraction=FRACTION)
    return receipt, nets, binding


def run(receipt_path, input_cache, cache, output):
    receipt, nets, binding = setup(receipt_path)
    key = pack.hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    sessions, all_stats, max_agents = {}, {s: [] for s in SEEDS}, 0
    for name, input_info in receipt['sessions'].items():
        stem = Path(name).stem
        p = input_cache / f'{stem}_prefix.npz'
        if pack.sha(p) != input_info['prefix_sha']:
            raise ValueError(f'Input changed: {name}')
        dest = cache / f'{stem}_forecasts.npz'
        log = cache / f'{stem}.json'
        if not log.exists():
            with np.load(p, allow_pickle=False) as data:
                h, offsets = data['history'], data['offsets']
            if offsets[-1] != len(h) or len(h) != input_info['agents'] or len(offsets)-1 != input_info['queries']:
                raise ValueError('Input roster drift')
            paths = forecasts(h, nets)
            stats = {f'stats_{seed}': count_queries(paths, offsets, seed) for seed in SEEDS}
            value = dict(key=key, prefix_sha=input_info['prefix_sha'],
                         forecast_sha=pack.save_npz(dest, dict(**paths, **stats, offsets=offsets)))
            pack.write_json(log, value)
        saved = json.loads(log.read_text())
        if saved != dict(key=key, prefix_sha=input_info['prefix_sha'], forecast_sha=pack.sha(dest)):
            raise ValueError(f'Saved forecast bank changed: {name}')
        with np.load(dest, allow_pickle=False) as f:
            max_agents = max(max_agents, int(np.diff(f['offsets']).max(initial=0)))
            for seed in SEEDS:
                all_stats[seed].append(f[f'stats_{seed}'])
        sessions[name] = dict(site=input_info['site'], queries=input_info['queries'], agents=input_info['agents'], **saved)
        tmp = cache / 'heartbeat.pending'
        tmp.write_text(json.dumps(dict(pid=os.getpid(), sessions=len(sessions), last=name)) + '\n')
        tmp.replace(cache / 'heartbeat.json')
        print(name, input_info['agents'], flush=True)
    if sum(r['queries'] for r in sessions.values()) != receipt['total_queries']:
        raise ValueError('Bank did not preserve all audited windows')
    info = {str(seed): aggregate(np.concatenate(s)) for seed, s in all_stats.items()}
    report = dict(result_source='fresh_run', inputs='cached_verified', binding=binding, candidate_families=FAMILIES,
                  donor_agent_type='pedestrian', receiver_agent_type='vehicle', sessions=sessions,
                  queries=receipt['total_queries'], agents=sum(v['agents'] for v in sessions.values()),
                  max_agents_in_query=max_agents, structural_slot_counts=info,
                  scope='unranked_candidates_and_slot_feasibility_only',
                  arm_selected=False, split_assigned=False, policy_fitted=False, forecasts_scored=False,
                  future_index_loaded=False, future_xy_loaded=False, independent_confirmation=False,
                  original_2pct_unchanged=True, reserved_outcomes_opened=False)
    pack.write_json(output, report)
    print(json.dumps(dict(queries=report['queries'], agents=report['agents'], max_agents=max_agents,
                          structural_slot_counts=info), indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt', type=Path, default=pack.PUBLIC / 'receipt.json')
    ap.add_argument('--input-cache', type=Path, default=pack.CACHE)
    ap.add_argument('--cache', type=Path, default=CACHE)
    ap.add_argument('--output', type=Path, default=PUBLIC / 'receipt.json')
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    a.cache.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with (a.cache / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a.receipt, a.input_cache, a.cache, a.output)
