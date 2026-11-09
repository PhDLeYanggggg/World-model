"""Prepare unscored vehicle forecasts for the complete first-day Songdo input pack."""

import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_first_day as first
from scripts import build_m3w_songdo_candidate_bank as bank
from scripts import run_m3w_opendd_vehicle_residual as donor

np, torch, pack = bank.np, bank.torch, first.pack
CACHE = first.PRIVATE / 'vehicle_bank'
MODEL = donor.model


def weights():
    key, _ = donor.register()
    path = donor.OUT / 'freeze.json'
    f = first.read(path)
    if f['registration'] != key or f['songdo_read']:
        raise ValueError('Require the frozen OpenDD vehicle-only training')
    nets, checksums = {}, {}
    for seed in bank.SEEDS:
        nets[seed] = []
        for held in donor.SITES:
            k = f'{held}_{seed}_completion'
            ref = f['checkpoints'][k]
            p = donor.LOCAL / f'{k}.pt'
            h = pack.sha(p)
            if h != ref['sha256'] or ref['step'] != donor.STEPS:
                raise ValueError(f'Checkpoint changed: {k}')
            state = torch.load(p, weights_only=True, map_location='cpu')
            ident = state['identity']
            if (ident != ref['identity'] or state['step'] != donor.STEPS or ident['arm'] != 'completion'
                    or ident['registration'] != key or ident['held'] != held or ident['seed'] != seed
                    or set(ident['donors']) != set(donor.SITES)-{held}):
                raise ValueError(f'Unexpected training sources: {k}')
            net = MODEL.motion.network(17)
            net.load_state_dict(state['weights'])
            nets[seed].append(net.eval())
            checksums[k] = h
    return nets, checksums, pack.sha(path)


@torch.inference_mode()
def forecasts(h, nets):
    cv, ols = bank.analytic(h)
    result = dict(cv=cv, ols4=ols, median4=bank.linear_menu(h, 12)[:, 2])
    if len(h):
        x, base, rotation, scale = MODEL.encode(h)
        x, base = torch.from_numpy(x), torch.from_numpy(base)
        for seed, models in nets.items():
            blocks = []
            for a, b in zip(x.split(4096), base.split(4096), strict=True):
                blocks.append(np.mean([MODEL.predict(m, a, b).numpy().astype(float) for m in models], axis=0))
            local = np.concatenate(blocks)
            result[f'self_{seed}'] = (local*scale[:, None, None]) @ rotation.transpose(0, 2, 1)
    else:
        result.update({f'self_{seed}': cv.copy() for seed in nets})
    if any(p.shape != (len(h), 12, 2) or not np.isfinite(p).all() for p in result.values()):
        raise ValueError('Expected finite past-only displacement predictions')
    return result


def run():
    p = first.PUBLIC / 'input_receipt.json'
    receipt = first.read(p)
    link = first.read(first.PUBLIC / 'lineage.json')
    if (receipt['split_assigned'] or receipt['forecasts_scored'] or receipt['future_coordinates_parsed']
            or link['input_receipt_sha256'] != pack.sha(p) or len(receipt['sessions']) != 200):
        raise ValueError('Require all split-neutral, linked first-day prefix packets')
    nets, checkpoints, frozen = weights()
    bound = dict(prefix=pack.sha(p), lineage=pack.sha(first.PUBLIC / 'lineage.json'), forecasts=frozen,
                 implementation=pack.sha(Path(__file__)), slot_counter=pack.sha(Path(bank.__file__)),
                 donor_weights=checkpoints)
    identity = pack.hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
    output, stats = {}, {s: [] for s in bank.SEEDS}
    for name, item in receipt['sessions'].items():
        stem = Path(name).stem
        prefix = (pack.CACHE if item['cache_role'] == 'previous' else first.PRIVATE) / f'{stem}_prefix.npz'
        packet = CACHE / f'{stem}.npz'
        r = CACHE / f'{stem}.json'
        if pack.sha(prefix) != item['prefix_sha']:
            raise ValueError('Observed prefix changed')
        if not r.exists():
            with np.load(prefix, allow_pickle=False) as data:
                h, offsets = data['history'], data['offsets']
            if (len(h), len(offsets)-1) != (item['agents'], item['queries']) or offsets[-1] != len(h):
                raise ValueError('Observed roster changed')
            f = forecasts(h, nets)
            counts = {f'stats_{seed}': bank.count_queries(f, offsets, seed) for seed in bank.SEEDS}
            sha = pack.save_npz(packet, dict(**f, **counts, offsets=offsets))
            pack.write_json(r, dict(identity=identity, sha256=sha, prefix_sha=item['prefix_sha']))
        row = first.read(r)
        if row != dict(identity=identity, sha256=pack.sha(packet), prefix_sha=item['prefix_sha']):
            raise ValueError('Cached forecast does not bind these observed inputs and weights')
        with np.load(packet, allow_pickle=False) as z:
            for seed in bank.SEEDS:
                stats[seed].append(z[f'stats_{seed}'])
        output[name] = dict(site=item['site'], queries=item['queries'], agents=item['agents'], **row)
        first.heartbeat('vehicle_bank', completed=len(output), session=name, rows=item['agents'])
    pack.write_json(first.PUBLIC / 'vehicle_bank.json', dict(
        binding=bound, result_source='fresh_run', inputs='cached_verified',
        candidates=['cv', 'ols4', 'median4', 'opendd_vehicle_completion'], seeds=list(bank.SEEDS),
        sessions=output, rows=sum(v['agents'] for v in output.values()), queries=sum(v['queries'] for v in output.values()),
        per_seed_slots={str(s): bank.aggregate(np.concatenate(v)) for s, v in stats.items()},
        future_index_loaded=False, future_coordinates_loaded=False, forecasts_scored=False,
        split_assigned=False, policy_selected=False, independent_confirmation=False))


if __name__ == '__main__':
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (CACHE / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
