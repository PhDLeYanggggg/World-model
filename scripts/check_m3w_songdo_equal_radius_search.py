"""Compare exact equal-radius implementations without reading any scoring labels."""

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_candidate_bank as bank
from scripts import run_m3w_pair_effect as donor
from src.world_model import m3w_equal_radius_slates as search
from src.world_model import m3w_slate_shared_menu as old

np, torch, pack = bank.np, bank.torch, bank.pack


def run(candidate_receipt, candidate_cache, input_cache, out):
    manifest = json.loads(candidate_receipt.read_text())
    if manifest['forecasts_scored'] or manifest['future_index_loaded'] or manifest['split_assigned']:
        raise ValueError('Expected an unscored candidate bank')
    largest = 0
    selected = None
    for name, row in manifest['sessions'].items():
        stem = Path(name).stem
        p = input_cache / f'{stem}_prefix.npz'
        if pack.sha(p) != row['prefix_sha']:
            raise ValueError('Observed prefix changed')
        with np.load(p, allow_pickle=False) as f:
            n = np.diff(f['offsets'])
            if n.max(initial=0) > largest:
                largest = int(n.max())
                q = int(n.argmax())
                lo, hi = f['offsets'][q:q+2]
                selected = (name, row, stem, q, lo, hi, f['history'][lo:hi])
    if largest != manifest['max_agents_in_query'] or selected is None:
        raise ValueError('Densest-query receipt did not replay')
    name, row, stem, q, lo, hi, h = selected
    bank_path = candidate_cache / f'{stem}_forecasts.npz'
    if pack.sha(bank_path) != row['forecast_sha']:
        raise ValueError('Candidate forecasts changed')
    freeze_path = donor.exp.OUT / 'freeze.json'
    freeze = donor.study.read(freeze_path)
    if freeze['registration'] != pack.sha(donor.exp.OUT / 'registration.json'):
        raise ValueError('Unverified ranking checkpoint freeze')
    rows = {}
    with np.load(bank_path, allow_pickle=False) as f:
        cv, ols = f['cv'][lo:hi], f['ols4'][lo:hi]
        for seed in bank.SEEDS:
            candidates = np.stack((ols, f['median4'][lo:hi], f[f'self_{seed}'][lo:hi]), axis=1)
            radius = float(f[f'stats_{seed}'][q, -1])
            net = donor.slate_for(freeze, 'rdb1', seed)
            start = perf_counter()
            a, j, n = old.choose(net, h, cv, ols, candidates, [radius, radius])
            original_seconds = perf_counter() - start
            start = perf_counter()
            values, unique = search.choose(net, h, cv, ols, candidates, radius)
            faster_seconds = perf_counter() - start
            np.testing.assert_allclose(a, values['additive']['path'], atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(j, values['joint']['path'], atol=1e-9, rtol=1e-9)
            if n != 2 * unique or n != f[f'stats_{seed}'][q, 1] or values['uniform'] is None:
                raise ValueError('Incorrect symmetric pair count or missing control')
            rows[str(seed)] = dict(ordered_candidates=n, unique_candidates=unique, additive_agrees=True,
                                   joint_agrees=True, same_family_control_supported=True)
            print(seed, dict(original_seconds=original_seconds, symmetric_seconds=faster_seconds), flush=True)
    result = dict(result_source='fresh_run', scope='densest_query_runtime_and_algorithm_equivalence_only',
        candidate_receipt=pack.sha(candidate_receipt), donor_checkpoint_freeze=pack.sha(freeze_path),
        code_sha256=pack.sha(Path(search.__file__)), query_source=name, agents=largest, seeds=rows,
        future_xy_loaded=False, future_index_loaded=False, forecasts_scored=False, split_assigned=False,
        chosen_policy_saved=False, independent_confirmation=False, original_2pct_unchanged=True)
    pack.write_json(out, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt', type=Path, default=bank.PUBLIC / 'receipt.json')
    ap.add_argument('--candidate-cache', type=Path, default=bank.CACHE)
    ap.add_argument('--input-cache', type=Path, default=pack.CACHE)
    ap.add_argument('--output', type=Path, default=bank.PUBLIC / 'equal_radius_check.json')
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run(a.receipt, a.candidate_cache, a.input_cache, a.output)
