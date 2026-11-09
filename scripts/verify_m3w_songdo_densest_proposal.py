"""Verify the optimized proposal against direct models on the largest real query."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_matched_proposals as run

np, torch = run.np, run.torch


@torch.inference_mode()
def verify():
    receipt, bank, heads, binding = run.setup()
    path = run.OUT / 'receipt.json'
    frozen = run.first.read(path)
    if frozen['binding'] != binding or frozen['forecasts_scored'] or frozen['split_assigned']:
        raise ValueError('Require the frozen unscored proposal receipt')
    count, name, seed = max((d['pairs'], n, int(s)) for n, r in frozen['sessions'].items()
                            for s, d in r['seeds'].items() if d is not None)
    info = receipt['sessions'][name]
    stem = Path(name).stem
    cache = run.pack.CACHE if info['cache_role'] == 'previous' else run.first.PRIVATE
    prefix, forecasts, choices = cache / f'{stem}_prefix.npz', run.bank.CACHE / f'{stem}.npz', run.LOCAL / f'{stem}.npz'
    for p, h in ((prefix, info['prefix_sha']), (forecasts, bank['sessions'][name]['sha256']),
                 (choices, frozen['sessions'][name]['packet_sha256'])):
        if run.pack.sha(p) != h:
            raise ValueError('Verification input changed')
    with (np.load(prefix, allow_pickle=False) as a, np.load(forecasts, allow_pickle=False) as b,
          np.load(choices, allow_pickle=False) as saved):
        lo, hi = saved[f'seed{seed}_bounds']
        q = slice(lo, hi)
        h, cv, ols = a['history'][q], b['cv'][q], b['ols4'][q]
        raw = np.stack([b[k][q] for k in ('ols4', 'median4', f'self_{seed}')], axis=1)
        paths = run.old.matched.same_radius(cv, raw, np.array([0, len(cv)]))
        peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1)
        x, unit, *_ = run.old.bridge.features(h, cv, ols, paths)
        m = dict(cv=cv, paths=paths, units=unit, peak=peak, speed=np.diff(h[:, -4:], axis=1).mean(axis=1))
        row, family, i, j = run.learn.choices(peak, 0, len(h))
        if len(i) != count:
            raise ValueError('Largest query size changed')
        x = torch.from_numpy(x[row, family])
        rel = torch.from_numpy(run.learn.relative(m, row[i], row[j], family[i], family[j]))
        per_action, optimized = run.logits(heads[seed], x, i, j, rel)
        blocks = []
        for start in range(0, len(i), run.CHUNK):
            sl = slice(start, start+run.CHUNK)
            u, v, r = x[i[sl]], x[j[sl]], rel[sl]
            blocks.append(np.mean([n(u, v, r).numpy().astype(float) for n in heads[seed]['joint']], axis=0))
        direct = np.concatenate(blocks)
        np.testing.assert_allclose(optimized, direct, atol=1e-6, rtol=1e-6)
        select = run.ranked_pairs(row, family, i, j, per_action, direct)
        for arm, (indices, _) in select.items():
            np.testing.assert_array_equal(saved[f'seed{seed}_{arm}_rows'], lo+row[indices])
            np.testing.assert_array_equal(saved[f'seed{seed}_{arm}_families'], family[indices])
            np.testing.assert_array_equal(saved[f'seed{seed}_{arm}_paths'], paths[row[indices], family[indices]])
    output = dict(proposal_sha256=run.pack.sha(path), session=name, seed=seed, unique_pairs=count,
        actors=len(h), maximum_logit_difference=float(np.max(np.abs(optimized-direct))),
        all_four_selections_reproduce=True, future_coordinates_loaded=False, future_index_loaded=False)
    run.pack.write_json(run.OUT / 'densest_verification.json', output)
    print(output)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    verify()
