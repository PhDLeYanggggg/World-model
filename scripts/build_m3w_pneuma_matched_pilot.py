"""Prepare fixed-count unscored proposals on the split-neutral pilot."""

import fcntl
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_m3w_pneuma_pair_memory as work


pilot, prior, learn, storage, torch, np = work.pilot, work.prior, work.learn, work.storage, work.torch, work.np
OUT, LOCAL = pilot.OUT, pilot.LOCAL/'matched'
ARMS = prior.ARMS


def beat(n, total, name, seed, queries):
    f = LOCAL/'heartbeat.pending'
    f.write_text(json.dumps(dict(pid=os.getpid(), completed=n, total=total, file=name, seed=seed, queries=queries))+'\n')
    f.replace(LOCAL/'heartbeat.json')


def proposals(h, offsets, forecasts, seed, net, progress):
    cv, ols = forecasts['cv'], forecasts['ols4']
    raw = np.stack([forecasts[k] for k in ('ols4', 'median4', f'self_{seed}')], axis=1)
    paths = learn.old.matched.same_radius(cv, raw, offsets)
    peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1)
    expected = prior.bank.bank.count_queries(forecasts, offsets, seed)
    result = {f'{arm}_{suffix}': [] for arm in ARMS for suffix in ('rows', 'families', 'paths', 'score')}
    radii = []
    for k, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        q = slice(lo, hi)
        x, units, _, _ = learn.old.bridge.features(h[q], cv[q], ols[q], paths[q])
        model = dict(cv=cv[q], paths=paths[q], units=units, peak=peak[q], speed=np.diff(h[q, -4:], axis=1).mean(axis=1))
        row, family = np.nonzero(peak[q] > 0.)
        value, n, _ = work.streamed(net, torch.from_numpy(x[row, family]), model, row, family)
        if n*2 != expected[k, 1] or set(value) != set(ARMS):
            raise ValueError('The full causal candidate menu or matched arms changed')
        radius = expected[k, 4]
        for arm, (ij, score) in value.items():
            r, f = row[ij]+lo, family[ij]
            if r[0] == r[1]:
                raise ValueError('An edited pair must use two distinct recipients')
            np.testing.assert_allclose(peak[r, f], radius, atol=1e-9, rtol=1e-9)
            for suffix, a in zip(('rows', 'families', 'paths', 'score'), (r, f, paths[r, f], score), strict=True):
                result[f'{arm}_{suffix}'].append(a)
        radii.append(radius)
        if (k+1) % 10 == 0:
            progress(k+1)
    return dict(**{k: np.asarray(v) for k, v in result.items()}, offsets=offsets, radius=np.asarray(radii))


def run():
    p, q, t = (OUT/f'{k}_receipt.json' for k in ('prefix', 'forecast', 'pair_stream'))
    a, b, c = (pilot.read(k) for k in (p, q, t))
    if (b['binding']['prefix_receipt'] != storage.sha(p) or c['binding']['forecast_receipt'] != storage.sha(q)
            or c['binding']['driver'] != storage.sha(Path(work.__file__)) or not c['exact_indices_and_scores_match']
            or b['forecasts_scored'] or b['predictive_roles_assigned'] or b['future_coordinates_loaded']):
        raise ValueError('Require an exact, split-neutral observed-only pilot')
    bound = dict(prefix=storage.sha(p), forecast=storage.sha(q), pair_stream=storage.sha(t),
                 driver=storage.sha(Path(__file__)), benchmark=storage.sha(Path(work.__file__)))
    files, rows = {}, 0
    total = len(b['neural_seeds'])*len(b['files'])
    for seed in b['neural_seeds']:
        nets, freeze = work.heads(seed)
        for name, info in b['files'].items():
            stem = Path(name).stem
            x, f = (pilot.LOCAL/f'{stem}.{k}.npz' for k in ('prefix', 'forecast'))
            if (storage.sha(x) != info['prefix_sha256'] or storage.sha(f) != info['forecast_sha256']
                    or info['prefix_sha256'] != a['files'][name]['prefix_sha256']):
                raise ValueError('Changed causal bank')
            tag = f'{stem}_{seed}'
            result, meta = (LOCAL/f'{tag}.{k}' for k in ('npz', 'json'))
            identity = dict(binding=bound, head_freeze=freeze, name=name, seed=seed,
                            prefix_sha256=info['prefix_sha256'], forecast_sha256=info['forecast_sha256'])
            if not meta.exists():
                with np.load(x, allow_pickle=False) as d, np.load(f, allow_pickle=False) as y:
                    h, offset = d['history'], d['offsets']
                    if len(offset)-1 != info['queries'] or len(h) != info['agents'] or offset[-1] != len(h):
                        raise ValueError('Changed fixed roster')
                    out = proposals(h, offset, y, seed, nets, lambda n: beat(rows, total, name, seed, n))
                choice = out['joint_rows']*3+out['joint_families']
                changed = {k: int(np.any(choice != out[f'{k}_rows']*3+out[f'{k}_families'], axis=1).sum())
                           for k in ARMS[:-1]}
                storage.write_json(meta, dict(identity=identity, proposals_sha256=storage.save_npz(result, out),
                    queries=len(offset)-1, different_from_joint=changed))
            item = pilot.read(meta)
            if (item['identity'] != identity or item['proposals_sha256'] != storage.sha(result)
                    or item['queries'] != info['queries']):
                raise ValueError('Changed proposal packet')
            files[tag] = item
            rows += 1
            beat(rows, total, name, seed, item['queries'])
            print('Matched', rows, '/', total, name, seed, item['queries'], flush=True)
    storage.write_json(OUT/'matched_receipt.json', dict(result_source='fresh_unscored_fixed_count_proposals',
        binding=bound, files=files, seeds=b['neural_seeds'], arms=ARMS, edits_per_arm_per_query=2,
        queries_per_seed=a['queries'], query_seed_views=sum(r['queries'] for r in files.values()),
        distinct_recipients_verified=True, equal_radius_verified=True, future_coordinates_loaded=False,
        forecast_accuracy_scored=False, predictive_roles_assigned=False, formal_clock_selected=False,
        release_threshold_selected=False, original_2pct=True, new_fit=False, deployment_changed=False))


if __name__ == '__main__':
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
