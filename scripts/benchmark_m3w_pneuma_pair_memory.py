"""Rehearse exact dense-roster ranking on an observed-only engineering input."""

import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_pneuma_prefix_pilot as pilot
from scripts import build_m3w_songdo_matched_proposals as prior
from src.evaluation import m3w_matched_pair_stream as stream


np, torch, storage = pilot.np, pilot.torch, pilot.storage
learn = prior.learn
SEED = 17
CHUNK = 4096


def heads(seed=SEED):
    path = learn.OUT/'freeze.json'
    key, *_ = learn.register()
    f = pilot.read(path)
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Need the original frozen OpenDD heads')
    result = {a: [] for a in ('additive', 'joint')}
    for c in (x for _, x in sorted(f['cells'].items()) if x['seed'] == seed):
        for arm in result:
            r = c['weights'][arm]
            i = r['identity']
            if (i['registration'] != key or i['seed'] != seed or i['role'] != arm or r['step'] != learn.STEPS
                    or i['fit_sites'] != c['fit_sites'] or len(i['excluded']) != 2):
                raise ValueError('Unexpected checkpoint provenance')
            result[arm].append(prior.base.load_net(learn.LOCAL/f"{i['name']}.pt", r,
                                                learn.model.PairValue(arm == 'joint'), i['excluded']))
    if any(len(v) != 21 for v in result.values()):
        raise ValueError('Missing pre-existing seed ensemble')
    return result, storage.sha(path)


@torch.inference_mode()
def streamed(ensembles, x, model, row, family):
    a = sum((n.head(n.actor(x)).numpy().astype(float) for n in ensembles['additive']),
            start=np.zeros((len(x), 4)))/len(ensembles['additive'])
    embed = [n.actor(x) for n in ensembles['joint']]
    utility = learn.model.merit(a)
    best, candidates, maximum = {}, 0, 0
    for i, j in stream.blocks(row, CHUNK):
        r = torch.from_numpy(learn.relative(model, row[i], row[j], family[i], family[j]))
        out = np.zeros((len(i), 4))
        for n, z in zip(ensembles['joint'], embed, strict=True):
            u, v = z[i], z[j]
            out += n.head(torch.cat((.5*(u+v), (u-v).abs(), r), dim=1)).numpy()
        y = learn.model.merit(out/len(ensembles['joint']))
        stream.update_best(best, i, j, dict(independent=utility[i]+utility[j],
            additive=learn.model.merit(.5*(a[i]+a[j])), same_family=np.where(family[i] == family[j], y, -np.inf), joint=y))
        candidates += len(i)
        maximum = max(maximum, len(i))
    return best, candidates, maximum


def main():
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    inp, pred, slots = (pilot.OUT/f'{k}_receipt.json' for k in ('prefix', 'forecast', 'slot'))
    b, c = pilot.read(pred), pilot.read(slots)
    if (b['binding']['prefix_receipt'] != storage.sha(inp) or c['forecast_receipt'] != storage.sha(pred)
            or c['ranked'] or b['forecasts_scored'] or c['predictive_roles_assigned']):
        raise ValueError('Require the unranked pilot')
    # Predeclare one dense window from the label-free slot count, not from any accuracy.
    name = max(c['per_file'], key=lambda f: c['per_file'][f]['by_seed'][str(SEED)]['max_ordered_pairs_in_query'])
    stem = Path(name).stem
    p, q = (pilot.LOCAL/f'{stem}.{k}.npz' for k in ('prefix', 'forecast'))
    if storage.sha(p) != b['files'][name]['prefix_sha256'] or storage.sha(q) != b['files'][name]['forecast_sha256']:
        raise ValueError('Changed pilot arrays')
    with np.load(p, allow_pickle=False) as x, np.load(q, allow_pickle=False) as y:
        offsets = x['offsets']
        ix = int(prior.bank.bank.count_queries(y, offsets, SEED)[:, 1].argmax())
        start, stop = offsets[ix:ix+2]
        h = x['history'][start:stop]
        cv, ols = (y[k][start:stop] for k in ('cv', 'ols4'))
        raw = np.stack([y[k][start:stop] for k in ('ols4', 'median4', f'self_{SEED}')], axis=1)
    paths = learn.old.matched.same_radius(cv, raw, np.array([0, len(h)]))
    peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1)
    features, units, _, _ = learn.old.bridge.features(h, cv, ols, paths)
    model = dict(cv=cv, paths=paths, units=units, peak=peak, speed=np.diff(h[:, -4:], axis=1).mean(axis=1))
    net, freeze = heads()
    row, family = np.nonzero(peak > 0.)
    x = torch.from_numpy(features[row, family])
    t = perf_counter()
    actual, n, maximum = streamed(net, x, model, row, family)
    streamed_seconds = perf_counter()-t
    t = perf_counter()
    r, f, i, j = learn.choices(peak, 0, len(h))
    np.testing.assert_array_equal(r, row)
    np.testing.assert_array_equal(f, family)
    rel = learn.relative(model, row[i], row[j], family[i], family[j])
    a, y = prior.logits(net, x, i, j, torch.from_numpy(rel))
    expected = prior.ranked_pairs(row, family, i, j, a, y)
    dense_seconds = perf_counter()-t
    if n != len(i) or n*2 != c['totals'][str(SEED)]['max_ordered_pairs_in_query']:
        raise ValueError('Stream did not cover the dense candidate menu')
    if set(actual) != set(expected):
        raise ValueError('Missing matched arm')
    for k, (ids, score) in actual.items():
        np.testing.assert_array_equal(ids, expected[k][0])
        np.testing.assert_allclose(score, expected[k][1], rtol=0., atol=1e-12)
    binding = dict(forecast_receipt=storage.sha(pred), slot_receipt=storage.sha(slots), head_freeze=freeze,
                   driver=storage.sha(Path(__file__)), stream_code=storage.sha(Path(stream.__file__)))
    tag = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    report = dict(binding=binding, test_id=tag, seed=SEED, agents=len(h), distinct_recipient_unordered_candidates=n,
        maximum_live_pair_rows=maximum, dense_relative_feature_bytes=rel.nbytes,
        exact_indices_and_scores_match=True, arms=list(prior.ARMS), future_positions_loaded=False,
        predictive_roles_assigned=False, outcomes_scored=False, policies_fitted=False, engineering_only=True)
    storage.write_json(pilot.OUT/'pair_stream_receipt.json', report)
    print(dict(**report, stream_seconds=round(streamed_seconds, 3), dense_seconds=round(dense_seconds, 3)), flush=True)


if __name__ == '__main__':
    main()
