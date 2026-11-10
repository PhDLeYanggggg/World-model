"""Freeze recipient-by-path controls on the unscored Athens and Swiss pilots."""

import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_pneuma_matched_pilot as pilot
from scripts import build_m3w_swiss_roundabout_proposals as swiss
from src.evaluation import m3w_fixed_recipient_path_choice as chooser

np, torch, learn = pilot.np, pilot.torch, pilot.learn
store = pilot.storage
PUB = pilot.OUT.parent
OUT = PUB/'external_recipient_factorial_v1'
LOCAL = pilot.LOCAL.parent.parent/OUT.name
RULES = dict(I='independent', A='additive', J='joint')
CELLS = tuple(a+b for a in RULES for b in RULES)
CONTRASTS = dict(
    original_joint_independent=('JJ', 'II'), original_joint_additive=('JJ', 'AA'),
    path_j_i_on_i=('IJ', 'II'), path_j_i_on_j=('JJ', 'JI'),
    path_j_a_on_a=('AJ', 'AA'), path_j_a_on_j=('JJ', 'JA'),
    recipient_j_i_with_i=('JI', 'II'), recipient_j_a_with_a=('JA', 'AA'),
)


def candidate_pairs(row, groups):
    """Enumerate only nominated recipient pairs, in original row-major order."""
    pairs = set()
    for r in groups.values():
        a, b = sorted(r)
        if a == b:
            raise ValueError('A matched edit requires two different actors')
        pairs.update((int(i), int(j)) for i in np.flatnonzero(row == a) for j in np.flatnonzero(row == b))
    if not pairs:
        raise ValueError('No common path menu on the fixed recipients')
    return np.asarray(sorted(pairs), dtype=np.int64).T


def original_blocks(row, groups):
    target_i, target_j = candidate_pairs(row, groups)
    target = target_i*len(row)+target_j
    left, right, keep = [], [], []
    for i, j in pilot.work.stream.blocks(row, pilot.work.CHUNK):
        take = np.isin(i*len(row)+j, target)
        if take.any():
            left.append(i)
            right.append(j)
            keep.append(take)
    i, j, select = (np.concatenate(v) for v in (left, right, keep))
    np.testing.assert_array_equal(i[select], target_i)
    np.testing.assert_array_equal(j[select], target_j)
    return i, j, select


@torch.inference_mode()
def cross(h, cv, ols, paths, selected, nets):
    peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(-1)
    row, family = np.nonzero(peak > 0.)
    # Preserve the original floating-point batch shapes near score ties.
    i, j, take = original_blocks(row, selected)
    x, units, _, _ = learn.old.bridge.features(h, cv, ols, paths)
    model = dict(cv=cv, paths=paths, units=units, peak=peak, speed=np.diff(h[:, -4:], axis=1).mean(axis=1))
    relative = learn.relative(model, row[i], row[j], family[i], family[j])
    a, z = pilot.prior.logits(nets, torch.from_numpy(x[row, family]), i, j, torch.from_numpy(relative))
    merit = learn.model.merit(a)
    i, j = i[take], j[take]
    scores = dict(I=merit[i]+merit[j], A=learn.model.merit(.5*(a[i]+a[j])), J=learn.model.merit(z[take]))
    output = {}
    for g, r in selected.items():
        for rule, (idx, value) in chooser.select(row, i, j, scores, r).items():
            p, f = row[idx], family[idx]
            output[g+rule] = (p, f, paths[p, f], value)
    return output


def distance_envelope(cv, left, right):
    """Uniform bounds over the original query roster, not over edits alone."""
    a, b = ({int(i): p for i, p in zip(edit[0], edit[2], strict=True)} for edit in (left, right))
    keys = sorted(a.keys() | b.keys())
    delta = np.stack([a.get(k, cv[k])-b.get(k, cv[k]) for k in keys])
    norm = np.linalg.norm(delta, axis=-1)
    i, j = np.triu_indices(len(keys), 1)
    pair = np.linalg.norm(delta[i]-delta[j], axis=-1).mean(axis=1).sum()
    n = len(cv)
    rel = (pair+(n-len(keys))*norm.mean(1).sum())/(n*(n-1)//2)
    return np.array([norm.mean(1).sum()/n, norm[:, -1].sum()/n, rel,
                     bool((norm > 1e-12).any()), a.keys() != b.keys()])


def packet(history, offsets, bank, frozen, queries, seed, nets):
    cv, ols = bank['cv'], bank['ols4']
    raw = np.stack([bank[k] for k in ('ols4', 'median4', f'self_{seed}')], axis=1)
    paths = learn.old.matched.same_radius(cv, raw, offsets)
    if (queries.ndim != 1 or not np.issubdtype(queries.dtype, np.integer) or
            np.any(np.diff(queries) <= 0) or np.any((queries < 0) | (queries >= len(offsets)-1))):
        raise ValueError('Changed matched query subset')
    out = defaultdict(list)
    for k, q in enumerate(queries):
        lo, hi = offsets[q:q+2]
        s = slice(lo, hi)
        original = {a: frozen[rule+'_rows'][k] for a, rule in RULES.items()}
        if any(np.any((r < lo) | (r >= hi)) for r in original.values()):
            raise ValueError('A frozen edit is outside its causal query')
        cells = cross(history[s], cv[s], ols[s], paths[s], {a: r-lo for a, r in original.items()}, nets)
        for cell, (rows, family, p, value) in cells.items():
            if set(rows+lo) != set(original[cell[0]]):
                raise ValueError('A fixed-recipient path rule changed recipients')
            np.testing.assert_allclose(np.linalg.norm(p-cv[rows+lo], axis=-1).max(1),
                                       frozen['radius'][k], atol=1e-9, rtol=1e-9)
            if cell[0] == cell[1]:
                rule = RULES[cell[0]]
                np.testing.assert_array_equal(rows+lo, original[cell[0]])
                np.testing.assert_array_equal(family, frozen[rule+'_families'][k])
                np.testing.assert_allclose(p, frozen[rule+'_paths'][k], atol=1e-12, rtol=0.)
                np.testing.assert_allclose(value, frozen[rule+'_score'][k], atol=2e-6, rtol=0.)
            for key, val in zip(('rows', 'families', 'paths', 'score'), (rows+lo, family, p, value), strict=True):
                out[f'{cell}_{key}'].append(val)
        for name, (a, b) in CONTRASTS.items():
            out[name].append(distance_envelope(cv[s], cells[a], cells[b]))
    return dict(**{k: np.asarray(v) for k, v in out.items()}, query_index=queries, offsets=offsets,
                radius=frozen['radius'])


def cohort(which):
    athens = which == 'athens'
    pub, local = (pilot.OUT, pilot.LOCAL.parent) if athens else (swiss.source.OUT, swiss.pilot.LOCAL)
    paths = {k: pub/f'{k}_receipt.json' for k in ('prefix', 'forecast', 'matched')}
    p, f, m = (pilot.pilot.read(v) for v in paths.values())
    digests = {k: store.sha(v) for k, v in paths.items()}
    if (m['binding']['prefix'] != digests['prefix'] or m['binding']['forecast'] != digests['forecast']
            or set(p['files']) != set(f['files'])):
        raise ValueError('Prefix, forecast and matched receipts disagree')
    if athens:
        if m['predictive_roles_assigned'] or m['forecast_accuracy_scored'] or m['future_coordinates_loaded']:
            raise ValueError('Require the original unscored Athens packet')
        sources = PUB/'pneuma_source_intake_v1'/'samples.json'
        if p['binding']['samples'] != store.sha(sources):
            raise ValueError('Athens date provenance changed')
        groups = {k: v['day'] for k, v in pilot.pilot.read(sources)['members'].items()}
        seeds, info = f['neural_seeds'], m['files']
        binding = dict(digests, source=store.sha(sources))
    else:
        if m['roles_assigned'] or m['forecasts_scored'] or m['future_coordinates_parsed']:
            raise ValueError('Require the original unscored Swiss packet')
        groups = {k: Path(k).stem.rsplit('_', 1)[-1] for k in p['files']}
        seeds, info = f['seeds'], m['packets']
        binding = digests
    if seeds != [17, 29, 43] or set(groups) != set(f['files']) or len(info) != len(groups)*3:
        raise ValueError('Incomplete three-seed external forecast bank')
    return local, p['files'], f['files'], info, groups, binding


def checked(path, expected):
    if store.sha(path) != expected:
        raise ValueError(f'Changed frozen input: {path.name}')
    return path


def summary(outputs):
    out = {}
    for k, x in outputs.items():
        out[k] = dict(views=len(x), path_changes=int(x[:, 3].sum()), recipient_changes=int(x[:, 4].sum()),
                      ade_envelope=float(x[:, 0].mean()), fde_envelope=float(x[:, 1].mean()),
                      relative_envelope=float(x[:, 2].mean()), max_query_ade_envelope=float(x[:, 0].max()))
    return out


def run(which):
    local, prefixes, forecasts, old, groups, binding = cohort(which)
    binding.update(driver=store.sha(Path(__file__)), choice=store.sha(Path(chooser.__file__)),
                   protocol=store.sha(OUT/'protocol.md'))
    folder = LOCAL/which
    folder.mkdir(parents=True, exist_ok=True)
    records, totals = {}, defaultdict(lambda: defaultdict(list))
    for seed in (17, 29, 43):
        nets, heads = pilot.work.heads(seed)
        for name in sorted(groups):
            stem = Path(name).stem
            tag = f'{stem}_{seed}'
            r = old[tag]
            orig = checked(local/'matched'/f'{tag}.npz', r.get('proposals_sha256', r.get('sha256')))
            past = checked(local/f'{stem}.prefix.npz', prefixes[name]['prefix_sha256'])
            bank = checked(local/f'{stem}.forecast.npz', forecasts[name]['forecast_sha256'])
            ident = dict(binding=binding, old=store.sha(orig), past=store.sha(past), bank=store.sha(bank), heads=heads)
            if (r['identity']['head_freeze'] != heads or r['identity']['seed'] != seed or
                    r['identity'].get('name', r['identity'].get('member')) != name or
                    r['identity']['prefix_sha256'] != ident['past'] or r['identity']['forecast_sha256'] != ident['bank']):
                raise ValueError('Frozen original and source roles disagree')
            dest, meta = (folder/f'{tag}.{k}' for k in ('npz', 'json'))
            if not meta.exists():
                with (np.load(past, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as f,
                      np.load(orig, allow_pickle=False) as z):
                    offsets = p['offsets']
                    q = np.arange(len(offsets)-1) if which == 'athens' else z['query_index']
                    expected = r.get('queries', r.get('available'))
                    if len(q) != expected or len(offsets)-1 != r.get('queries', r.get('original')):
                        raise ValueError('Changed matched or full query count')
                    if len(q):
                        np.testing.assert_array_equal(np.diff(offsets)[q], np.diff(z['offsets']))
                        result = packet(p['history'], offsets, f, z, q, seed, nets)
                    else:
                        result = dict(query_index=q, offsets=offsets)
                store.write_json(meta, dict(identity=ident, sha256=store.save_npz(dest, result),
                                           matched=len(q), full=len(offsets)-1))
            v = pilot.pilot.read(meta)
            if v['identity'] != ident:
                raise ValueError('Changed recipient-factorial freeze identity')
            with np.load(checked(dest, v['sha256']), allow_pickle=False) as z:
                if v['matched']:
                    for k in CONTRASTS:
                        totals[groups[name]][k].append(z[k])
            records[tag] = dict(packet_sha256=v['sha256'], original_sha256=ident['old'], source_group=groups[name],
                                seed=seed, matched=v['matched'], full=v['full'])
            hb = dict(pid=os.getpid(), dataset=which, completed=len(records), total=len(old))
            pending = folder/'heartbeat.pending'
            pending.write_text(json.dumps(hb)+'\n')
            pending.replace(folder/'heartbeat.json')
            print(hb, flush=True)
    counts = (sum(r['matched'] for r in records.values()), sum(r['full'] for r in records.values()))
    expected = (1929, 1929) if which == 'athens' else (7431, 7689)
    if counts != expected or set(totals) != set(groups.values()):
        raise ValueError('Changed external pilot coverage')
    by_group = {g: summary({k: np.concatenate(v) for k, v in rows.items()}) for g, rows in sorted(totals.items())}
    overall = summary({k: np.concatenate([v for rows in totals.values() for v in rows[k]]) for k in CONTRASTS})
    result = dict(result_source='fresh_run', binding=binding, packets=records, matched_views=counts[0],
                  full_views=counts[1], original_diagonals_reproduced=True, by_group=by_group,
                  contrasts=overall, future_coordinates_read=False, accuracy_scored=False, roles_assigned=False,
                  training=False, original_risk_limit=.02, deployment_changed=False)
    dest = OUT/f'{which}_freeze.json'
    if dest.exists() and pilot.pilot.read(dest) != result:
        raise ValueError('Refusing to replace differing external freeze')
    if not dest.exists():
        store.write_json(dest, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('athens', 'swiss'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.dataset)
