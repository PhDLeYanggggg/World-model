"""Separate direction coordination from recipients and displacement on OpenDD."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_m3w_vehicle_relative_ceiling as ref
from src.evaluation.m3w_interaction_frontier import INTERACTION_TOL, pair_reduction
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_vehicle_relative_ceiling import relative_lower
from src.world_model import m3w_vehicle_angular_pair as model


study, base, pair, old, np, torch = ref.study, ref.base, ref.pair, ref.old, ref.np, ref.torch
sha, write = ref.sha, ref.write
OUT = study.OUT.parent/'opendd_vehicle_angular_pair_v1'
LOCAL = study.LOCAL.parent/OUT.name
ARMS = ('original', *model.CAUSAL)
CONTRASTS = tuple((k, 'original') for k in model.CAUSAL)+(('coupled','independent'), ('coupled','shared'))


def score_menu(cv, ols, y, valid, weight, anchor, selected, cut, ols_offset, signed_cap):
    bank = model.candidates(cv[selected], anchor[selected])
    width = bank.shape[1]
    rows = np.repeat(selected, width)
    paths = bank.reshape(-1, *cv.shape[1:])
    first = np.repeat(np.arange(width), width)
    second = np.tile(np.arange(width, 2*width), width)
    m = moments(cv[rows], paths, y[rows], valid[rows], cut[0])
    risk = np.column_stack([m[k] for k in ('all_excess', 'easy_excess', 'easy_signed')])
    safe = ((risk[first]+risk[second] <= [0., 0., signed_cap]).all(1)
            & (m['ref_easy'][first]+m['ref_easy'][second] > 0.))
    paired = base.parent.paired
    local_ols = paired(ols[rows], cv[rows], y[rows], valid[rows])[1].sum(1)
    terms = np.column_stack((paired(cv[rows], paths, y[rows], valid[rows])[1].sum(1),
                            paired(ols[rows], paths, y[rows], valid[rows])[1].sum(1)-local_ols,
                            paired(anchor[rows], paths, y[rows], valid[rows])[1].sum(1)))*weight[rows, None]
    g = terms[first]+terms[second]+[0., ols_offset, 0.]
    dual = safe & (g[:, :2] > 0.).all(1)
    # All non-selected paths are unchanged and are CV. Subtract the shared edge only once.
    relative = relative_lower(cv, anchor, y, valid, rows, paths, first, second)
    win = dual & (g[:, 2] >= -1e-10) & (relative > INTERACTION_TOL)
    p, q = first, second-width
    count = dict(original_dual=int(dual[0]), angular_dual=int(dual.any()),
        any_relative_win=int(win.any()), shared_relative_win=int((win & (p == q)).any()),
        single_rotation_relative_win=int((win & ((p == 0) ^ (q == 0))).any()),
        two_rotation_relative_win=int((win & (p > 0) & (q > 0)).any()),
        original_dual_and_relative_win=int(dual[0] and win.any()))
    if win.any():
        best = int(np.where(win, relative, -np.inf).argmax())
        witness = anchor.copy()
        witness[selected] = paths[[first[best], second[best]]]
        np.testing.assert_allclose(relative[best], pair_reduction(witness[None], anchor, y, valid)[0, 1], atol=1e-9, rtol=0.)
    return count, bank


def evaluate(packet, d, cut):
    cv, ols = packet['cv'], packet['ols4']
    y, valid, weight = (d[k] for k in ('future', 'valid', 'weight'))
    names, rec = np.unique(d['recording'], return_inverse=True)
    anchor = pair.prediction(packet, 'additive')
    paths = {k: anchor.copy() for k in ARMS}
    ols_offset = np.bincount(rec, weights=weight*base.parent.paired(ols, cv, y, valid)[1].sum(1))
    signed_cap = .02*np.bincount(rec, weights=moments(cv, cv, y, valid, cut[0])['ref_easy'])
    count, rel = Counter(), {f'{a}_minus_{b}': np.zeros(2) for a, b in CONTRASTS}
    for k, (name, (lo, hi)) in enumerate(zip(packet['records'], packet['bounds'], strict=True)):
        q = slice(lo, hi)
        at = np.searchsorted(names, name)
        if names[at] != name or not np.all(rec[q] == at):
            raise ValueError('Wrong query-to-recording binding')
        selected = packet['additive_rows'][k]
        if selected.shape != (2,) or len(set(selected)) != 2 or not np.all((lo <= selected) & (selected < hi)):
            raise ValueError('Expected two distinct fixed recipients at the same query')
        ix = selected-lo
        c, bank = score_menu(cv[q], ols[q], y[q], valid[q], weight[q], anchor[q], ix, cut, ols_offset[at], signed_cap[at])
        count.update(c)
        count.update(views=1, known_pair_query=int((valid[q].sum(0) >= 2).any()))
        for arm, choice in model.choices(bank, ols[selected]).items():
            paths[arm][selected] = bank[np.arange(2), choice]
        for a, b in CONTRASTS:
            rel[f'{a}_minus_{b}'] += pair_reduction(paths[a][None, q], paths[b][q], y[q], valid[q])[0]
    result = {}
    for k, p in paths.items():
        x = pair.evaluate(p, packet, d, cut)
        result[k] = dict(dual=int(x['dual'].sum()), risk=int(x['risk'].sum()), missing=int(x['missing'].sum()), p95=x['p95'])
    if result['original']['dual'] != count['original_dual'] or len({r['missing'] for r in result.values()}) != 1:
        raise ValueError('Original evaluator and exact menu disagree on risk support')
    roster = np.isin(d['recording'], packet['records'])
    mass = (weight[roster]*np.where(valid[roster], np.linalg.norm(cv[roster]-y[roster], axis=-1), 0.).sum(1)).sum()
    effects = {}
    for a, b in CONTRASTS:
        key = f'{a}_minus_{b}'
        observed, lower = base.parent.paired(paths[b][roster], paths[a][roster], y[roster], valid[roster])
        effects[key] = dict(observed_pp=float(100.*(weight[roster]*observed.sum(1)).sum()/mass),
            lower_pp=float(100.*(weight[roster]*lower.sum(1)).sum()/mass),
            pair_known=float(rel[key][0]/max(1, count['known_pair_query'])), pair_lower=float(rel[key][1]/count['views']))
    return dict(counts=dict(count), arms=result, contrasts=effects)


def summarize(rows):
    by_site = {s: [r for r in rows if r['identity']['site'] == s] for s in base.SITES}
    draws = np.random.default_rng(2917).integers(len(by_site), size=(3000, len(by_site)))

    def ci(key, field):
        x = np.asarray([np.mean([np.mean([r['contrasts'][key][field] for r in group if r['identity']['seed'] == seed])
                                for seed in base.SEEDS]) for group in by_site.values()])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    return dict(counts=dict(sum((Counter(r['counts']) for r in rows), Counter())),
                arms={k: {f: sum(r['arms'][k][f] for r in rows) for f in ('dual','risk','missing')} for k in ARMS},
                paired={k: {f: ci(k, f) for f in rows[0]['contrasts'][k]} for k in rows[0]['contrasts']})


def main():
    source, _, frozen, previous, _ = ref.settings()
    path = OUT/'registration.json'
    files = (Path(__file__), Path(model.__file__), OUT/'protocol.md', study.OUT/'freeze.json', study.OUT/'readout.json',
             Path(relative_lower.__code__.co_filename))
    write(path, dict(inputs={str(q.resolve().relative_to(base.prior.ROOT)): sha(q) for q in files},
        angles=model.DEGREES, no_training=True, original_2pct=True, external_outcomes_read=False,
        fixed_recipients=True, fixed_stepwise_norm=True), immutable=True)
    data, rows, recordings = base.parent.load(source), [], set()
    key = sha(path)
    for name, cell in frozen['cells'].items():
        excluded, seed = sorted(cell['packets']), cell['seed']
        if set(cell['fit_sites']) != set(base.SITES)-set(excluded):
            raise ValueError('Additive producer used held outcomes')
        for site, r in cell['packets'].items():
            f = LOCAL/f'{name}_{site}.json'
            ident = dict(registration=key, packet=r['sha256'], site=site, seed=seed)
            with ref.prior.packet(study.LOCAL, r) as packet:
                recordings.update(packet['records'].tolist())
                if not f.exists():
                    write(f, dict(identity=ident, **evaluate(packet, data[site], cell['cut'])), immutable=True)
            v = old.read(f)
            if v['identity'] != ident:
                raise ValueError('Resumed cell is not bound to the original packet')
            rows.append(v)
        write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), completed=len(rows), total=126))
        if len(rows) % 14 == 0:
            print(len(rows), '/ 126 complete', flush=True)
    s = summarize(rows)
    if (len(rows), len(recordings), s['counts']['views'], s['arms']['original']['dual']) != (
            126, 493, 8874, previous['summary']['additive']['matched_dual']):
        raise ValueError('Roster or original additive control changed')
    write(OUT/'readout.json', dict(registration=key, summary=s, by_direction=rows,
        unique_recordings=len(recordings), no_training=True, original_2pct=True,
        external_outcomes_read=False, independent_confirmation=False, deployment_changed=False), immutable=True)
    print(s, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
