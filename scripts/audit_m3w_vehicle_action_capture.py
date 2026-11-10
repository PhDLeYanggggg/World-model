"""Separate missed recordings, recipients and directions on the frozen menu."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_action_value as study
from src.evaluation.m3w_interaction_frontier import INTERACTION_TOL
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_vehicle_relative_ceiling import relative_lower


ctx, np, base, pair = study.context, study.np, study.base, study.pair
sha, read, write = study.sha, study.read, study.write
OUT = study.OUT
LOCAL = study.LOCAL


def legal_wins(cv, ols, y, valid, weight, anchor, selected, cut, ols_offset, signed_cap):
    bank = study.model.candidates(cv[selected], anchor[selected])
    n = bank.shape[1]
    rows = np.repeat(selected, n)
    paths = bank.reshape(-1, *cv.shape[1:])
    first = np.r_[np.arange(1, n), np.zeros(n-1, dtype=int)]
    second = np.r_[np.full(n-1, n), np.arange(n+1, 2*n)]
    m = moments(cv[rows], paths, y[rows], valid[rows], cut[0])
    risk = np.column_stack([m[k] for k in ('all_excess', 'easy_excess', 'easy_signed')])
    ok = ((risk[first]+risk[second] <= [0., 0., signed_cap]).all(1)
          & (m['ref_easy'][first]+m['ref_easy'][second] > 0.))
    p = base.parent.paired
    cv0 = p(ols[rows], cv[rows], y[rows], valid[rows])[1].sum(1)
    u = np.column_stack((p(cv[rows], paths, y[rows], valid[rows])[1].sum(1),
                         p(ols[rows], paths, y[rows], valid[rows])[1].sum(1)-cv0,
                         p(anchor[rows], paths, y[rows], valid[rows])[1].sum(1))) * weight[rows, None]
    g = u[first]+u[second]+[0., ols_offset, 0.]
    ix = np.flatnonzero(ok & (g[:, :2] > 0.).all(1) & (g[:, 2] >= -1e-10))
    out = np.zeros(len(first), dtype=bool)
    if len(ix):
        out[ix] = relative_lower(cv, anchor, y, valid, rows, paths, first[ix], second[ix]) > INTERACTION_TOL
    return out.reshape(2, n-1)


def partition(wins, choice):
    if not wins.any():
        return 'no_feasible_improvement'
    actor, angle = choice
    if actor < 0:
        return 'no_rotation'
    if not wins[actor].any():
        return 'wrong_actor'
    return 'captured' if wins[actor, angle-1] else 'wrong_angle'


def evaluate(packet, d, cut, choices):
    cv, ols = packet['cv'], packet['ols4']
    y, v, w = (d[k] for k in ('future', 'valid', 'weight'))
    names, rec = np.unique(d['recording'], return_inverse=True)
    ols_off = np.bincount(rec, weights=w*base.parent.paired(ols, cv, y, v)[1].sum(1))
    cap = .02*np.bincount(rec, weights=moments(cv, cv, y, v, cut[0])['ref_easy'])
    anchor = pair.prediction(packet, 'additive')
    counts = {a: Counter() for a in study.ARMS[1:]}
    for k, (lo, hi) in enumerate(packet['bounds']):
        q = slice(lo, hi)
        at = rec[lo]
        if not np.all(rec[q] == at) or names[at] != packet['records'][k]:
            raise ValueError('Changed query and recording alignment')
        p = packet['additive_rows'][k]-lo
        wins = legal_wins(cv[q], ols[q], y[q], v[q], w[q], anchor[q], p, cut, ols_off[at], cap[at])
        for a, c in zip(study.ARMS[1:], choices[1:, k], strict=True):
            counts[a][partition(wins, c)] += 1
    return {k: dict(v) for k, v in counts.items()}


def main():
    key, _, _, upstream, _ = study.register()
    fp = OUT/'freeze.json'
    freeze = read(fp)
    previous = ctx.prior.prior.OUT/'readout.json'
    p = OUT/'capture_registration.json'
    if freeze['registration'] != key or not freeze['freeze_before_held_score']:
        raise ValueError('Require the completed action-choice freeze')
    write(p, dict(inputs={str(f.resolve().relative_to(base.prior.ROOT)): sha(f) for f in
        (Path(__file__), fp, previous)}, original_2pct=True, oracle_diagnostic=True, external_outcomes_read=False), immutable=True)
    source = read(base.parent.OUT/'cohorts.json')['registration']
    data = ctx.load_features(source)
    r, n = dict.fromkeys(study.ARMS[1:]), 0
    totals = {a: Counter() for a in r}
    for name, c in upstream['cells'].items():
        f = freeze['cells'][name]
        for s, packet in c['packets'].items():
            ch = f['packets'][s]
            out = LOCAL/f'{name}_{s}_capture.json'
            ident = dict(registration=sha(p), actions=ch['sha256'], original=packet['sha256'])
            if not out.exists():
                with ctx.prior.prior.ref.prior.packet(ctx.study.LOCAL, packet) as a, ctx.prior.prior.ref.prior.packet(LOCAL, ch) as b:
                    count = evaluate(a, study.labels(data[s]['file']), c['cut'], b['choices'])
                write(out, dict(identity=ident, counts=count), immutable=True)
            value = read(out)
            if value['identity'] != ident:
                raise ValueError('Unbound diagnostic')
            for a, v in value['counts'].items():
                totals[a].update(v)
            n += 1
        write(LOCAL/'capture_heartbeat.json', dict(pid=os.getpid(), completed=n, total=126))
    original = read(previous)['summary']['counts']['single_rotation_relative_win']
    if n != 126 or any(sum(v.values()) != 8874 or 8874-v['no_feasible_improvement'] != original for v in totals.values()):
        raise ValueError('One-rotation cap must match the existing exhaustive oracle')
    result = dict(registration=sha(p), strict_one_rotation_capacity=original, counts=totals, views=8874, recordings=493,
        original_2pct=True, hindsight_only=True, external_outcomes_read=False, no_training=True)
    write(OUT/'capture_readout.json', result, immutable=True)
    print(result, flush=True)


if __name__ == '__main__':
    with (LOCAL/'capture.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
