"""Count exact safe opportunities in the scored one-recipient menu."""

from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_recipient_transport as run
from src.evaluation.m3w_recording_switch_ceiling import moments

np = run.np
LIMIT = .02
KEYS = ('views', 'safe_swap', 'safe_utility_swap', 'strict_opportunity', 'rescuable',
        'original_dual', 'mean_strict', 'robust_strict', 'mean_rescues', 'robust_rescues')


def safe(m, signed_cap):
    return ((m['all_excess'].sum(-1) <= 0.) & (m['easy_excess'].sum(-1) <= 0.)
            & (m['easy_signed'].sum(-1) <= signed_cap) & (m['ref_easy'].sum(-1) > 0.))


def ceiling(packet, feature, d, cut, actions):
    cv, anchor, bounds = run.geometry(packet, feature)
    ols = packet['ols4']
    y, valid, w = (d[k] for k in ('future', 'valid', 'weight'))
    recs, rec = np.unique(d['recording'], return_inverse=True)
    rix = np.searchsorted(recs, packet['records'])
    np.testing.assert_array_equal(recs[rix], packet['records'])
    m = moments(cv, cv, y, valid, cut[0])
    cap = LIMIT*np.bincount(rec, weights=m['ref_easy'])
    paired = run.base.parent.paired
    ols0 = paired(ols, cv, y, valid)[1].sum(1)
    ols_gain = np.bincount(rec, weights=w*ols0)
    incumbent = run.pair.evaluate(anchor, packet, d, cut)
    stats = Counter(dict.fromkeys(KEYS, 0))
    witness = anchor.copy()
    chosen = []
    for i, (a, b, new, _) in enumerate(run.menus(packet, feature, cv, anchor, bounds)):
        stats['views'] += 1
        before = bool(incumbent['dual'][i])
        stats['original_dual'] += before
        if not len(a):
            continue
        lo, hi = packet['bounds'][i]
        rid = rix[i]
        if not np.all(rec[lo:hi] == rid):
            raise ValueError('Query not wholly within the expected recording')
        r = packet['additive_rows'][i]
        keep = np.where(a == r[0], r[1], r[0])
        use = np.column_stack((keep, b))
        edit = np.stack((anchor[keep], new), axis=1)
        f = use.flatten()
        p = edit.reshape(-1, *cv.shape[1:])
        m = {k: val.reshape(use.shape) for k, val in moments(cv[f], p, y[f], valid[f], cut[0]).items()}
        okay = safe(m, cap[rid])
        gv = (w[f]*paired(cv[f], p, y[f], valid[f])[1].sum(1)).reshape(use.shape).sum(1)
        go = (w[f]*(paired(ols[f], p, y[f], valid[f])[1].sum(1)-ols0[f])).reshape(use.shape).sum(1)+ols_gain[rid]
        strict = okay & (gv > 0.) & (go > 0.)
        gain = (w[a]*paired(anchor[a], cv[a], y[a], valid[a])[1].sum(1)
                + w[b]*paired(cv[b], new, y[b], valid[b])[1].sum(1))
        improve = strict & (gain > 0.)
        stats['safe_swap'] += bool(okay.any())
        stats['safe_utility_swap'] += bool(strict.any())
        stats['strict_opportunity'] += bool(improve.any())
        stats['rescuable'] += bool(improve.any()) and not before
        for arm in ('mean', 'robust'):
            j = actions[arm][i]
            if j >= 0 and improve[j]:
                stats[f'{arm}_strict'] += 1
                stats[f'{arm}_rescues'] += not before
        if improve.any():
            j = int(np.where(improve, gain, -np.inf).argmax())
            witness[a[j]], witness[b[j]] = cv[a[j]], new[j]
            chosen.append(i)
    verified = run.pair.evaluate(witness, packet, d, cut)
    if not verified['dual'][chosen].all() or int(verified['dual'].sum()) != stats['original_dual']+stats['rescuable']:
        raise ValueError('The hindsight witness did not pass the unchanged scorer')
    return dict(stats)


def execute():
    key, source, original, _ = run.register()
    freeze = run.OUT/'freeze.json'
    readout = run.OUT/'readout.json'
    f, r = run.read(freeze), run.read(readout)
    if (r['freeze'] != run.sha(freeze) or f['registration'] != key or r['external_outcomes_read']
            or not r['original_2pct']):
        raise ValueError('Expected the completed registered transport run')
    identity = dict(readout=run.sha(readout), freeze=run.sha(freeze), script=run.sha(Path(__file__)))
    features, data = run.prior.load_features(source), run.base.parent.load(source)
    counts = {s: Counter() for s in run.base.SITES}
    n = 0
    for name, c in original['cells'].items():
        run.check_cell(c)
        for held, packet in c['packets'].items():
            local = run.LOCAL/f'{name}_{held}_ceiling.json'
            a = f['cells'][f"{held}_{c['seed']}"]['packets'][name]
            p = run.LOCAL/a['file']
            sig = dict(identity, packet=packet['sha256'], actions=a['sha256'])
            if a['original'] != packet['sha256'] or a['sha256'] != run.sha(p):
                raise ValueError('Changed original or frozen selection')
            if not local.exists():
                with run.ref.prior.packet(run.prior.study.LOCAL, packet) as z, np.load(p, allow_pickle=False) as act:
                    stats = ceiling(z, features[held], data[held], c['cut'], act)
                run.write(local, dict(identity=sig, stats=stats), immutable=True)
            row = run.read(local)
            if row['identity'] != sig:
                raise ValueError('Changed ceiling resume identity')
            counts[held].update(row['stats'])
            n += 1
        run.write(run.LOCAL/'ceiling_heartbeat.json', dict(pid=os.getpid(), completed=n, total=126))
    totals = sum(counts.values(), Counter())
    if (n != 126 or totals['views'] != r['summary']['counts']['views']
            or totals['original_dual'] != r['summary']['arms']['original']['dual']):
        raise ValueError('The fixed roster changed')
    run.write(run.OUT/'ceiling.json', dict(inputs=identity, totals=dict(totals),
        by_site={s: dict(v) for s, v in counts.items()}, future_oracle_only=True,
        new_external_outcomes=False, risk_rule_unchanged=True, deployment_changed=False), immutable=True)
    print(dict(totals), flush=True)


if __name__ == '__main__':
    run.torch.set_num_threads(4)
    run.torch.set_num_interop_threads(1)
    with (run.LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute()
