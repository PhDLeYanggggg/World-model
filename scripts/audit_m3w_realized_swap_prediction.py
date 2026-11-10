"""Check value discrimination separately from the fixed-count allocation."""

from collections import defaultdict
from pathlib import Path
import sys

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_realized_swap_value as run

np = run.np


def inspect(packet, f, d, decisions, original):
    cv, anchor, bounds = run.old.geometry(packet, f)
    y, valid = d['future'], d['valid']
    chosen, scale, predictions = (decisions[k] for k in ('actions', 'scale', 'prediction'))
    if not np.array_equal(chosen, original['actions']):
        raise ValueError('Changed frozen nominations')
    observed, estimate = [], []
    for i, (q, a, b, new, x) in enumerate(run.menus(packet, f, cv, anchor, bounds)):
        j = int(chosen[i])
        if j < 0:
            continue
        s = slice(j, j+1)
        _, weight, info = run.model.geometry(cv[q], anchor[q], a[s], b[s], new[s], x[s])
        np.testing.assert_allclose(weight[0], scale[i], atol=1e-10, rtol=0.)
        value = run.model.targets(y[q], valid[q], a[s], b[s], info)
        o = ((value[:, [0, 2]]-value[:, [1, 3]])*weight)[0]
        p = (predictions[i, [0, 2]]-predictions[i, [1, 3]])*scale[i]
        observed.append(o)
        estimate.append([*p, original['raw_gain'][i]])
    return np.asarray(observed), np.asarray(estimate)


def metric(o, p, baseline):
    good = o > 0.
    error, zero = float(np.abs(o-p).mean()), float(np.abs(o).mean())
    auc = float(roc_auc_score(good, p)) if good.any() and not good.all() else None
    alternative = float(roc_auc_score(good, baseline)) if auc is not None and baseline is not None else None
    return dict(positive=int(good.sum()), count=len(o), model_auc=auc, archived_gain_auc=alternative,
                native_mae=error, zero_mae=zero, mean_prediction=float(p.mean()), mean_target=float(o.mean()))


def main():
    key, source, original, parent, _ = run.register()
    f, r = (run.OUT/k for k in ('freeze.json', 'readout.json'))
    freeze, scored = run.read(f), run.read(r)
    if (freeze['key'] != key or scored['freeze'] != run.sha(f) or scored['external_outcomes_read']
            or not scored['original_2pct']):
        raise ValueError('Require a completed development experiment')
    features, data = run.old.prior.load_features(source), run.base.parent.load(source)
    scores = defaultdict(list)
    for name, cell in original['cells'].items():
        run.old.check_cell(cell)
        for held, orig in cell['packets'].items():
            tag = f"{held}_{cell['seed']}"
            local = freeze['cells'][tag]['packets'][name]
            old = parent['cells'][tag]['packets'][name]
            p, q = run.LOCAL/local['file'], run.prior.LOCAL/old['file']
            if (run.sha(p) != local['sha256'] or run.sha(q) != old['sha256']
                    or local['nomination'] != old['sha256'] or local['original'] != orig['sha256']):
                raise ValueError('Frozen nomination or prediction hash changed')
            with run.prior.prior.packet(orig) as z, np.load(p, allow_pickle=False) as a, np.load(q, allow_pickle=False) as b:
                scores[held, cell['seed']].append(inspect(z, features[held], data[held], a, b))
    rows = {}
    for (site, seed), x in scores.items():
        y, p = (np.concatenate(z) for z in zip(*x, strict=True))
        rows[f'{site}/{seed}'] = dict(trajectory=metric(y[:, 0], p[:, 0], p[:, 2]),
                                      relative=metric(y[:, 1], p[:, 1], None))
    if len(rows) != 21 or sum(v['trajectory']['count'] for v in rows.values()) != 8814:
        raise ValueError('Nominated action roster changed')

    def avg(metric, field):
        v = [x[metric][field] for x in rows.values() if x[metric][field] is not None]
        return float(np.mean(v)) if v else None

    summary = {k: {m: avg(k, m) for m in ('model_auc', 'archived_gain_auc', 'native_mae', 'zero_mae',
                                          'mean_prediction', 'mean_target')} for k in ('trajectory', 'relative')}
    run.write(run.OUT/'value_audit.json', dict(readout=run.sha(r), fit_freeze=run.sha(f), by_site_seed=rows,
        summary=summary, positive='conservative_gain_above_zero', sites=7, seed_repeats=3,
        no_fit_or_threshold_changes=True, external_outcomes_read=False), immutable=True)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
