"""Separate trajectory capacity from risk and the placement of edits."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_ols_anchor_menu as study

old, np = study.old, study.np
previous, base, prior = study.previous, study.base, study.prior


def upper(gain, q, selected, mode):
    """Maximize each reference separately; not a joint feasible witness."""
    if mode == 'record':
        return np.sort(gain, axis=0)[-int(selected.sum()):].sum(0) if selected.any() else np.zeros(2)
    if mode != 'query':
        raise ValueError('Unknown budget mode')
    value = np.zeros(2)
    for g in prior.query_groups(q):
        k = int(selected[g].sum())
        if k:
            value += np.sort(gain[g], axis=0)[-k:].sum(0)
    return value


def best_in_bank(history, y, valid):
    b = prior.study.motion.bank(history)
    cv, ols = b[:, 0], b[:, 1]
    out = np.empty((len(cv), 2))
    for i in range(0, len(cv), 512):
        a = slice(i, i+512)
        c, o = cv[a], ols[a]
        rays = base.expanded(c, b[a, 1:])
        old_bank = np.stack([previous.rollout(c, rays, j)
                             for j in range(rays.shape[1]*previous.amplitude.M)], axis=1)
        paths = np.concatenate((old_bank, study.model.candidates(history[a])), axis=1)
        _, _, gain, _ = study.candidate_terms(c, o, paths, y[a], valid[a])
        legal = (paths != c[:, None]).any((2, 3))
        out[a] = np.where(legal[..., None], gain, -np.inf).max(1)
    mass, offset, _, _ = study.candidate_terms(cv, ols, cv[:, None], y, valid)
    return out, mass, offset


def main():
    src = study.OUT / 'readout.json'
    r = old.read(src)
    if (r['registration']['sha256'] != old.sha(study.OUT / 'registration.json')
            or r['a_labels_opened'] or r['independent_confirmation']):
        raise ValueError('Require the checked exposed-development readout')
    status = {(x['key']['cell'], x['key']['recording']): x['expanded']['status'] for x in r['views']}
    base.register()
    data = base.fit_population(lambda **_: None)
    rows, cache = [], {}
    for i, c in enumerate(old.read(previous.OUT / 'freeze.json')['cells']):
        ident = c['identity']
        _, h = base.subset(data, ident['held'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(h['ids'], z['ids'])
            anchor = z['anchor']
        for record in np.unique(h['recording']):
            key = i, int(record)
            if status[key] != 'nonpositive_bound':
                continue
            take = h['recording'] == record
            q, history, y, valid = (h[k][take] for k in ('query', 'history', 'target', 'valid'))
            if record not in cache:
                cache[record] = (h['ids'][take], *best_in_bank(history, y, valid))
            ids, gain, mass, offset = cache[record]
            np.testing.assert_array_equal(ids, h['ids'][take])
            w = old.weight(q)
            den = float(w @ mass)
            if den <= 0.:
                raise ValueError('Undefined CV reference')
            base_gain = np.array([0., float(w @ offset)])
            edit = anchor[take] >= 0
            bounds = {}
            for mode in ('query', 'record'):
                value = 100.*(base_gain+upper(gain*w[:, None], q, edit, mode))/den
                bounds[mode] = [float(x) if np.isfinite(x) else None for x in value]
            rows.append(dict(cell=i, recording=int(record), edits=int(edit.sum()), bounds=bounds))
        print(dict(cell=i, examined=len(rows)), flush=True)
    if len(rows) != r['summary']['statuses']['expanded']['nonpositive_bound']:
        raise ValueError('Roster mismatch')
    counts = Counter()
    for row in rows:
        q, rec = (row['bounds'][k][1] for k in ('query', 'record'))
        if q is None or rec is None or rec < q-1e-7:
            raise ValueError('Unresolved or nonnested bound')
        if rec <= 0.:
            counts['still_nonpositive_after_reallocating'] += 1
        elif q <= 0.:
            counts['positive_upper_only_after_reallocating'] += 1
        else:
            counts['positive_upper_at_exact_query_counts'] += 1
    result = dict(parent=old.artifact(src), code=old.artifact(Path(__file__)), examined=len(rows),
        counts=dict(counts), bounds=rows, risk_relaxed_diagnostic_only=True, original_rule_changed=False,
        independent_confirmation=False, a_labels_opened=False)
    old.once(study.OUT / 'location_readout.json', old.stable(result))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
