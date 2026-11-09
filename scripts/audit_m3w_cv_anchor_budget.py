"""Separate insufficient edit budget from learning failure on the moving A roster."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_motion_cost_stability as parent
from src.evaluation import m3w_cv_anchor_budget as audit
from src.evaluation.m3w_selection_forecast_readout import policy

np, read, sha, once, stable, artifact = parent.np, parent.read, parent.sha, parent.once, parent.stable, parent.artifact
OUT = parent.OUT.parent / 'moving_cv_anchor_budget_v1'
CAPS = (0, 1, 2, 4, 8, 16, 32, 'unlimited')
REFERENCES = ('cv', 'ols4')
MULTIPLIERS = (0., .001, .01, .1, 1.)


def record(cv, ols, bank, truth, valid, query, easy, hard):
    order, offsets = audit.row_ranks(query)
    _, inv, counts = np.unique(query, return_inverse=True, return_counts=True)
    w = 1. / counts[inv]
    mass, fallback, delta, arbitrary = audit.candidate_terms(cv, ols, bank, truth, valid)
    ref, base = float(w @ mass), np.array([0., w @ fallback])
    s = w[:, None, None] * delta
    curves = np.stack([audit.cap_curve(s[..., j].max(1), order, offsets) for j in range(2)])
    generous = np.stack([audit.cap_curve(w * arbitrary[:, j], order, offsets) for j in range(2)])
    if np.any(curves > generous + 1e-8):
        raise ValueError('Finite action menu exceeded the arbitrary path bound')
    cost = audit.risk_excess(cv, bank, truth, valid, easy)
    penalties = curves.copy()
    for a in MULTIPLIERS:
        for b in MULTIPLIERS:
            if a or b:
                adjusted = s - (a * cost[..., :1] + b * cost[..., 1:])
                for j in range(2):
                    penalties[j] = np.minimum(penalties[j], audit.cap_curve(adjusted[..., j].max(1), order, offsets))
    witnesses = {}
    for j, name in enumerate(REFERENCES):
        choice = audit.choice_at_cap(s[..., j], order, offsets, 1)
        use = choice >= 0
        forecast = cv.copy()
        forecast[use] = bank[use, choice[use]]
        m = base + s[use, choice[use]].sum(0)
        np.testing.assert_allclose(m[j], base[j] + curves[j, 1], rtol=1e-10, atol=1e-8)
        risk = policy(forecast, cv, cv, truth, valid, np.repeat('r', len(use)),
                      easy_cut=easy, hard_cut=hard)['recordings']['r']
        witnesses[name] = dict(edits=int(use.sum()), gain=m.tolist(), risk=risk,
                               joint_pass=bool((m > 0.).all() and risk['supported']))
    matrix = np.stack((curves, penalties, generous), axis=0)
    picks = [min(k, curves.shape[1]-1) if isinstance(k, int) else curves.shape[1]-1 for k in CAPS]
    out = matrix[:, :, picks] + base[None, :, None]
    return dict(reference=ref, scores=out.tolist(), witnesses=witnesses,
                minimum_cap_vs_ols4={k: audit.minimum_cap(v[1], base[1]) for k, v in
                                    zip(('menu', 'risk_relaxed', 'arbitrary'), matrix, strict=True)})


def summarize(rows):
    groups = sorted({r['source'] for r in rows})
    source = []
    for g in groups:
        sub = [r for r in rows if r['source'] == g]
        d = sum(r['reference'] for r in sub)
        if d <= 0.:
            raise ValueError('Source CV reference not defined')
        source.append(100 * np.sum([r['scores'] for r in sub], axis=0) / d)
    s = np.stack(source)
    rng = np.random.default_rng(73107)
    ids = rng.integers(len(groups), size=(3000, len(groups)))
    avg = s.mean(0)
    lo, hi = np.quantile(s[ids].mean(1), [.025, .975], axis=0)
    out = np.asarray([r['scores'] for r in rows])
    support = (out > 0.).sum(0)
    return dict(records=len(rows), sources=len(groups), draws=len(ids), axis_order=['bound', 'reference', 'cap'],
                bounds=['menu', 'risk_relaxed', 'arbitrary'], references=REFERENCES, caps=CAPS,
                point_pp=avg.tolist(), interval95=np.stack((lo, hi), -1).tolist(), positive_records=support.tolist(),
                any_hindsight_witness=sum(any(w['joint_pass'] for w in r['witnesses'].values()) for r in rows),
                no_ols4_at_cap_one_even_arbitrary=int((out[:, 2, 1, 1] <= 0.).sum()),
                minimum_cap_vs_ols4={k: [r['minimum_cap_vs_ols4'][k] for r in rows] for k in
                                    ('menu', 'risk_relaxed', 'arbitrary')})


def main():
    previous = parent.OUT / 'sparse_anchor_risk.json'
    p = read(previous)
    if p['recording_count'] != 34 or not p['original_2pct'] or p['independent_confirmation']:
        raise ValueError('Require the verified prior anchor bound')
    reg = OUT / 'registration.json'
    once(reg, stable(dict(parent=artifact(previous), files=[artifact(p) for p in
        (Path(__file__), Path(audit.__file__), Path(parent.model.motion.__file__), OUT / 'protocol.md')],
        caps=CAPS, multipliers=MULTIPLIERS, independent_confirmation=False, original_2pct=True)))
    old = parent.prior.bank.old
    a = parent.a_inputs()
    data = old.prior.arrays(('target_eval', 'origin', 'valid'))
    truth, valid = data['target_eval'] - data['origin'][:, None], data['valid']
    b = parent.model.motion.bank(a['history'])
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s: cuts[str(v['job']['producer_fold'])] for v in read(old.PARENTS[3])['forecasts']
           for s in v['job']['donor_sites']}
    rows = []
    for key in np.unique(a['recordings']):
        ix = a['recordings'] == key
        local = b[ix]
        s = np.unique(a['sites'][ix])
        if len(s) != 1:
            raise ValueError('Recording spans source groups')
        c = cut[s[0]]
        r = record(local[:, 0], local[:, 1], local[:, 1:], truth[ix], valid[ix], a['queries'][ix], c['easy'], c['hard'])
        rows.append(dict(r, source=str(s[0]), recording=str(key)))
    stat = summarize(rows)
    if (len(a['history']), stat['records'], stat['sources']) != (125540, 34, 14):
        raise ValueError('Original development roster changed')
    once(OUT / 'readout.json', stable(dict(registration=sha(reg), summary=stat,
        original_2pct=True, new_fits=False, independent_confirmation=False, submission_ready=False)))
    private = parent.PRIVATE.parent / OUT.name
    once(private / 'recordings.json', stable(dict(registration=sha(reg), rows=rows)))
    print({k: stat[k] for k in ('records', 'sources', 'no_ols4_at_cap_one_even_arbitrary', 'any_hindsight_witness')})


if __name__ == '__main__':
    main()
