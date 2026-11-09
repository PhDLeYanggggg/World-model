"""Resolve expanded-menu LP rounding gaps with bounded integer feasibility."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from scripts import eval_m3w_cv_risk_frontier_transfer as transfer

audit = transfer.audit
study, old, base, prior, np = audit.study, audit.old, audit.base, audit.prior, audit.np
SECONDS = 8.
MIN_PP = 1e-6


def solve(blocks, ref, offset, cap):
    data = np.concatenate(blocks)
    widths = np.array([len(x) for x in blocks])
    start = np.r_[0, widths.cumsum()]
    n = len(data)
    g = data[:, :2] * (100. / ref)
    d = max(ref / 12., cap, 1.)
    extra = sparse.csr_matrix(np.vstack((g.T, data[:, 2:5].T / d, data[:, 5])))
    one = sparse.coo_matrix((np.ones(n), (np.repeat(np.arange(len(widths)), widths), np.arange(n))),
        shape=(len(widths), n)).tocsr()
    mat = sparse.vstack((one, extra), format='csc')
    lo = np.r_[np.ones(len(widths)), MIN_PP, MIN_PP - offset * (100. / ref), -np.inf, -np.inf, -np.inf, 1.]
    hi = np.r_[np.ones(len(widths)), np.inf, np.inf, 0., 0., cap / d, np.inf]
    res = milp(np.zeros(n), integrality=np.ones(n), bounds=Bounds(0., 1.),
        constraints=LinearConstraint(mat, lo, hi), options={'time_limit': SECONDS})
    if res.x is None:
        return None, 'integer_infeasible_at_margin' if res.status == 2 else 'open'
    picks = [int(res.x[a:b].argmax()) for a, b in zip(start[:-1], start[1:], strict=True)]
    return picks, 'candidate'


def main():
    r = audit.OUT / 'readout.json'
    frozen = old.read(r)
    if not frozen['labels_hindsight_only'] or not frozen['original_2pct'] or frozen['a_labels_opened'] or frozen['independent_confirmation']:
        raise ValueError('Require development-only menu audit')
    reg = old.read(audit.OUT / 'registration.json')
    cuts = old.read(old.checked(reg['parents'][1]))['cells']
    base.register()
    data = base.fit_population(lambda **_: None)
    rows, attempts = [], 0
    for pointer, cell in zip(frozen['cells'], cuts, strict=True):
        c = old.read(old.checked(pointer))
        ident = c['binding']['identity']
        if cell['identity'] != ident:
            raise ValueError('Integer audit source role changed')
        cut = cell['cut']
        _, h = base.subset(data, ident['held'])
        cv, ols, bank = study.parent.analytic(h['history'])
        stem = f'{ident["fold"]}_{ident["seed"]}'
        packet, menu = transfer.restore(h, stem)
        if menu != c['choices']:
            raise ValueError('Integer audit menu differs from earlier feasibility audit')
        recs = np.unique(h['recording'])
        for row, rec in zip(c['rows'], recs, strict=True):
            if row['expanded']['status'] == 'open':
                p = [x for x in packet if h['recording'][x['at'][0]] == rec]
                blocks, plans, roster, ref, offset, cap = audit.original.record_problem(h, cv, ols, bank, p, cut[0])
                chosen, status = solve(blocks, ref, offset, cap)
                if chosen is not None:
                    proof = audit.original.check(h, cv, ols, bank, roster, plans, chosen, cut)
                    if proof is not None:
                        status = 'verified'
                        row['expanded']['witness_pp'] = proof
                        if row['full_rows']['status'] not in ('verified', 'unresolved'):
                            raise ValueError('Expanded integer witness contradicts row feasibility')
                        if row['full_rows']['status'] != 'verified':
                            row['full_rows'].update(status='verified', witness_pp=proof, inherited_witness=True)
                    else:
                        status = 'open'
                row['expanded']['status'] = status
                attempts += 1
                print(dict(cell=stem, attempted=attempts, status=status), flush=True)
            rows.append(row)
    summary, effects = audit.summarize(rows)
    old.once(audit.OUT / 'integer_readout.json', old.stable(dict(parent=old.artifact(r),
        code=old.artifact(Path(__file__)), attempts=attempts, summary=summary, effects=effects, min_pp=MIN_PP,
        seconds_per_record=SECONDS, labels_hindsight_only=True, original_2pct=True,
        independent_confirmation=False, a_labels_opened=False)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
