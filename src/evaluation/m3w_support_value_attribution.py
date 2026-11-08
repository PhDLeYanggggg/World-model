"""Separate easy-reference availability, signed harm and gain at fixed counts."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_streaming_menu_capacity import POSITIVE_PP, respects_tokens, token_rows
from src.world_model import m3w_direct_count_value as value
from src.world_model.m3w_ols_replacement import rollout


def prefix_errors(h):
    out = []
    for k in (4, 5, 6):
        cv, ols, _, _ = rollout(h[:, :k], 8-k)
        y = h[:, k:]-h[:, k-1:k]
        bank = cv[:, None]+value.MIX[None, :, None, None]*(ols-cv)[:, None]
        ec = np.linalg.norm(cv-y, axis=-1).mean(1)
        ea = np.linalg.norm(bank-y[:, None], axis=-1).mean(2)
        out.append((ec, ea))
    return out


def prefix_scores(errors, donor_cuts, row, arm, scale):
    if len(errors) != len(donor_cuts):
        raise ValueError("Require one donor-only cutoff per prefix")
    feats = []
    for (ec, ea), cut in zip(errors, donor_cuts, strict=True):
        g = ec[:, None]-ea
        easy = (ec > 0.) & (ec <= cut)
        harm = (-g).clip(min=0.)
        r = harm-.02*ec[:, None]
        # Same easy reference definition, applied only to observed-prefix errors.
        packed = np.stack((g, r, easy[:, None]*r, -easy[:, None].astype(float)*g,
                           np.broadcast_to(easy[:, None], g.shape)), axis=-1)
        feats.append(value.accumulate(packed, row, arm))
    comp = np.asarray(feats)
    g, r, er, net = (comp[..., i].mean(0)/scale[:, None] for i in range(4))
    support = (comp[..., 4] > 0.).mean(0)
    out = g-2.*r.clip(min=0.)-4.*er.clip(min=0.)-net.clip(min=0.)+.025*support
    out[:, 0] = 0.
    return out


def oracle_scores(pred, truth):
    variants = {"motion": value.score(pred), "gain_oracle": None, "risk_oracle": None,
                "support_oracle": None, "all_oracle_local": None}
    for name, cols in (("gain_oracle", (0, 1)), ("risk_oracle", (2, 3, 5)), ("support_oracle", (9,)),
                       ("all_oracle_local", (0, 1, 2, 3, 5, 9))):
        x = pred.copy()
        for col in cols:
            x[..., col] = truth[..., col] if col != 9 else np.where(truth[..., 9] > 0., 40., -40.)
        variants[name] = value.score(x)
    return variants


def fixed_count_witness(gain, risk, easy, legal, offset, easy_cap, mass, total, seconds=2.):
    """Maximin witness with the original risk and exact recording edit count."""
    if total == 0:
        return None, "zero_edits"
    q, c = np.nonzero(legal[:, 1:])
    c += 1
    e = easy[q, c]
    if not e.any() or mass <= 0.:
        return None, "no_legal_easy"
    n, nq = len(c), len(legal)
    g, r = gain[q, c].T, risk[q, c].T
    one = coo_array((np.ones(n), (q, np.arange(n))), shape=(nq, n+1))
    by_q = coo_array((c.astype(float), (q, np.arange(n))), shape=one.shape)
    scale = np.abs(r).max(1).clip(1.)
    head = vstack((one, csc_array(np.r_[c, 0.][None]), csc_array(np.column_stack((r/scale[:, None], np.zeros(3)))),
        csc_array(np.r_[e, 0.][None]), csc_array(np.column_stack((-g/mass, np.ones(2))))), format="csc")
    tail, upper = token_rows(by_q, legal[:, 1])
    mat = vstack((hstack((head, csc_array((head.shape[0], nq)))), tail), format="csc")
    lo = np.r_[np.full(nq, -np.inf), total, np.full(3, -np.inf), 1., -np.inf, -np.inf, np.full(2*nq, -np.inf)]
    hi = np.r_[np.ones(nq), total, 0., 0., easy_cap/scale[2], np.inf, np.array([0., offset])/mass, upper]
    sol = milp(c=np.r_[np.zeros(n), -1., np.zeros(nq)], integrality=np.r_[np.ones(n), np.zeros(nq+1)],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf, np.zeros(nq)], np.r_[np.ones(n), np.inf, np.full(nq, 10.)]),
        constraints=LinearConstraint(mat, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    if sol.x is not None:
        at = sol.x[:n] > .5
        counts = np.bincount(q[at], weights=c[at], minlength=nq).astype(np.int64)
        harm = r[:, at].sum(1)
        if (len(np.unique(q[at])) == at.sum() and counts.sum() == total and e[at].any()
                and respects_tokens(counts, legal[:, 1]) and (harm[:2] <= 0.).all() and harm[2] <= easy_cap
                and 100.*(g[:, at].sum(1)+[0., offset]).min()/mass > POSITIVE_PP):
            return counts, "positive_incumbent"
    if sol.status == 2:
        return None, "risk_infeasible"
    dual = getattr(sol, "mip_dual_bound", None)
    if dual is not None and -100.*dual <= POSITIVE_PP:
        return None, "no_strict_positive_bound"
    return None, "unresolved"
