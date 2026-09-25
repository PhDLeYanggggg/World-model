"""Source-held calibration of aligned policy pairs; no risk certificate."""
import numpy as np
from src.world_model.m3w_dual_event_bridge import choices

RULES = ('reference', 'none', 'population_rescale', 'selected_risk_grid')


def apply(utility, moments, moving, envelope, rule, budget=.02):
    raw = choices(utility, moments, moments, moving, envelope, arm='all_risk_only')
    kind = rule['kind']
    if kind not in RULES or budget != .02: raise ValueError('Unregistered calibration rule or budget')
    if rule['abstain'] or kind == 'reference': return np.zeros(len(raw), bool)
    if kind == 'none': return raw
    if kind == 'selected_risk_grid':
        tau = rule['threshold']
        if not np.isfinite(tau) or not 0 <= tau <= budget:
            raise ValueError('Threshold cannot relax the registered budget')
    else:
        h, d = rule['harm_multiplier'], rule['denominator_multiplier']
        if not np.isfinite([h, d]).all() or h < 1 or not 0 < d <= 1:
            raise ValueError('Only conservative finite rescaling is allowed')
        tau = budget*d/h
    u, m = np.asarray(utility), np.asarray(moments)
    return moving & (envelope > 0) & (u[:, 0] > u[:, 1]) & (m[:, 1] <= tau*m[:, 0])


def evidence(bits, cv, reference, candidate, sites, *, easy_cut, budget=.02):
    cv, r, p, s, bits = map(np.asarray, (cv, reference, candidate, sites, bits))
    if (r.ndim != 1 or any(x.shape != r.shape for x in (cv, p, s, bits)) or bits.dtype != bool
            or not np.isfinite(easy_cut) or easy_cut <= 0 or budget != .02
            or any(np.isinf(x).any() or (x[np.isfinite(x)] < 0).any() for x in (cv, r, p))
            or not np.array_equal(np.isnan(cv), np.isnan(r)) or not np.array_equal(np.isnan(r), np.isnan(p))):
        raise ValueError('Aligned labels, fixed CV event and nonnegative costs required')
    known = np.isfinite(r); ade = np.where(bits, p, r); rows, gains, feasible = {}, [], True
    for site in sorted(set(s)):
        pop = s == site; use = pop & known; easy = use & (cv > 0) & (cv <= easy_cut)
        events = {}
        for event, mask in (('all', use), ('easy', easy)):
            den = float(r[mask].sum()); harm = float(np.maximum(ade[mask]-r[mask], 0).sum())
            events[event] = dict(rows=int(mask.sum()), reference_mass=den, positive_harm=harm,
                ratio=harm/den if den > 0 else None, supported=bool(mask.any() and den > 0),
                passes=bool(mask.any() and den > 0 and harm <= budget*den))
        cmass = float(cv[easy].sum()); amass = float(ade[easy].sum())
        easy_degradation = 100*(amass/cmass-1) if cmass > 0 else None
        zero_harm = int((use & (cv == 0) & (ade > 0)).sum())
        refmass = float(r[use].sum())
        gain = 100*(1-float(ade[use].sum())/refmass) if refmass > 0 else None
        good = (all(v['passes'] for v in events.values()) and easy_degradation is not None
                and easy_degradation <= 2 and zero_harm == 0)
        rows[str(site)] = dict(events=events, easy_degradation_percent=easy_degradation,
            zero_CV_harm=zero_harm, gain_percent=gain, feasible=bool(good),
            selected=int(bits[pop].sum()), unknown_selected=int((bits & pop & ~known).sum()))
        feasible &= good
        if gain is not None: gains.append(gain)
    return dict(by_locality=rows, feasible=bool(feasible and len(rows) > 0),
        equal_locality_gain_percent=float(np.mean(gains)) if len(gains) == len(rows) and rows else None,
        selected=int(bits.sum()), unknown_selected=int((bits & ~known).sum()),
        conditional_event='producer_training_positive_CV_easy', risk_reference='same_delivered_R_policy',
        calibrated_guarantee=False)


def fit(utility, moments, moving, envelope, cv, reference, candidate, sites, *, easy_cut, grid):
    if sorted(set(grid)) != list(grid) or grid[0] != 0 or grid[-1] != .02:
        raise ValueError('Fixed complete threshold grid required')
    u, m, cv, r, p, s = map(np.asarray, (utility, moments, cv, reference, candidate, sites))
    known = np.isfinite(r)
    def ev(rule):
        return evidence(apply(u, m, moving, envelope, rule), cv, r, p, s, easy_cut=easy_cut)
    raw = dict(kind='none', abstain=False)
    # Validate label support before fitting any moment correction.
    raw_evidence = ev(raw)
    pm, tm = [], []
    for site in sorted(set(s)):
        use = (s == site) & known
        if not use.any():
            pm.append([0., 0.]); tm.append([0., 0.]); continue
        pm.append(m[use].mean(0)); tm.append([r[use].mean(), np.maximum(p[use]-r[use], 0).mean()])
    ph, th = np.mean(pm, axis=0), np.mean(tm, axis=0)
    unsupported = ph[0] <= 0 or th[0] <= 0 or (ph[1] <= 0 and th[1] > 0)
    scale = dict(kind='population_rescale', abstain=bool(unsupported),
        harm_multiplier=max(1., float(th[1]/ph[1])) if ph[1] > 0 else 1.,
        denominator_multiplier=min(1., float(th[0]/ph[0])) if ph[0] > 0 and th[0] > 0 else 1.)
    fallback = dict(kind='selected_risk_grid', abstain=True, threshold=None)
    best, rank, records = fallback, (0., 0, 0.), []
    for tau in grid:
        rule = dict(kind='selected_risk_grid', abstain=False, threshold=float(tau)); e = ev(rule)
        records.append(dict(threshold=float(tau), **e))
        if e['feasible']:
            candidate_rank = (e['equal_locality_gain_percent'], -e['selected'], -float(tau))
            if candidate_rank > rank: best, rank = rule, candidate_rank
    rules = dict(reference=dict(kind='reference', abstain=True), none=raw,
                 population_rescale=scale, selected_risk_grid=best)
    return dict(rules=rules, evidence={k: ev(rule) for k, rule in rules.items()}, grid_evidence=records,
        predicted_equal_locality_moments=ph.tolist(), actual_equal_locality_moments=th.tolist(),
        fit_localities=sorted(set(s)), calibrated_guarantee=False,
        role='opened_source_C_excluded_from_forecast_and_scoring_fit')


def zero_violation_scene_upper(n, delta):
    """Exact zero-event binomial bound, NOT a bound on ADE harm ratios."""
    if int(n) != n or n < 1 or not 0 < delta < 1: raise ValueError('Positive count and error probability required')
    return float(-np.expm1(np.log(delta)/n))


def hb_zero_loss_floor(n, alpha):
    """Most optimistic HB p-value for a hypothetical [0,1] loss at mean zero."""
    if int(n) != n or n < 1 or not 0 < alpha < 1: raise ValueError('Invalid hypothetical bounded loss setting')
    return float(np.exp(n*np.log1p(-alpha)))
