"""Fixed-support attenuation diagnostics, not outcome-selected deployment."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import fraction


def attenuate(floor, proposal, amplitude):
    b, p = np.asarray(floor, float), np.asarray(proposal, float)
    if (b.ndim != 3 or b.shape[-1] != 2 or p.shape != b.shape
            or not all(np.isfinite(v).all() for v in (b, p))
            or not np.isscalar(amplitude) or not np.isfinite(amplitude)
            or not 0 <= amplitude <= 1):
        raise ValueError('Finite paired trajectories and a fixed amplitude in [0,1] required')
    if amplitude == 0:
        return b.copy()
    if amplitude == 1:
        return p.copy()
    return b + amplitude * (p-b)


def risk_diagnosis(prediction, floor, target, valid, recordings, *, easy, baseline_cost):
    """Expose support and unknown bounds; never convert undefined ratios to zero."""
    from src.evaluation.m3w_selection_forecast_readout import errors
    p, b, mask, rec = map(np.asarray, (prediction, floor, valid, recordings))
    pe, _ = errors(p, target, mask)
    be = np.asarray(baseline_cost)
    known = mask.any(1)
    if (be.shape != (len(p),) or np.asarray(easy).shape != (len(p),)
            or not np.array_equal(np.isfinite(be), known)):
        raise ValueError('Identical floor/label support required')
    disagreement = np.linalg.norm(p-b, axis=-1).max(1)
    selected = disagreement > 0
    harm = np.where(known, np.maximum(pe-be, 0), 0)
    rows = {}
    for r in sorted(set(rec)):
        at = rec == r
        take = at & selected & known
        unknown = at & selected & ~known
        mass = float(be[take].sum())
        emass = float(be[take & easy].sum())
        radius = float(disagreement[unknown].sum())
        h = float(harm[take].sum())
        eh = float(harm[take & easy].sum())
        rows[str(r)] = dict(
            selected_known_reference_mass=mass, selected_known_easy_reference_mass=emass,
            known_harm_mass=h, known_easy_harm_mass=eh,
            unknown_disagreement_radius=radius,
            known_all_risk=fraction(h, mass), known_easy_risk=fraction(eh, emass),
            completion_all_risk_upper=fraction(h+radius, mass),
            completion_easy_risk_upper=fraction(eh+radius, emass),
            selected_rows=int((at & selected).sum()),
            unknown_selected=int(unknown.sum()),
            known_all_pass_but_completion_fails=bool(mass > 0 and h/mass <= .02 < (h+radius)/mass),
            known_easy_pass_but_completion_fails=bool(emass > 0 and eh/emass <= .02 < (eh+radius)/emass))
    return rows


def aggregate_risk(groups, family, amplitude):
    key = str(amplitude)
    scored = [g['policies'][family][key] for g in groups]
    views = [v for g in scored for v in g['recordings'].values()]
    details = [v for g in groups for v in g['diagnosis'][family][key].values()]
    easy_net = [v['known_easy_net_degradation'] for v in views
                if v['known_easy_net_degradation'] is not None]
    defined = lambda field: [v[field] for v in views if v[field] is not None]
    return dict(
        required_record_views=len(views), supported_record_views=sum(v['supported'] for v in views),
        known_all_violations=sum(v > .02 for v in defined('known_all_risk')),
        known_easy_violations=sum(v > .02 for v in defined('known_easy_risk')),
        completion_all_violations=sum(v > .02 for v in defined('completion_all_risk_upper')),
        completion_easy_violations=sum(v > .02 for v in defined('completion_easy_risk_upper')),
        undefined_all_views=sum(v['completion_all_risk_upper'] is None for v in views),
        undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
        undefined_easy_net_views=sum(v['known_easy_net_degradation'] is None for v in views),
        worst_signed_easy_net_percent=100*max(easy_net) if easy_net else None,
        worst_known_all_risk_percent=100*max(defined('known_all_risk')) if defined('known_all_risk') else None,
        worst_known_easy_risk_percent=100*max(defined('known_easy_risk')) if defined('known_easy_risk') else None,
        all_failures_due_to_unknown_radius=sum(v['known_all_pass_but_completion_fails'] for v in details),
        easy_failures_due_to_unknown_radius=sum(v['known_easy_pass_but_completion_fails'] for v in details),
        full_contract_pass=bool(views and all(v['supported'] for v in views)),
        independent_confirmation=False)


def locality_interval(groups, getter, roster, resamples, seed):
    if {g['site'] for g in groups} != set(roster):
        raise ValueError('Every fixed SOURCE locality must remain')
    by_site = {}
    for site in sorted(roster):
        rows = [r for r in groups if r['site'] == site]
        if (len(rows) != 6 or len({(r['trial'], r['seed']) for r in rows}) != 6
                or {r['seed'] for r in rows} != {17, 29, 43}):
            raise ValueError('Two excluded producer folds and three actual seeds required')
        values = [getter(r) for r in rows]
        by_site[site] = (None if any(v is None for v in values) else float(np.mean(values)))
    if any(v is None for v in by_site.values()):
        return dict(point=None, CI95=None, by_locality=by_site, status='required_view_not_estimable')
    x = np.array(list(by_site.values()))
    if not np.isfinite(x).all():
        raise ValueError('Finite values or explicit missing support required')
    draws = np.random.default_rng(seed).choice(x, (resamples, len(x))).mean(1)
    return dict(point=float(x.mean()), CI95=np.quantile(draws, [.025, .975]).tolist(),
                by_locality=by_site, resamples=resamples, seed=seed,
                status='exposed_SOURCE_crossfit_not_independent_confirmation')
