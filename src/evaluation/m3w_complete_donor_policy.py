"""Complete-population policies, with the original risk and support definition."""
import numpy as np

from src.evaluation import m3w_selection_forecast_readout as original
from src.world_model.m3w_source_separated_critics import matched_self


def veto(proposal, risk):
    own, p = np.asarray(proposal), np.asarray(risk)
    if own.ndim!=1 or own.dtype.kind not in 'iu' or p.shape!=(len(own),2) or not np.isfinite(p).all():
        raise ValueError('Aligned causal proposal and finite signed-risk scores only')
    if np.any((own < -1) | (own > 6)):
        raise ValueError('Frozen seven-action proposal namespace required')
    return np.where((own>=0)&(p<=0).all(1), own, -1).astype(np.int64)


def paths_for(paths, floor, choice):
    p, b, c = np.asarray(paths), np.asarray(floor), np.asarray(choice)
    if (p.shape!=(len(c),7,12,2) or b.shape!=(len(c),12,2) or c.dtype.kind not in 'iu'
            or np.any((c < -1)|(c > 6)) or not np.isfinite(p).all() or not np.isfinite(b).all()):
        raise ValueError('Finite frozen paths and valid causal actions required')
    result = b.copy()
    take = np.flatnonzero(c>=0)
    result[take] = p[take,c[take]]
    return result


def safety(policies):
    records = [v for p in policies for v in p['recordings'].values()]
    def violations(key):
        return sum(v[key] is not None and v[key]>.02 for v in records)
    return dict(required_record_views=len(records), supported_record_views=sum(v['supported'] for v in records),
        undefined_all_views=sum(v['completion_all_risk_upper'] is None for v in records),
        undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in records),
        known_all_violations=violations('known_all_risk'), known_easy_violations=violations('known_easy_risk'),
        completion_all_violations=violations('completion_all_risk_upper'),
        completion_easy_violations=violations('completion_easy_risk_upper'),
        easy_net_violations=violations('known_easy_net_degradation'),
        original_record_contract_pass=bool(records and all(v['supported'] for v in records)),
        identity_certification=False, empty_support_is_NA=True)


def matched(proposal, utility, selected, groups, ids):
    return matched_self(proposal, utility, selected, groups, ids)
