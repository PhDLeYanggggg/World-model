"""Paired signed-score accuracy and identical fixed-screen diagnostics."""
import numpy as np
from src.world_model.m3w_risk_excess import excess


def metric(new_score, old_score, y, candidate_error, constant, scale, easy_cut, hard_cut):
    new_score, old_score, y, error = map(lambda x: np.asarray(x, float),
                                        (new_score, old_score, y, candidate_error))
    if (new_score.ndim != 1 or old_score.shape != new_score.shape or error.shape != new_score.shape
            or y.shape != (len(error), 2) or not np.isfinite(new_score).all() or not np.isfinite(old_score).all()
            or not np.isfinite(scale) or scale <= 0 or not np.isfinite(constant)
            or not np.array_equal(np.isfinite(y).all(1), np.isfinite(error))):
        raise ValueError('Aligned finite causal scores and paired evaluation-only labels required')
    known = np.isfinite(y).all(1)
    if not known.any(): raise ValueError('Known locality support required')
    new_score, old_score, y, error = new_score[known], old_score[known], y[known], error[known]
    truth = excess(y); cv = y[:, 0]
    a, b, c = (float(np.mean((p-truth)**2)) for p in (new_score, old_score, constant))
    result = dict(rows=int(known.sum()), unknown=int((~known).sum()),
        new_MSE_scaled=a/scale**2, control_MSE_scaled=b/scale**2,
        MSE_gain_vs_control_percent=100*(1-a/b) if b > 0 else None,
        new_MSE_skill_vs_constant_percent=100*(1-a/c) if c > 0 else None,
        control_MSE_skill_vs_constant_percent=100*(1-b/c) if c > 0 else None)
    for name, p in [('new', new_score), ('control', old_score)]:
        take = p <= 0; selected = np.where(take, error, cv)
        result[name+'_screen_rate'] = float(take.mean())
        result[name+'_screen_positive_harm_ratio'] = float(y[take, 1].sum()/cv[take].sum()) if cv[take].sum() > 0 else None
        result[name+'_screen_actual_excess_mean_scaled'] = float(truth[take].mean()/scale) if take.any() else None
        result[name+'_screen_predicted_excess_mean_scaled'] = float(p[take].mean()/scale) if take.any() else None
        result[name+'_zero_reference_harmed'] = int((take & (cv == 0) & (error > 0)).sum())
        result[name+'_zero_reference_harm'] = float(error[take & (cv == 0)].sum())
        for subset, mask in dict(all=np.ones(len(cv), bool), easy=(cv > 0) & (cv <= easy_cut), hard=cv >= hard_cut).items():
            result[name+'_'+subset+'_ADE_gain_vs_CV_percent'] = float(100*(1-selected[mask].sum()/cv[mask].sum())) if cv[mask].sum() > 0 else None
    result['screen_rate_change'] = result['new_screen_rate']-result['control_screen_rate']
    result['screen_ADE_gain_change_points'] = result['new_all_ADE_gain_vs_CV_percent']-result['control_all_ADE_gain_vs_CV_percent'] if cv.sum() > 0 else None
    return result
