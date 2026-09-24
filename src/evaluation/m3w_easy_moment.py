"""Joint easy-risk moments; estimated decisions are not a safety certificate."""
import numpy as np

from src.evaluation.m3w_protected_motion_controls import protected_decisions


def targets(costs, cv, distance, known, cutoff):
    y, b, d, k = map(np.asarray, (costs, cv, distance, known))
    if (y.shape != (len(d),2) or b.shape != d.shape or k.shape != d.shape
            or k.dtype != bool or not np.isfinite(cutoff) or cutoff <= 0
            or not np.isfinite(d).all() or (d<0).any()
            or not np.isfinite(y[k]).all() or not np.isfinite(b[k]).all()
            or (y[k]<0).any() or (b[k]<0).any()
            or (y[k].sum(1)>d[k]+2e-6*(1+d[k])).any()):
        raise ValueError('Aligned supported source costs and positive training cutoff required')
    easy = k & (b <= cutoff)
    harm = np.divide(np.where(k, y[:,1], 0), d, out=np.zeros(len(d)), where=d>0).clip(0,1)
    result = np.column_stack((easy*harm, np.where(easy,b/cutoff,0), easy, harm))
    result[~k] = 0
    return result


def decisions(fractions, distance, cutoff, score, history, rho=.02):
    f, d = np.asarray(fractions), np.asarray(distance)
    if (f.shape != (len(d),4) or not np.isfinite(f).all() or (f<0).any()
            or (f>1+1e-12).any() or (f[:,0]>f[:,3]+1e-12).any()
            or (f[:,1]>f[:,2]+1e-12).any() or cutoff<=0
            or not np.isfinite(cutoff) or not np.isfinite(rho) or not 0<=rho<=1):
        raise ValueError('Finite bounded easy moments with consistent support required')
    old = protected_decisions(score, history, d)
    q, r, product = f[:,0]*d, f[:,1]*cutoff, f[:,2]*f[:,3]*d
    supported = old['net_stop'] & (r>0)
    return dict(**old, joint_easy_moment=supported & (q<=rho*r),
                product_easy_marginals=supported & (product<=rho*r))
