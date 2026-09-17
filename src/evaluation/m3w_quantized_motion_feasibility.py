"""Label-side CV feasibility conditional on supplied-H integer-pixel rounding."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def quantized_cv_feasibility(native_xy, step_times, homography, half_width=.501):
    """Four-variable LP in dataset-local coordinates, never an inference feature.

    A closed half-pixel box is an optimistic quantization model, not verified
    annotation uncertainty. Future observations constrain this diagnostic oracle.
    """
    xy, times, h = np.asarray(native_xy, float), np.asarray(step_times, float), np.asarray(homography, float)
    if (xy.ndim!=2 or xy.shape[1]!=2 or times.shape!=(len(xy),) or len(xy)<2
            or not np.isfinite(xy).all() or not np.isfinite(times).all()
            or not np.all(np.diff(times)>0) or h.shape!=(3,3) or not np.isfinite(h).all()
            or half_width<.5):
        raise ValueError('Finite aligned native positions, increasing steps and supplied H required')
    inverse = np.linalg.inv(h)
    homogeneous = np.c_[xy,np.ones(len(xy))] @ inverse.T
    w = homogeneous[:,2]
    if np.any(np.abs(w)<=1e-10) or len(np.unique(np.sign(w)))!=1:
        return {'status':'not_run_projective_chart_sign_or_pole','feasible':None}
    inverse = inverse*np.sign(w[0])
    pixels = homogeneous[:,:2]/w[:,None]
    rounded = np.round(pixels)
    if np.max(np.abs(pixels-rounded))>.001:
        return {'status':'not_run_no_integer_pixel_lineage','feasible':None}
    constraints, bounds = [], []
    for t, observed in zip(times,rounded):
        for axis in (0,1):
            lower,upper = observed[axis]-half_width,observed[axis]+half_width
            for row in (inverse[axis]-upper*inverse[2],lower*inverse[2]-inverse[axis]):
                constraints.append([row[0],row[1],t*row[0],t*row[1]])
                bounds.append(-row[2])
        row = inverse[2]
        constraints.append([-row[0],-row[1],-t*row[0],-t*row[1]])
        bounds.append(row[2]-1e-8)
    a,b = np.asarray(constraints),np.asarray(bounds)
    rescale = np.maximum(np.maximum(np.max(np.abs(a),axis=1),np.abs(b)),1.)
    solution = linprog(np.zeros(4),A_ub=a/rescale[:,None],b_ub=b/rescale,
                       bounds=[(None,None)]*4,method='highs')
    if solution.status==2:
        return {'status':'solver_infeasible_conditional_rounding_model','feasible':False,
                'solver_status':int(solution.status),'physical_motion_proven':False}
    if not solution.success:
        return {'status':'not_run_solver_inconclusive','feasible':None,'solver_status':int(solution.status)}
    fitted = solution.x[:2]+times[:,None]*solution.x[2:]
    fitted_h = np.c_[fitted,np.ones(len(fitted))] @ inverse.T
    if np.any(fitted_h[:,2]<=0):
        return {'status':'not_run_numerical_chart_violation','feasible':None}
    fitted_pixels=fitted_h[:,:2]/fitted_h[:,2:3]
    violation=float(np.max(np.abs(fitted_pixels-rounded)-half_width))
    if violation>1e-5:
        return {'status':'not_run_numerical_reprojection_violation','feasible':None,
                'max_pixel_constraint_excess':violation}
    return {'status':'feasible_conditional_rounding_model','feasible':True,'solver_status':0,
            'max_pixel_constraint_excess':max(violation,0.),'physical_motion_proven':False}
