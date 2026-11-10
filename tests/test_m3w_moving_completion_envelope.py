import numpy as np
import pytest
from scipy.optimize import linprog

from scripts import audit_m3w_moving_completion_envelope as study


@pytest.mark.parametrize('radii', [[0., 0., 0.], [.1, 1., 8.], [2., 0., 7.], [30., 4., 70.]])
def test_mixed_query_envelopes_match_actor_level_linear_program(radii):
    rng = np.random.default_rng(17)
    cv, ols, target = rng.normal(size=(3, 8, 12, 2))
    query = np.array([2, 1, 3, 1, 2, 1, 1, 2])
    valid = rng.random(size=cv.shape[:2]) > .25
    valid[3] = False
    target[~valid] = np.nan
    result = study.aggregate(cv, ols, target, valid, query, radii)
    known = np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.)
    rows, weights = np.where(valid)[0], 1./np.bincount(query)[query]
    objective = np.r_[(~valid).sum(1)*weights, weights[rows]]
    min_constraints = np.c_[-np.eye(len(query))[rows], np.eye(len(rows))]
    bounds = [(0., None)]*len(query) + [(0., x) for x in known[valid]]
    queries = result['completion_queries']
    for scope in ('query', 'record'):
        local = scope == 'query'
        budget = radii if local else [sum(radii)]
        membership = np.eye(4)[query, 1:].T if local else np.ones((1, len(query)))
        spend = np.c_[membership, np.zeros((len(budget), len(rows))) ]
        lp = linprog(-objective, A_ub=np.r_[min_constraints, spend],
                     b_ub=np.r_[np.zeros(len(rows)), budget], bounds=bounds)
        assert lp.success
        answer = sum(study.favorable([q], q['budget']) for q in queries) if local else study.favorable(queries, sum(radii))
        assert answer == pytest.approx(-lp.fun, abs=1e-9)


@pytest.mark.parametrize('complete', [True, False])
def test_radius_formula_can_be_attained_in_relaxed_waypoint_space(complete):
    cv = np.zeros((1, 12, 2))
    ols = np.ones_like(cv)*[2., 0.]
    target = np.ones_like(cv)*[1.5, 0.]
    valid = np.ones((1, 12), dtype=bool)
    if not complete:
        valid[:, 8:] = False
    radius = 1.25
    row = study.aggregate(cv, ols, target, valid, [1], [radius])
    score = study.bounds(row, study.verify.evaluate(row))['query']
    pred = np.where(valid[..., None], [radius, 0.], [-radius, 0.])
    completed = np.where(valid[..., None], target, pred)
    gain = np.linalg.norm(ols-completed, axis=-1)-np.linalg.norm(pred-completed, axis=-1)
    assert score['favorable_completion'] == pytest.approx(100*gain.sum()/row['cv_error_mass'])
    if complete:
        assert list(score.values()) == pytest.approx([score['favorable_completion']]*3)
