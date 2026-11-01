"""Same-count seven-action allocation with a causal predicted-path pair cost."""
from collections import Counter
import hashlib

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from src.world_model import m3w_frame_action_allocation as base

NAMES = ('unary_joint', 'pair_joint')


def geometry(rollouts, floor, origins, scales, ids, *, neighbors):
    r, f, o, s, ids = map(np.asarray, (rollouts, floor, origins, scales, ids))
    n = len(ids)
    if (r.shape != (n, 7, 12, 2) or f.shape != (n, 12, 2) or o.shape != (n, 2)
            or s.shape != (n,) or len(np.unique(ids)) != n or not n
            or any(not np.isfinite(v).all() for v in (r, f, o, s)) or (s <= 0).any()
            or not isinstance(neighbors, int) or isinstance(neighbors, bool) or neighbors < 1):
        raise ValueError('Aligned finite past-only geometry,positive scales and neighbor count required')
    paths = np.concatenate((f[:, None], r), 1).astype(float)+o[:, None, None]
    distance = np.linalg.norm(o[:, None]-o[None], axis=-1)
    edges = set()
    for i in range(n):
        order = np.lexsort((ids, distance[i]))
        for j in order[order != i][:neighbors]: edges.add(tuple(sorted((i, int(j)))))
    edges = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
    table = np.empty((len(edges), 8, 8), float)
    for e, (i, j) in enumerate(edges):
        reference = np.linalg.norm(paths[i, 0]-paths[j, 0], axis=-1)
        actual = np.linalg.norm(paths[i, :, None]-paths[j, None, :], axis=-1)
        table[e] = np.maximum(reference[None, None]-actual, 0).mean(-1)/np.sqrt(s[i]*s[j])
    return edges, table


def pair_value(edges, table, choice):
    c = np.asarray(choice)+1
    return float(table[np.arange(len(edges)), c[edges[:, 0]], c[edges[:, 1]]].mean()) if len(edges) else 0.


def solve(prediction, allowed, parent, edges, table, scales, *, weight, seconds):
    p, ok = base.validate(prediction, allowed)
    parent, scales = np.asarray(parent), np.asarray(scales)
    if (not base.feasible(p, ok, parent) or scales.shape != (len(p),)
            or not np.isfinite(scales).all() or (scales <= 0).any()
            or not np.isfinite(weight) or weight < 0 or not np.isfinite(seconds) or seconds <= 0
            or table.shape != (len(edges), 8, 8) or not np.isfinite(table).all() or (table < 0).any()
            or edges.shape != (len(edges), 2) or edges.dtype.kind not in 'iu'
            or (len(edges) and ((edges < 0).any() or (edges >= len(p)).any()
                or (edges[:, 0] >= edges[:, 1]).any() or len(np.unique(edges, axis=0)) != len(edges)))
            or (table[:, 0, 0] != 0).any()):
        raise ValueError('Feasible frozen parent and valid causal pair costs required')
    scale = float(scales.sum()); count = int((parent >= 0).sum())

    def objective(c):
        return -base.terms(p, c)[0]/scale+weight*pair_value(edges, table, c)

    initial = objective(parent)
    def result(c, reason, optimal, status=None):
        if not base.feasible(p, ok, c) or int((c >= 0).sum()) != count:
            raise AssertionError('Original budgets/count must survive every return path')
        return c.copy(), dict(reason=reason, optimal=optimal, solver_status=status,
            count=count, pair_before=pair_value(edges, table, parent), pair_after=pair_value(edges, table, c),
            objective_before=initial, objective_after=objective(c),
            changed_rows=int((c != parent).sum()), edges=len(edges), labels_used=False)

    if count == 0 or not len(edges) or weight == 0:
        return result(parent, 'unchanged_count_or_graph_or_zero_weight', False)
    if pair_value(edges, table, parent) == 0:
        return result(parent, 'zero_parent_pair_cost_cached_unary_control', False)
    legal = base.positive_actions(p, ok)
    options = [np.r_[-1, np.flatnonzero(row)] for row in legal]
    offsets = np.r_[0, np.cumsum([len(v) for v in options])]
    node_n = int(offsets[-1]); costs = []; row = []; col = []; val = []; lower = []; upper = []

    def constraint(columns, values, low, high):
        row.extend([len(lower)]*len(columns)); col.extend(columns); val.extend(values)
        lower.append(low); upper.append(high)

    for i, choices in enumerate(options):
        columns = np.arange(offsets[i], offsets[i+1])
        gain = np.r_[0., p[i, choices[1:], 0]-p[i, choices[1:], 1]]
        costs.extend((-gain/scale).tolist()); constraint(columns, np.ones(len(columns)), 1., 1.)
    columns = np.arange(node_n)
    switched = np.concatenate([v >= 0 for v in options]).astype(float)
    constraint(columns, switched, count, count)
    for h, r in ((1, 2), (4, 3)):
        risk = np.concatenate([np.r_[0., p[i, choices[1:], h]-.02*p[i, choices[1:], r]]
            for i, choices in enumerate(options)])
        normalizer = max(float(np.abs(risk).max()), 1e-300)
        constraint(columns, risk/normalizer, -np.inf, 0.)
    for e, (i, j) in enumerate(edges):
        ni, nj = len(options[i]), len(options[j]); start = len(costs)
        grid = np.arange(start, start+ni*nj).reshape(ni, nj)
        costs.extend((weight*table[e][np.ix_(options[i]+1, options[j]+1)]/len(edges)).ravel().tolist())
        for a in range(ni):
            constraint(np.r_[grid[a], offsets[i]+a], np.r_[np.ones(nj), -1.], 0., 0.)
        for b in range(nj):
            constraint(np.r_[grid[:, b], offsets[j]+b], np.r_[np.ones(ni), -1.], 0., 0.)
    matrix = coo_matrix((val, (row, col)), shape=(len(lower), len(costs))).tocsc()
    fit = milp(c=np.array(costs), integrality=np.r_[np.ones(node_n), np.zeros(len(costs)-node_n)],
        bounds=Bounds(np.zeros(len(costs)), np.ones(len(costs))),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options=dict(presolve=True, mip_rel_gap=0., time_limit=seconds))
    if fit.status != 0 or fit.x is None:
        return result(parent, f'solver_status_{fit.status}_parent_fallback', False, int(fit.status))
    chosen = np.array([v[np.argmax(fit.x[offsets[i]:offsets[i+1]])] for i, v in enumerate(options)], np.int64)
    binary = fit.x[:node_n]
    if (not np.allclose(binary, binary > .5, rtol=0, atol=1e-7) or not base.feasible(p, ok, chosen)
            or int((chosen >= 0).sum()) != count or objective(chosen) > initial+1e-10
            or not np.isclose(objective(chosen), fit.fun, rtol=1e-7, atol=1e-9)):
        return result(parent, 'numerical_postcheck_parent_fallback', False, 0)
    return result(chosen, 'optimal_pair_objective', True, 0)


def allocate(prediction, allowed, recordings, frames, ids, rollouts, floor, origins, scales, parent,
             *, weight, neighbors, seconds, checkpoint=lambda _: None, heartbeat=lambda **_: None, resume_state=None):
    p, ok = base.validate(prediction, allowed); parent = np.asarray(parent)
    groups = base.query_groups(recordings, frames, ids)
    binds = {k: hashlib.sha256(str((np.asarray(v).shape, np.asarray(v).dtype.str)).encode()
        +np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in
        zip(('prediction','allowed','recordings','frames','ids','rollouts','floor','origins','scales','parent'),
            (p,ok,recordings,frames,ids,rollouts,floor,origins,scales,parent))}
    binds.update(weight=weight, neighbors=neighbors, seconds=seconds, version=1)
    if parent.shape != (len(ids),): raise ValueError('Aligned frozen full-policy control required')
    out = parent.copy(); start = 0; reasons = Counter(); stats = dict(changed_queries=0, changed_rows=0,
        graph_edges=0, graph_pair_mass_before=0., graph_pair_mass_after=0., nonoptimal_solver_queries=0, solver_calls=0)
    if resume_state is not None:
        if resume_state['bindings'] != binds: raise ValueError('Frozen decoder inputs changed on resume')
        start = resume_state['completed_queries']; out = np.asarray(resume_state['choice']).copy()
        if not 0 <= start <= len(groups) or out.shape != parent.shape: raise ValueError('Invalid resume cursor')
        reasons = Counter(resume_state['reasons']); stats = dict(resume_state['stats'])
        for g in groups[:start]:
            if not base.feasible(p[g], ok[g], out[g]) or (out[g] >= 0).sum() != (parent[g] >= 0).sum():
                raise ValueError('Invalid processed query in resume')
        np.testing.assert_array_equal(out[np.concatenate(groups[start:])] if start < len(groups) else out[:0],
            parent[np.concatenate(groups[start:])] if start < len(groups) else parent[:0])
    for index in range(start, len(groups)):
        g = groups[index]
        edges, table = geometry(rollouts[g], floor[g], origins[g], scales[g], ids[g], neighbors=neighbors)
        chosen, info = solve(p[g], ok[g], parent[g], edges, table, scales[g], weight=weight, seconds=seconds)
        out[g] = chosen; reasons[info['reason']] += 1
        stats['changed_queries'] += int(info['changed_rows'] > 0); stats['changed_rows'] += info['changed_rows']
        stats['graph_edges'] += len(edges)
        for side in ('before','after'): stats['graph_pair_mass_'+side] += info['pair_'+side]*len(edges)
        stats['solver_calls'] += int(info['solver_status'] is not None)
        stats['nonoptimal_solver_queries'] += int(info['solver_status'] is not None and not info['optimal'])
        if index == 0 or (index+1) % 100 == 0 or index+1 == len(groups):
            checkpoint(dict(bindings=binds, completed_queries=index+1, choice=out.copy(), reasons=dict(reasons), stats=stats))
            heartbeat(queries=index+1, total_queries=len(groups), **stats)
    choices = dict(unary_joint=parent.copy(), pair_joint=out); ranks = {}
    for name, c in choices.items():
        at = np.flatnonzero(c >= 0); rank = np.zeros(len(ids)); rank[at] = p[at, c[at], 0]-p[at, c[at], 1]; ranks[name] = rank
    return choices, ranks, dict(queries=len(groups), reason_counts=dict(reasons), **stats,
        same_count_every_query=True, labels_used=False, reference_and_risk_unchanged=True,
        all_visible_agents_claim=False, global_optimality_claim=False, new_neural_training=False)
