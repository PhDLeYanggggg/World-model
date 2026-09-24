"""Strict original-unit and provenance guards for the registered net-risk study."""
import itertools
import math
from pathlib import Path

import numpy as np

from src.world_model.m3w_net_easy_risk import allocate as original_allocate


def allocate(gain, risk, eligible, budget, *, count=None, seconds=5.):
    bits, report = original_allocate(gain, risk, eligible, budget, count=count, seconds=seconds)
    q = np.asarray(risk, float); g = np.asarray(gain, float); ok = np.asarray(eligible)
    total = math.fsum(q[bits])
    # The allowance is not enlarged by unselected or large cancelling risks.
    # Treat stored floating-point risks as the original problem coefficients.
    if total <= budget:
        return bits, dict(report, predicted_risk=total, constraint_pass=True,
                          original_unit_check='fsum_no_allowance_relaxation')
    rows = np.flatnonzero(ok)
    if len(rows) <= 18:
        best = None; best_gain = -np.inf
        for v in itertools.product([False, True], repeat=len(rows)):
            b = np.array(v, bool)
            if count is not None and b.sum() != count:
                continue
            if math.fsum(q[rows[b]]) <= budget:
                score = math.fsum(g[rows[b]])
                if score > best_gain:
                    best, best_gain = b, score
        if best is not None:
            bits[:] = False; bits[rows] = best
            return bits, dict(status='exhaustive_original_units', optimal=True,
                selected=int(bits.sum()), predicted_risk=math.fsum(q[bits]), budget=float(budget),
                constraint_pass=True, exact_count_pass=count is None or bool(bits.sum() == count),
                numerical=None, original_solver_status=report['status'],
                original_unit_check='fsum_no_allowance_relaxation')
    bits[:] = False
    return bits, dict(status='original_units_rejected_floor', optimal=False, selected=0,
        predicted_risk=0., budget=float(budget), constraint_pass=True,
        exact_count_pass=count is None or count == 0, numerical=report['numerical'],
        original_unit_check='fsum_no_allowance_relaxation')


def validate_receipt(receipt, view, action, experiment_hash, frozen_head, checkpoint):
    identity = receipt['identity']
    if (identity['view'] != view or identity['action'] != action
            or identity['experiment_sha256'] != experiment_hash
            or identity['frozen_head_sha256'] != frozen_head['checkpoint_sha256']
            or checkpoint['identity'] != identity
            or checkpoint['seed'] != int(view.rsplit('_seed', 1)[1])
            or view.rsplit('_seed', 1)[0] in checkpoint['preprocess']['training_sites']):
        raise ValueError('Wrong source-excluded action/view/seed/checkpoint identity')


def require_replay_outputs(public, root, actions, seeds):
    paths = [Path(public)/'analysis.json']
    paths += [Path(root)/'outcomes'/f'{a}_seed{s}.npz' for a in actions for s in seeds]
    if not all(p.is_file() for p in paths):
        raise ValueError('Replay requires an existing original analysis and every outcome archive')
