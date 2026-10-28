"""Matched positive constraint rescaling; frozen original optimizer and gate."""
from contextlib import contextmanager
import threading

from src.world_model import m3w_constraint_scale as scaling

old = scaling.old
LOCK = threading.Lock()
MODES = ('unit_control', 'budget_scaled')


@contextmanager
def constraint_units(scales):
    if not LOCK.acquire(blocking=False):
        raise RuntimeError('Single-process sequential constraint context required')
    objective, update = old.decision.objective, old.decision.update_dual
    try:
        scaling.scaled_excess(old.torch.zeros(2), scales)
        def normalized_objective(result, dual):
            return objective({**result, 'risk_excess':scaling.scaled_excess(result['risk_excess'], scales)}, dual)
        def normalized_update(dual, excess, learning_rate, maximum):
            return update(dual, scaling.scaled_excess(excess, scales), learning_rate, maximum)
        old.decision.objective, old.decision.update_dual = normalized_objective, normalized_update
        yield
    finally:
        old.decision.objective, old.decision.update_dual = objective, update
        LOCK.release()


def fit(*args, mode, **kwargs):
    if mode not in MODES or 'arm' in kwargs:
        raise ValueError('Registered scaling arm; original risk objective only')
    budget = scaling.train_budget_scales(args[2], args[4], args[5], args[6], args[7]['scale'])
    scales = old.torch.from_numpy(budget) if mode == 'budget_scaled' else old.torch.ones(2)
    identity = {**kwargs['identity'], 'constraint_mode':mode,
                'constraint_scales':scales.tolist(), 'TRAIN_budget_scales':budget.tolist()}
    with constraint_units(scales):
        return old.fit(*args, **{**kwargs, 'identity':identity}, arm='task_risk')


def assert_control(state, prior):
    for key in ('settings','objective_config','preprocess','seed','step','initial_model','model',
                'optimizer','sampler_rng','torch_rng','dual','trace','draw_hash','row_draws',
                'queries','input_hashes','eligibility_sha256'):
        old.core.exact(state[key],prior[key])


def assert_matched(states):
    if tuple(s['identity']['constraint_mode'] for s in states) != MODES:
        raise ValueError('Both matched scaling arms required')
    for key in ('settings','objective_config','preprocess','seed','step','initial_model',
                'input_hashes','sampler_rng','draw_hash','row_draws','queries','eligibility_sha256'):
        old.core.exact(states[0][key],states[1][key])


predict = old.predict
