"""Two-sided budget-boundary supervision of coherent costs, not expert classes."""
from contextlib import contextmanager
from src.world_model import m3w_own_decision_cost as parent

np, torch = parent.np, parent.torch
ARMS = ('moment', 'boundary')
EPSILON = 1e-6
ORIGINAL_LOSS = parent.losses


def boundary_loss(prediction, target, legal, segments, count):
    if (prediction.shape != target.shape or prediction.shape[1:] != (7, 5)
            or legal.shape != prediction.shape[:2] or legal.dtype != torch.bool
            or not torch.isfinite(prediction).all() or (prediction < 0).any()
            or not torch.isfinite(target).all() or (target < 0).any()):
        raise ValueError('Finite costs and a separate known/legal loss mask required')
    # References retain their original regression loss; this auxiliary cannot
    # directly win by increasing a predicted denominator.
    harm = prediction[:, :, [1, 4]]
    budget = .02 * prediction[:, :, [2, 3]].detach()
    logits = torch.log(harm + EPSILON) - torch.log(budget + EPSILON)
    labels = (target[:, :, [1, 4]] > .02 * target[:, :, [2, 3]]).to(prediction.dtype)
    error = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none').mean(2)
    row = (error * legal).sum(1) / legal.sum(1).clamp_min(1)
    total = prediction.new_zeros(count).index_add(0, segments, row)
    support = prediction.new_zeros(count).index_add(0, segments, legal.any(1).double())
    value = (total / support.clamp_min(1)).mean()
    return value, dict(boundary_known_actions=int(legal.sum()),
        boundary_positive_actions_per_channel=[int((labels[:, :, k].bool() & legal).sum()) for k in range(2)])


@contextmanager
def installed(arm):
    if arm not in ARMS:
        raise ValueError('Registered moment or boundary arm required')
    before = parent.losses
    if before is not ORIGINAL_LOSS:
        raise ValueError('Nested objective mutation is not supported')

    def loss(prediction, data, rows, segments, count, normalizer, choices):
        terms = ORIGINAL_LOSS(prediction, data, rows, segments, count, normalizer, choices)
        c = data['causal']
        known = torch.from_numpy(data['known'][rows])
        target = torch.from_numpy(np.where(data['known'][rows, None, None],
            data['targets'][rows] / c['scales'][rows, None, None], 0.))
        legal = torch.from_numpy(c['allowed'][rows]) & known[:, None]
        auxiliary, info = boundary_loss(prediction, target, legal, segments, count)
        return dict(terms, total=terms['total'] + (auxiliary if arm == 'boundary' else 0 * auxiliary),
            anchor_objective=terms['total'], boundary_objective=auxiliary, **info)

    parent.losses = loss
    try:
        yield
    finally:
        parent.losses = before


def fit(data, *, warm, arm, **kwargs):
    if warm['arm'] != 'own_choice':
        raise ValueError('Both arms start from the same completed own-choice critic')
    # The architecture is identical to the learned-reference initializer.
    # Only its source tag is adapted; weights and feature hashes are unchanged.
    with installed(arm):
        return parent.fit(data, warm=dict(warm, arm='learned_references'), arm='fixed_choice', **kwargs)


predict = parent.predict


def assert_matched(left, right):
    if [s['identity']['boundary_arm'] for s in (left, right)] != list(ARMS):
        raise ValueError('Both registered objective arms required in order')
    normalize = lambda state: dict(state, identity={k: v for k, v in state['identity'].items() if k != 'boundary_arm'})
    parent.assert_matched(normalize(left), normalize(right))


assert_replay = parent.assert_replay
