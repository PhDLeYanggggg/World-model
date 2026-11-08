import numpy as np

from src.world_model import m3w_opendd_credit_value as model


def test_offline_value_matches_enumeration_with_expiry_and_terminal_cost():
    release = [2, 0, 1, 1, 0]
    rewards = [np.array(a) for a in ([0., .1, .25], [0., .4], [0., .7, .9], [0., -.2, -.5], [0., .3])]
    scale = [1., 1., 2., 3., 4.]

    def brute(t, pending):
        if t == len(release):
            return -model.EXPIRY*scale[-1]*len(pending)
        tokens = pending+[t]*release[t]
        out = []
        for use in range(min(len(rewards[t])-1, len(tokens))+1):
            remain = tokens[use:]
            future = [j for j in remain if j > t-2]
            out.append(rewards[t][use]-model.EXPIRY*scale[t]*(len(remain)-len(future))+brute(t+1, future))
        return max(out)

    labels, initial = model.action_targets(release, rewards, scale)
    np.testing.assert_allclose(initial, brute(0, []))
    assert labels[0][(0, 0)][2, 0] > 0.
    assert labels[0][(0, 0)][2, 1] < 0.


def test_online_spending_has_prefix_invariance_and_counts_expired_credits():
    def make(when, row, n):
        return np.array(row), (np.array([when]), np.zeros((len(row)+1, 1)), n)

    packets = [make(0, [0, 1], 2), make(1, [2], 0), make(2, [3], 0), make(3, [4], 0)]
    score = lambda x: np.arange(len(x)) if x[0, 0] == 2 else -np.arange(len(x))
    full, expired = model.stream(iter(packets), score)
    assert full == [(3, 0, 2, 2)]
    assert expired == 1
    for t in range(len(packets)):
        prefix, _ = model.stream(iter(packets[:t]), score)
        assert prefix == [e for e in full if e[2] < t]


def test_fixed_rank_cannot_skip_first_candidate_but_can_move_credit():
    from scripts.run_m3w_opendd_prefix_capacity import solve

    rows = [np.array([0, 1]), np.array([2])]
    gain = np.ones((3, 2))
    risk = np.zeros((3, 3))
    risk[0, 0] = .5
    support = np.ones(3, dtype=bool)
    assert solve([1, 0], rows, gain, risk, support, 0., 1., 0)[0] == "risk_infeasible"
    outcome, chosen = solve([1, 0], rows, gain, risk, support, 0., 1., 2)
    assert outcome == "qualified"
    np.testing.assert_array_equal(chosen, [2])
