"""Learn the value of keeping a released edit credit for a later query."""

import numpy as np
import torch
from torch import nn


EXPIRY = .05


def actions(state, released, cap):
    """FIFO queue with two old age bins and one newly released bin."""
    old, recent = state
    for k in range(min(old+recent+released, cap)+1):
        spent_recent = max(k-old, 0)
        yield k, (max(recent-spent_recent, 0), released-max(spent_recent-recent, 0)), max(old-k, 0)


def action_targets(releases, prefixes, scales):
    """Offline labels only: optimal continuation conditional on each action."""
    states = [{(0, 0)}]
    for release, reward in zip(releases, prefixes, strict=True):
        states.append({n for s in states[-1] for _, n, _ in actions(s, release, len(reward)-1)})
    tail = {s: -EXPIRY*scales[-1]*sum(s) for s in states[-1]}
    labels = []
    for t in reversed(range(len(releases))):
        packet = {}
        for state in sorted(states[t]):
            a = list(actions(state, releases[t], len(prefixes[t])-1))
            now = np.array([prefixes[t][k]-EXPIRY*scales[t]*lost for k, _, lost in a])
            ahead = now + np.array([tail[n] for _, n, _ in a])
            packet[state] = np.column_stack((now-now[0], ahead-ahead[0]))/scales[t]
            packet[state][0] = 0.
            if state == (0, 0) and t == 0:
                start = float(ahead.max())
        tail = {s: max(prefixes[t][k]-EXPIRY*scales[t]*lost+tail[n]
                       for k, n, lost in actions(s, releases[t], len(prefixes[t])-1))
                for s in packet}
        labels.append(packet)
    return labels[::-1], start


def feature_rows(packet, state):
    context, summaries, released = packet
    moves = list(actions(state, released, len(summaries)-1))
    return np.asarray([np.r_[context, summaries[k], np.log1p([*state, released, k, *n, lost])]
                       for k, n, lost in moves], dtype=np.float32)


def stream(packets, score):
    """Inputs are revealed one packet at a time; no final-query flag."""
    state = (0, 0)
    pending, events, expired = [], [], 0
    for q, (rows, packet) in enumerate(packets):
        released = packet[2]
        values = score(feature_rows(packet, state))
        if not np.isfinite(values).all() or values.shape != (min(sum(state)+released, len(rows))+1,):
            raise ValueError("Bad scores for legal FIFO actions")
        values = values.copy()
        values[0] = 0.
        move = int(values.argmax())
        pending.extend((q, int(r)) for r in rows[:released])
        events.extend((int(r), old, q, q-t) for r, (t, old) in zip(rows[:move], pending[:move], strict=True))
        remain = pending[move:]
        pending = [(t, r) for t, r in remain if q-t < 2]
        _, state, lost = list(actions(state, released, len(rows)))[move]
        if len(remain)-len(pending) != lost or len(pending) != sum(state):
            raise ValueError("FIFO state lost a credit")
        expired += lost
    return events, expired+len(pending)


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2))


def loss(net, x, y):
    return nn.functional.smooth_l1_loss(net(x), y, beta=.05)
