"""Allow both released role credits to expire at a fixed causal deadline."""

import numpy as np

from src.world_model.m3w_opendd_dual_clock import LOOKAHEAD, proposals


def stream(packets, score):
    spent = ()
    released, ended = {}, set()
    for q, packet in enumerate(packets):
        if len(ended) == 2:
            break
        for r, rows in enumerate(packet):
            if rows and r not in released:
                released[r] = q
        due = {r for r in released if r not in ended and q-released[r] == LOOKAHEAD}
        active = tuple(rows if r not in ended else () for r, rows in enumerate(packet))
        menu = tuple(a for a in proposals(active, spent) if a)
        if menu:
            z = np.asarray(score(q, packet, spent, menu))
            if z.shape != (len(menu), 3) or not np.isfinite(z).all():
                raise ValueError("Invalid current decision scores")
            values = np.minimum(z[:, 1], z[:, 2])
            urgency = np.array([sum(r in due for r, _ in a) for a in menu])
            if urgency.max() > 0:
                # At a deadline maximize the number of due credits spent; avoid
                # an extra non-due edit unless it has positive estimated value.
                extra = np.array([len(a) for a in menu])-urgency
                take = (urgency == urgency.max()) & ((extra == 0) | (values > 0.))
                pick = int(np.where(take, z[:, 1], -np.inf).argmax())
            else:
                pick = int(values.argmax()) if values.max() > 0. else None
            if pick is not None:
                spent += menu[pick]
                ended.update(r for r, _ in menu[pick])
        ended.update(due)
    return spent
