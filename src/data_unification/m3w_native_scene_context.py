"""Identity-resolved current-scene context for source-only 8-to-12 queries."""
import numpy as np


class SourceSceneContext:
    def __init__(self, rows, labels, stride=12):
        a, labels = np.asarray(rows, float), np.asarray(labels)
        if (a.ndim != 2 or a.shape[1] != 9 or labels.shape != (len(a),) or stride != 12
                or not np.isfinite(a).all() or np.any(a[:, 3:5] < a[:, 1:3])
                or not np.isin(a[:, 6:9], [0, 1]).all()
                or np.any(a[:, [0, 5]] != np.rint(a[:, [0, 5]])) or np.any(a[:, [0, 5]] < 0)):
            raise ValueError('Valid source annotation identities and boxes required')
        order = np.lexsort((a[:, 5], a[:, 0])); a, labels = a[order], labels[order]
        if ((np.diff(a[:, 0]) == 0) & (np.diff(a[:, 5]) == 0)).any():
            raise ValueError('Duplicate annotation identity')
        selected = (a[:, 6] == 0) & (a[:, 5] % stride == 0)
        a, labels = a[selected], labels[selected]
        self.agent, self.frame = a[:, 0].astype(np.int64), a[:, 5].astype(np.int64)
        self.xy, self.box, self.labels = (a[:, 1:3]+a[:, 3:5])/2, a[:, 1:5], labels
        starts = np.r_[0, np.flatnonzero(np.diff(self.agent))+1]
        self.track_start = np.repeat(starts, np.diff(np.r_[starts, len(a)]))
        self.frame_order = np.argsort(self.frame, kind='stable')
        self.frame_values = self.frame[self.frame_order]

    def at_frame(self, frame):
        lo, hi = np.searchsorted(self.frame_values, frame, side='left'), np.searchsorted(self.frame_values, frame, side='right')
        return self.frame_order[lo:hi]

    def history(self, current):
        current = np.asarray(current, dtype=np.int64)
        indices = current[:, None]-np.arange(7, -1, -1)
        mask = indices >= self.track_start[current, None]
        safe = np.maximum(indices, self.track_start[current, None])
        xy = np.where(mask[..., None], self.xy[safe], 0.)
        offsets = np.where(mask, self.frame[safe]-self.frame[current, None], 0)
        if np.any(offsets[mask] > 0):
            raise ValueError('Future context')
        return xy, offsets, mask

    def build(self, keys, origin, rotation, scale, geometry, global_ids):
        keys, origin, rotation, scale, g, global_ids = map(np.asarray,
            (keys, origin, rotation, scale, geometry, global_ids))
        n = len(keys)
        if (keys.shape != (n, 2) or origin.shape != (n, 2) or rotation.shape != (n, 2, 2)
                or scale.shape != (n,) or g.shape != (n, 476) or global_ids.shape != (n,)
                or keys.dtype.kind not in 'iu' or len(np.unique(keys, axis=0)) != n
                or len(np.unique(global_ids)) != n):
            raise ValueError('Unique aligned target keys and geometry required')
        neighbor_source = np.full((n, 8), -1, np.int64)
        neighbors_target = np.full((n, 8), -1, np.int64)
        context_parts, context_groups = [], []
        target_lookup = {tuple(k):int(i) for k,i in zip(keys, global_ids)}
        for f in np.unique(keys[:, 0]):
            ids = np.flatnonzero(keys[:, 0] == f); vis = self.at_frame(int(f))
            context_parts.append(vis); context_groups.extend([int(f)]*len(vis))
            where = {int(self.agent[p]):int(p) for p in vis}
            current = np.array([where[int(keys[i, 1])] for i in ids])
            np.testing.assert_array_equal(self.xy[current], origin[ids])
            np.testing.assert_array_equal(g[ids, 300], len(vis)-1)
            for i, p in zip(ids, current):
                others = vis[vis != p]
                distances = np.linalg.norm(self.xy[others]-self.xy[p], axis=1)
                ordered = np.lexsort((self.agent[others], distances))
                nr = others[ordered[:8]]
                neighbor_source[i, :len(nr)] = nr
                neighbors_target[i, :len(nr)] = [target_lookup.get((int(f), int(self.agent[j])), -1) for j in nr]
        observed = neighbor_source >= 0
        nr = neighbor_source[observed]
        hist, offsets, valid = self.history(nr)
        owner = np.broadcast_to(np.arange(n)[:, None], (n, 8))[observed]
        local = np.einsum('nti,nij->ntj', hist-origin[owner, None], rotation[owner])/scale[owner, None, None]
        local = np.where(valid[..., None], local, 0.)
        expected_xy = g[:, 38:166].reshape(n, 8, 8, 2)[observed]
        expected_t = g[:, 166:230].reshape(n, 8, 8)[observed]
        expected_mask = g[:, 230:294].reshape(n, 8, 8)[observed]
        np.testing.assert_array_equal(valid, expected_mask.astype(bool))
        np.testing.assert_allclose(local.astype(np.float32), expected_xy, atol=1e-7, rtol=2e-7)
        np.testing.assert_array_equal((offsets/144).astype(np.float32), expected_t)
        assert not g[:, 230:294].reshape(n, 8, 8)[~observed].any()
        neighbor_agents = np.full((n, 8), -1, np.int64); neighbor_agents[observed] = self.agent[nr]
        context_rows = np.concatenate(context_parts)
        cx, ct, cm = self.history(context_rows)
        # Short or irregular histories remain explicit; no invented zero velocity.
        velocity_valid = cm[:, -2:].all(1)
        dt = ct[:, -1]-ct[:, -2]
        velocity = np.zeros((len(cx), 2))
        velocity[velocity_valid] = (cx[velocity_valid, -1]-cx[velocity_valid, -2])/dt[velocity_valid, None]
        future = self.xy[context_rows, None]+velocity[:, None]*np.arange(1, 13)[None, :, None]*12
        support = np.broadcast_to(velocity_valid[:, None], (len(cx), 12)).copy()
        target = np.array([target_lookup.get((int(f), int(a)), -1) for f,a in zip(self.frame[context_rows], self.agent[context_rows])])
        return dict(neighbor_agent_ids=neighbor_agents, neighbor_target_rows=neighbors_target,
            neighbor_observed=observed, context_agent_ids=self.agent[context_rows],
            context_frame_ids=np.asarray(context_groups), context_agent_type=self.labels[context_rows],
            context_xy=self.xy[context_rows], context_box=self.box[context_rows],
            context_history=cx, context_history_offsets=ct, context_history_mask=cm,
            context_target_rows=target, context_cv_rollout=future, context_cv_valid=support,
            compared_neighbor_history_points=int(valid.sum()),
            max_normalized_neighbor_error=float(np.max(np.abs(local-expected_xy))) if len(local) else 0.,
            irregular_neighbor_histories=int((np.any((np.diff(offsets, axis=1) != 12) & (valid[:, 1:] & valid[:, :-1]), axis=1)).sum()))
