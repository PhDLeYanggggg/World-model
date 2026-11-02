"""Factorial cost-loss repair with the original frozen causal controller model."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_selection_controller as base

ARMS = {
    "weighted_centered": (4., True),
    "mean_centered": (1., True),
    "weighted_absolute": (4., False),
    "mean_absolute": (1., False),
}
NEW_ARMS = tuple(ARMS)[1:]


def objective(prediction, data, ix, segments, count, arm):
    weight, centered = ARMS[arm]
    known = torch.from_numpy(data["known"][ix])
    legal = torch.from_numpy(data["causal"]["allowed"][ix]) & known[:, None]
    target = data["truth"][ix]
    rms = torch.from_numpy(data["preprocess"]["rms"])
    error = (prediction-target)/rms
    weights = torch.ones_like(error)
    for k in (1, 4):
        weights[..., k] = torch.where(prediction[..., k] < target[..., k], weight, 1.)
    actions = legal.sum(1).clamp_min(1)
    regression = (error.square().mul(weights).mean(-1)*legal).sum(1)/actions
    delta = ((prediction[..., 0]-prediction[..., 1])-(target[..., 0]-target[..., 1]))/torch.linalg.vector_norm(rms[:2])
    if centered:
        delta = delta-(delta*legal).sum(1)[:, None]/actions[:, None]
    ranking = (delta.square()*legal).sum(1)/actions
    supported = legal.any(1)
    membership = F.one_hot(segments, count).to(prediction.dtype).T
    def query_mean(value):
        return ((membership @ (value*supported))/(membership @ supported.float()).clamp_min(1)).mean()
    reg, rank = query_mean(regression), query_mean(ranking)
    return dict(total=reg+.5*rank, regression=reg, ranking=rank,
                unknown_supervised_rows=int((legal & ~known[:, None]).sum()))


def fit(data, *, arm, seed, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError("Fixed factorial loss arm required")
    c, pr = data["causal"], data["preprocess"]
    model = base.CostHead(c["x"].shape[-1], settings["width"], seed, pr["bias"], "context")
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=.0001)
    sampler = base.CyclicQueries(c["groups"], c["source_groups"], settings["query_batch_size"], seed)
    budget = base.steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, arm="context", objective_arm=arm, seed=seed,
        preprocess=pr, initial_model=initial, parameters=sum(p.numel() for p in model.parameters()),
        budget=budget, source_sites=sorted(set(c["sites"])), fit_ids=c["ids"].copy())
    step, elapsed, trace, draws, unknown = 0, 0., [], "0"*64, 0
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise ValueError("Explicit same-binding resume required")
        old = base.read_checkpoint(path)
        for key, value in binding.items(): base.exact(old[key], value)
        model.load_state_dict(old["model"]); opt.load_state_dict(old["optimizer"])
        sampler.restore(old["sampler"]); torch.set_rng_state(old["torch_rng"])
        step, elapsed, trace, draws, unknown = (old[k] for k in ("step", "seconds", "trace", "draw_hash", "unknown_draws"))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError("Nondecreasing complete-coverage budget required")
    started = time.monotonic()
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace, draw_hash=draws,
            unknown_draws=unknown, unknown_supervised_rows=0))
    if step == 0:
        save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segment = torch.from_numpy(seg)
            env = torch.from_numpy((c["envelope"][ix]/c["scales"][ix, None]).astype(np.float32))
            opt.zero_grad(set_to_none=True)
            terms = objective(model(data["z"][ix], env, segment, len(qids)), data, ix, segment, len(qids), arm)
            if not torch.isfinite(terms["total"]) or terms["unknown_supervised_rows"]:
                raise FloatingPointError("Finite known-only targets required")
            terms["total"].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            opt.step(); step += 1
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            unknown += int((~data["known"][ix]).sum())
            if step == 1 or step % settings["heartbeat_every"] == 0 or step == budget:
                record = dict(step=step, budget=budget,
                    **{k:float(v.detach()) if isinstance(v, torch.Tensor) else v for k,v in terms.items()},
                    gradient_norm=float(norm), minimum_query_visits=int(sampler.counts.min()),
                    never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(record); heartbeat(state="cardinal_cost_training", objective_arm=arm, **record)
            if step % settings["checkpoint_every"] == 0 or step == limit: save()
    except (KeyboardInterrupt, TimeoutError):
        # Keep the last atomic completed-update checkpoint, never a half-step.
        raise
    state = base.read_checkpoint(path)
    if step == budget and int(sampler.counts.min()) < settings["largest_source_cycles"]:
        raise AssertionError("Every fitting query needs the original complete coverage")
    return state
