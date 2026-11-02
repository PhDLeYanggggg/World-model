"""SOURCE-only, query-context cost learning for the frozen selection forecast bank."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_floor_relative import matched_features
from src.world_model.m3w_european_source_intervention import causal_cost_features
from src.world_model.m3w_multiaction_cost import preprocessing, exact
from src.world_model.m3w_reference_cost_refit import CyclicQueries, steps_for
from src.world_model.m3w_causal_descriptor_head import read_checkpoint, save_checkpoint
from src.world_model.m3w_frame_action_allocation import query_groups
from src.evaluation.m3w_native_metrics import native_errors

ARMS = ("row", "context")
CAUSAL_KEYS = {"x", "envelope", "scales", "allowed", "groups", "source_groups", "sites", "ids"}


def features(*, geometry, detector_history, rollouts, floor_index, recordings, frames, ids, sites):
    """Rollouts are relative to the observed current position; no labels accepted."""
    g, d, p = np.asarray(geometry), np.asarray(detector_history), np.asarray(rollouts)
    n = len(g)
    if (g.shape != (n, 476) or d.shape != (n, 64) or p.shape != (n, 7, 12, 2)
            or floor_index not in range(6) or not all(np.isfinite(v).all() for v in (g, d, p))):
        raise ValueError("Exact causal geometry, detector64 and seven finite rollouts required")
    if np.shape(sites) != (n,) or np.shape(ids) != (n,):
        raise ValueError("Aligned causal row namespace and sites required")
    cv, floor = p[:, 1], p[:, floor_index]
    floor_switch = np.any(floor != cv, axis=(1, 2))
    x, envelope = np.empty((n, 7, 444), np.float32), np.empty((n, 7), np.float64)
    for k in range(7):
        base, envelope[:, k] = matched_features(g, cv, floor, p[:, k], floor_switch)
        x[:, k, :380] = base
        x[:, k, 380:] = d
    _, scales = causal_cost_features(g, floor, floor)
    groups = query_groups(recordings, frames, ids)
    names = sorted(set(sites))
    group_sites = np.asarray([sites[ix[0]] for ix in groups])
    if any(len(set(np.asarray(sites)[ix])) != 1 for ix in groups):
        raise ValueError("A contemporaneous recording query cannot span sites")
    allowed = np.linalg.norm(p-floor[:, None], axis=-1).max(-1) > 0
    return dict(x=x, envelope=envelope, scales=scales.astype(np.float32), allowed=allowed,
                groups=groups, source_groups=[np.flatnonzero(group_sites == s) for s in names],
                sites=np.asarray(sites), ids=np.asarray(ids))


def labels(rollouts, target, valid, floor_index, easy_cut):
    """Future targets are used here only, after the inference payload is built."""
    costs = np.stack([native_errors(rollouts[:, k], target, valid, np.ones(len(target)))[0]
                      for k in range(7)], 1)
    reference, cv = costs[:, floor_index], costs[:, 1]
    known = np.isfinite(reference)
    easy = known & (cv > 0) & (cv <= easy_cut)
    benefit = np.maximum(reference[:, None]-costs, 0)
    harm = np.maximum(costs-reference[:, None], 0)
    r = np.broadcast_to(reference[:, None], costs.shape)
    result = np.stack((benefit, harm, r, r*easy[:, None], harm*easy[:, None]), -1)
    result[~known] = np.nan
    return result, known


def training_data(causal, target):
    if set(causal) != CAUSAL_KEYS:
        raise ValueError("Only the declared causal schema may enter this learner")
    pr = preprocessing(causal["x"], causal["envelope"], target, causal["scales"], causal["sites"])
    z = np.clip((causal["x"]-pr["mean"])/pr["std"], -pr["clip"], pr["clip"]).astype(np.float32)
    known = np.isfinite(target).all((1, 2))
    truth = np.where(known[:, None, None], target/causal["scales"][:, None, None], 0).astype(np.float32)
    return dict(causal=causal, preprocess=pr, z=torch.from_numpy(z), truth=torch.from_numpy(truth), known=known)


class CostHead(nn.Module):
    def __init__(self, dimension, width, seed, bias, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Registered row/context comparison required")
        torch.manual_seed(seed)
        self.arm = arm
        self.encoder = nn.Linear(dimension+7, width)
        self.action = nn.Linear(width, 3)
        self.reference = nn.Linear(width, 2)
        with torch.no_grad():
            self.action.weight.zero_(); self.reference.weight.zero_()
            self.action.bias.copy_(torch.as_tensor(bias[[0, 1, 4]]))
            self.reference.bias.copy_(torch.as_tensor(bias[[2, 3]]))

    def forward(self, x, envelope, segments, count):
        if x.ndim != 3 or x.shape[1] != 7 or envelope.shape != x.shape[:2] or segments.shape != (len(x),):
            raise ValueError("Complete causal query batches required")
        code = torch.eye(7, dtype=x.dtype).expand(len(x), -1, -1)
        hidden = F.silu(self.encoder(torch.cat((x, code), -1)))
        if self.arm == "context":
            # Fixed membership products avoid parallel scatter accumulation in backprop.
            membership = F.one_hot(segments, count).to(hidden.dtype).T
            total = membership @ hidden.flatten(1)
            sizes = membership.T @ membership.sum(1)
            others = ((membership.T @ total).reshape_as(hidden)-hidden)/(sizes-1).clamp_min(1)[:, None, None]
            hidden = hidden + others
        hidden = F.silu(hidden)
        raw = self.action(hidden)
        ref = self.reference(hidden.mean(1))
        reference = F.softplus(ref[:, 0])[:, None].expand(-1, 7)
        easy_reference = reference*torch.sigmoid(ref[:, 1, None])
        fractions = F.softmax(torch.cat((raw[..., :2], torch.zeros_like(raw[..., :1])), -1), -1)
        benefit = torch.minimum(envelope*fractions[..., 0], reference)
        harm = envelope*fractions[..., 1]
        return torch.stack((benefit, harm, reference, easy_reference, harm*torch.sigmoid(raw[..., 2])), -1)


def loss(prediction, data, ix, segments, count):
    known = torch.from_numpy(data["known"][ix])
    legal = torch.from_numpy(data["causal"]["allowed"][ix]) & known[:, None]
    target = data["truth"][ix]
    rms = torch.from_numpy(data["preprocess"]["rms"])
    error = (prediction-target)/rms
    weights = torch.ones_like(error)
    for k in (1, 4):
        weights[..., k] = torch.where(prediction[..., k] < target[..., k], 4., 1.)
    actions = legal.sum(1).clamp_min(1)
    regression = (error.square().mul(weights).mean(-1)*legal).sum(1)/actions
    delta = ((prediction[..., 0]-prediction[..., 1])-(target[..., 0]-target[..., 1]))/torch.linalg.vector_norm(rms[:2])
    center = (delta*legal).sum(1)/actions
    ranking = ((delta-center[:, None]).square()*legal).sum(1)/actions
    supported = legal.any(1)
    membership = F.one_hot(segments, count).to(prediction.dtype).T
    def query_mean(value):
        sums = membership @ (value*supported)
        ns = membership @ supported.float()
        return (sums/ns.clamp_min(1)).mean()
    reg, rank = query_mean(regression), query_mean(ranking)
    return dict(total=reg+.5*rank, regression=reg, ranking=rank,
                unknown_supervised_rows=int((legal & ~known[:, None]).sum()))


def fit(data, *, arm, seed, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    c, pr = data["causal"], data["preprocess"]
    model = CostHead(c["x"].shape[-1], settings["width"], seed, pr["bias"], arm)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=.0001)
    sampler = CyclicQueries(c["groups"], c["source_groups"], settings["query_batch_size"], seed)
    budget = steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, arm=arm, seed=seed, preprocess=pr,
                   initial_model=initial, parameters=sum(p.numel() for p in model.parameters()),
                   budget=budget, source_sites=sorted(set(c["sites"])), fit_ids=c["ids"].copy())
    step, elapsed, trace, draws, unknown = 0, 0., [], "0"*64, 0
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise ValueError("Explicit same-binding resume required")
        old = read_checkpoint(path)
        for key, value in binding.items(): exact(old[key], value)
        model.load_state_dict(old["model"]); opt.load_state_dict(old["optimizer"])
        sampler.restore(old["sampler"]); torch.set_rng_state(old["torch_rng"])
        step, elapsed, trace, draws, unknown = (old[k] for k in ("step", "seconds", "trace", "draw_hash", "unknown_draws"))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError("Nondecreasing full query-coverage budget required")
    started = time.monotonic()
    def save():
        guard(); save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace, draw_hash=draws,
            unknown_draws=unknown, unknown_supervised_rows=0))
    while step < limit:
        guard(); ix, seg, qids = sampler.draw(); segment = torch.from_numpy(seg)
        env = torch.from_numpy((c["envelope"][ix]/c["scales"][ix, None]).astype(np.float32))
        opt.zero_grad(set_to_none=True)
        terms = loss(model(data["z"][ix], env, segment, len(qids)), data, ix, segment, len(qids))
        if not torch.isfinite(terms["total"]) or terms["unknown_supervised_rows"]:
            raise FloatingPointError("Finite, known-only cost supervision required")
        terms["total"].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        opt.step(); step += 1
        draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        unknown += int((~data["known"][ix]).sum())
        if step == 1 or step % settings["heartbeat_every"] == 0 or step == budget:
            record = dict(step=step, budget=budget, **{k:float(v.detach()) if isinstance(v, torch.Tensor) else v for k,v in terms.items()},
                          gradient_norm=float(norm), minimum_query_visits=int(sampler.counts.min()),
                          never_drawn_queries=int((sampler.counts == 0).sum()))
            trace.append(record); heartbeat(state="controller_training", arm=arm, **record)
        if step % settings["checkpoint_every"] == 0 or step == limit: save()
    state = read_checkpoint(path)
    if step == budget and int(sampler.counts.min()) < settings["largest_source_cycles"]:
        raise AssertionError("Every original SOURCE query must receive full registered coverage")
    return state


def predict(state, causal):
    if set(causal) != CAUSAL_KEYS:
        raise ValueError("Inference refuses targets, validity and future endpoint fields")
    pr = state["preprocess"]
    model = CostHead(causal["x"].shape[-1], state["settings"]["width"], state["seed"], pr["bias"], state["arm"])
    model.load_state_dict(state["model"]); model.eval()
    result = np.empty((len(causal["x"]), 7, 5), np.float64)
    pending, size = [], 0
    def flush():
        ix = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(v) for v in pending])
        z = np.clip((causal["x"][ix]-pr["mean"])/pr["std"], -pr["clip"], pr["clip"]).astype(np.float32)
        scale = causal["scales"][ix]
        env = (causal["envelope"][ix]/scale[:, None]).astype(np.float32)
        with torch.no_grad():
            p = model(torch.from_numpy(z), torch.from_numpy(env), torch.from_numpy(seg), len(pending)).numpy()
        result[ix] = p.astype(float)*scale[:, None, None]
    for group in causal["groups"]:
        if pending and size+len(group) > 1024: flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending: flush()
    if not np.isfinite(result).all() or (result < 0).any():
        raise FloatingPointError("Finite nonnegative coherent cost predictions required")
    return result
