"""Causal seed-spread covariates for the fixed cardinal-utility cost critic."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_selection_cardinal_cost as cardinal

base = cardinal.base
ARMS = ("disabled", "real", "noise")
FEATURES = tuple(f"radial_spread_step{i:02d}" for i in range(1,13)) + (
    "maximum_member_path_disagreement", "endpoint_direction_dispersion",
    "mean_step_speed_spread", "total_path_length_spread")


def spread_features(predictions, scales):
    """Three relative predicted paths; no observed future or outcome mask accepted."""
    p, s = np.asarray(predictions, float), np.asarray(scales, float)
    if (p.ndim != 4 or p.shape[1:] != (3,12,2) or s.shape != (len(p),)
            or not np.isfinite(p).all() or not np.isfinite(s).all() or (s <= 0).any()):
        raise ValueError("Exactly three finite relative forecast paths and causal scales required")
    center = p.mean(1)
    radial = np.sqrt(np.square(p-center[:,None]).sum(-1).mean(1))/s[:,None]
    pair = np.stack([np.linalg.norm(p[:,a]-p[:,b],axis=-1).max(1) for a,b in ((0,1),(0,2),(1,2))],1)
    lengths = np.linalg.norm(p[:,:,-1],axis=-1).mean(1)
    direction = np.zeros(len(p)); moving = lengths > 0
    direction[moving] = np.clip(1-np.linalg.norm(center[moving,-1],axis=-1)/lengths[moving],0,1)
    speeds = np.linalg.norm(np.diff(p,axis=2,prepend=np.zeros((len(p),3,1,2))),axis=-1)
    extra = np.column_stack((pair.max(1)/s,direction,speeds.std(1).mean(1)/s,speeds.sum(2).std(1)/s))
    result = np.column_stack((radial,extra)).astype(np.float32)
    if not np.isfinite(result).all() or (result < 0).any(): raise FloatingPointError("Finite spread descriptors required")
    return result


def fit_preprocess(values, sites, ids):
    u, sites, ids = np.asarray(values), np.asarray(sites), np.asarray(ids)
    if u.shape != (len(ids),16) or sites.shape != (len(ids),) or not np.isfinite(u).all():
        raise ValueError("Aligned fitting-SOURCE-only uncertainty required")
    names = sorted(set(sites))
    if not names: raise ValueError("Fitting sources required")
    mean = np.mean([u[sites==name].mean(0,dtype=np.float64) for name in names],0)
    second = np.mean([np.square(u[sites==name].astype(float)).mean(0) for name in names],0)
    std = np.sqrt(np.maximum(second-mean*mean,0)).clip(1e-6)
    rng = np.random.default_rng(20261102)
    sampled = np.concatenate([rng.choice(np.flatnonzero(sites==name),512,replace=True) for name in names])
    catalog = np.clip((u[sampled]-mean)/std,-8,8).astype(np.float32)
    return dict(mean=mean,std=std,clip=8.,source_sites=names,fitting_rows=len(u),
                catalog=catalog,catalog_source_ids=ids[sampled].copy(),catalog_seed=20261102)


def standardize(values, pr):
    if np.shape(values)[-1:] != (16,) or not np.isfinite(values).all():
        raise ValueError("Sixteen finite causal descriptors required")
    return np.clip((values-pr["mean"])/pr["std"],-pr["clip"],pr["clip"]).astype(np.float32)


def noise_rows(ids, pr, domain):
    if domain not in ("source","selection") or np.asarray(ids).ndim != 1:
        raise ValueError("Declared input namespace required")
    # SplitMix64 gives row-order-independent draws from a fitting-SOURCE catalog.
    with np.errstate(over="ignore"):
        x = np.asarray(ids,dtype=np.uint64) + np.uint64(20261102 if domain=="source" else 61102202)
        x = (x ^ (x >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        x ^= x >> np.uint64(31)
    return pr["catalog"][(x % len(pr["catalog"])).astype(np.int64)].copy()


def augment_data(data, uncertainty):
    result = dict(data); c = data["causal"]
    pr = fit_preprocess(uncertainty,c["sites"],c["ids"])
    result["uncertainty_preprocess"] = pr
    result["u"] = {"disabled":torch.zeros((len(c["ids"]),16)),
        "real":torch.from_numpy(standardize(uncertainty,pr)),
        "noise":torch.from_numpy(noise_rows(c["ids"],pr,"source"))}
    return result


class SpreadCostHead(base.CostHead):
    def __init__(self, dimension, width, seed, bias, feature_arm):
        super().__init__(dimension,width,seed,bias,"context")
        if feature_arm not in ARMS: raise ValueError("Matched uncertainty arm required")
        self.feature_arm = feature_arm
        rng = torch.get_rng_state()
        self.uncertainty = nn.Linear(16,width,bias=False)
        with torch.no_grad(): self.uncertainty.weight.zero_()
        torch.set_rng_state(rng)

    def forward(self, x, envelope, segments, count, uncertainty):
        if uncertainty.shape != (len(x),16): raise ValueError("Aligned uncertainty covariates required")
        if self.feature_arm == "disabled": return super().forward(x,envelope,segments,count)
        if x.ndim!=3 or x.shape[1]!=7 or envelope.shape!=x.shape[:2] or segments.shape!=(len(x),):
            raise ValueError("Complete causal query required")
        code = torch.eye(7,dtype=x.dtype).expand(len(x),-1,-1)
        hidden = F.silu(self.encoder(torch.cat((x,code),-1))+self.uncertainty(uncertainty)[:,None])
        membership = F.one_hot(segments,count).to(hidden.dtype).T
        total = membership @ hidden.flatten(1)
        sizes = membership.T @ membership.sum(1)
        others = ((membership.T @ total).reshape_as(hidden)-hidden)/(sizes-1).clamp_min(1)[:,None,None]
        hidden = F.silu(hidden+others)
        raw = self.action(hidden); ref = self.reference(hidden.mean(1))
        reference = F.softplus(ref[:,0])[:,None].expand(-1,7)
        easy_reference = reference*torch.sigmoid(ref[:,1,None])
        fractions = F.softmax(torch.cat((raw[...,:2],torch.zeros_like(raw[...,:1])),-1),-1)
        benefit = torch.minimum(envelope*fractions[...,0],reference); harm = envelope*fractions[...,1]
        return torch.stack((benefit,harm,reference,easy_reference,harm*torch.sigmoid(raw[...,2])),-1)


def compare_disabled(state, original):
    if state["feature_arm"] != "disabled": raise ValueError("Only the disabled control is equivalent")
    for key in ("preprocess","initial_model","model"):
        value={k:v for k,v in state[key].items() if k!="uncertainty.weight"} if key!="preprocess" else state[key]
        base.exact(value,original[key])
    for key in ("budget","sampler","draw_hash","fit_ids","torch_rng","step","trace","unknown_draws"):
        base.exact(state[key],original[key])
    for k,v in original["optimizer"]["state"].items(): base.exact(state["optimizer"]["state"][k],v)
    assert not state["model"]["uncertainty.weight"].count_nonzero()


def fit(data, *, arm, seed, settings, identity, path, heartbeat=lambda **_:None,
        guard=lambda:None, resume=False, stop_at=None):
    c, pr = data["causal"], data["preprocess"]
    model = SpreadCostHead(c["x"].shape[-1],settings["width"],seed,pr["bias"],arm)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=.0001)
    sampler = base.CyclicQueries(c["groups"],c["source_groups"],settings["query_batch_size"],seed)
    budget = base.steps_for(c,settings)
    binding = dict(identity=identity,settings=settings,feature_arm=arm,seed=seed,preprocess=pr,
        uncertainty_preprocess=data["uncertainty_preprocess"],initial_model=initial,
        parameters=sum(p.numel() for p in model.parameters()),budget=budget,
        source_sites=sorted(set(c["sites"])),fit_ids=c["ids"].copy())
    step,elapsed,trace,draws,unknown = 0,0.,[],"0"*64,0
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not resume: raise ValueError("Explicit same-binding resume required")
        old = base.read_checkpoint(path)
        for key,value in binding.items(): base.exact(old[key],value)
        model.load_state_dict(old["model"]);opt.load_state_dict(old["optimizer"])
        sampler.restore(old["sampler"]);torch.set_rng_state(old["torch_rng"])
        step,elapsed,trace,draws,unknown = (old[k] for k in ("step","seconds","trace","draw_hash","unknown_draws"))
    limit = budget if stop_at is None else min(stop_at,budget)
    if not 0<=step<=limit or limit<=0:raise ValueError("Fixed complete-coverage budget required")
    started = time.monotonic()
    def save():
        guard();base.save_checkpoint(path,dict(**binding,model=model.state_dict(),optimizer=opt.state_dict(),
            sampler=sampler.state(),torch_rng=torch.get_rng_state(),step=step,
            seconds=elapsed+time.monotonic()-started,trace=trace,draw_hash=draws,
            unknown_draws=unknown,unknown_supervised_rows=0))
    if step==0:save()
    while step<limit:
        guard();ix,seg,qids=sampler.draw();segment=torch.from_numpy(seg)
        env=torch.from_numpy((c["envelope"][ix]/c["scales"][ix,None]).astype(np.float32))
        opt.zero_grad(set_to_none=True)
        terms=cardinal.objective(model(data["z"][ix],env,segment,len(qids),data["u"][arm][ix]),
            data,ix,segment,len(qids),"mean_absolute")
        if not torch.isfinite(terms["total"]) or terms["unknown_supervised_rows"]:
            raise FloatingPointError("Finite known-only SOURCE cost loss required")
        terms["total"].backward();norm=nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip"],error_if_nonfinite=True)
        opt.step();step+=1
        draws=hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        unknown+=int((~data["known"][ix]).sum())
        if step==1 or step%settings["heartbeat_every"]==0 or step==budget:
            record=dict(step=step,budget=budget,
                **{k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in terms.items()},
                gradient_norm=float(norm),minimum_query_visits=int(sampler.counts.min()),
                never_drawn_queries=int((sampler.counts==0).sum()))
            trace.append(record);heartbeat(state="uncertainty_cost_training",arm=arm,**record)
        if step%settings["checkpoint_every"]==0 or step==limit:save()
    if step==budget and sampler.counts.min()<settings["largest_source_cycles"]:
        raise AssertionError("Every original fitting query needs complete coverage")
    return base.read_checkpoint(path)


def predict(state, causal, uncertainty, *, domain):
    if set(causal)!=base.CAUSAL_KEYS or np.shape(uncertainty)!=(len(causal["ids"]),16) or not np.isfinite(uncertainty).all():
        raise ValueError("Only declared causal inputs and forecast-spread fields admitted")
    if domain not in ("source","selection"):raise ValueError("Declared inference role required")
    pr=state["preprocess"];up=state["uncertainty_preprocess"];arm=state["feature_arm"]
    if arm=="disabled":u=np.zeros((len(uncertainty),16),np.float32)
    elif arm=="noise":u=noise_rows(causal["ids"],up,domain)
    else:u=standardize(uncertainty,up)
    model=SpreadCostHead(causal["x"].shape[-1],state["settings"]["width"],state["seed"],pr["bias"],arm)
    model.load_state_dict(state["model"]);model.eval();result=np.empty((len(u),7,5),np.float64)
    pending=[];size=0
    def flush():
        ix=np.concatenate(pending);seg=np.repeat(np.arange(len(pending)),[len(v) for v in pending])
        z=np.clip((causal["x"][ix]-pr["mean"])/pr["std"],-pr["clip"],pr["clip"]).astype(np.float32)
        scale=causal["scales"][ix];env=(causal["envelope"][ix]/scale[:,None]).astype(np.float32)
        with torch.no_grad():p=model(torch.from_numpy(z),torch.from_numpy(env),torch.from_numpy(seg),len(pending),torch.from_numpy(u[ix])).numpy()
        result[ix]=p.astype(float)*scale[:,None,None]
    for group in causal["groups"]:
        if pending and size+len(group)>1024:flush();pending=[];size=0
        pending.append(group);size+=len(group)
    if pending:flush()
    if not np.isfinite(result).all() or (result<0).any():raise FloatingPointError("Finite coherent costs required")
    return result
