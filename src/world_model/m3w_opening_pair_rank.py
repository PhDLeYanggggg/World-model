"""Matched pointwise and within-query ranking for useful opening actions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STEPS = 140
QUERIES_PER_UPDATE = 12
PAIR_WEIGHT = .5


def features(x):
    n,k,_ = x.shape
    # A shared trunk with a family indicator. Remove the old constant column.
    indicator = np.broadcast_to(np.eye(k,dtype=np.float32),(n,k,k))
    return np.concatenate((x[:,:,1:].astype(np.float32),indicator),axis=-1)


def query_pool(groups, legal, good, priority):
    pools = []
    for ix in groups:
        if len(ix)<2:
            continue
        allowed = np.flatnonzero(legal[ix].ravel())
        positive = allowed[good[ix].ravel()[allowed]]
        negative = allowed[~good[ix].ravel()[allowed]]
        if not len(negative):
            continue
        hard = negative[np.argsort(-priority[ix].ravel()[negative],kind='stable')[:8]]
        global_row = lambda v: ix[v//legal.shape[1]]*legal.shape[1]+v%legal.shape[1]
        pools.append(tuple(global_row(v) for v in (positive,negative,hard)))
    return pools


def batch_plan(pools, seed, steps=STEPS):
    """Sample identical fitting pairs and pointwise cases for both arms."""
    rng = np.random.default_rng(seed)
    ordinary = [v for v in pools if v]
    useful = [[q for q in v if len(q[0])] for v in ordinary]
    useful = [v for v in useful if v]
    if not ordinary or not useful:
        raise ValueError('Both ordinary and useful fitting queries required')
    out = []
    for _ in range(steps):
        queries = []
        for i in range(QUERIES_PER_UPDATE):
            src = ordinary if i%2==0 else useful
            q = src[rng.integers(len(src))]
            p,n,h = q[rng.integers(len(q))]
            positive = rng.choice(p,min(len(p),4),replace=False)
            extra = np.setdiff1d(n,h,assume_unique=True)
            negative = np.r_[h,rng.choice(extra,min(len(extra),8),replace=False)]
            queries.append((positive,negative))
        out.append(queries)
    return out


def objective(logits, layout, pair_weight):
    point, rank = [], []
    offset = 0
    for p,n in layout:
        pos = logits[offset:offset+p]
        neg = logits[offset+p:offset+p+n]
        offset += p+n
        a = F.softplus(neg).mean()
        point.append((F.softplus(-pos).mean()+a)/2 if p else a)
        if p:
            rank.append(F.softplus(neg[None]-pos[:,None]).mean())
    a = torch.stack(point).mean()
    b = torch.stack(rank).mean() if rank else a*0.
    return a+pair_weight*b


def network(width,seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(width,24),nn.Tanh(),nn.Linear(24,1))


def train(x, plan, seed, pair_weight, checkpoint, beat):
    net = network(x.shape[1],seed)
    opt = torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001)
    step = 0
    if checkpoint.exists():
        old = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if old['seed']!=seed or old['pair_weight']!=pair_weight:
            raise ValueError('Incompatible saved training state')
        net.load_state_dict(old['network'])
        opt.load_state_dict(old['optimizer'])
        step = old['step']
    if not 0<=step<=len(plan):
        raise ValueError('Cannot change registered update count')
    for i in range(step,len(plan)):
        q = plan[i]
        ix = np.concatenate([a for pair in q for a in pair])
        layout = [(len(p),len(n)) for p,n in q]
        loss = objective(net(torch.from_numpy(x[ix])).squeeze(-1),layout,pair_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite opening-rank objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i+1)%35==0 or i+1==len(plan):
            tmp = checkpoint.with_suffix('.pending')
            torch.save(dict(seed=seed,pair_weight=pair_weight,step=i+1,
                            network=net.state_dict(),optimizer=opt.state_dict()),tmp)
            tmp.replace(checkpoint)
            beat(step=i+1,loss=float(loss.detach()))
    return net.eval()


def predict(net,x):
    with torch.no_grad():
        return net(torch.from_numpy(x.reshape(-1,x.shape[-1]))).sigmoid().numpy().reshape(x.shape[:2])
