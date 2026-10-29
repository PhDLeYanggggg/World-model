"""TRAIN-internal shared risk learning with frozen whole-recording holdouts."""
import copy
import hashlib
import math
from pathlib import Path
import time

from src.world_model import m3w_scaled_risk_training as local
from src.world_model.m3w_false_safe_diagnostic import moving_from_features

old, np, torch = local.old, local.old.np, local.old.torch


def recording_partition(source, recordings):
    names = sorted(set(map(str, recordings)))
    if len(names) < 2:
        raise ValueError('Whole-recording inner holdout requires two TRAIN recordings')
    ordered = sorted(names, key=lambda r:hashlib.sha256(
        ('shared-risk-inner-v1|'+source+'|'+r).encode()).hexdigest())
    held = ordered[:max(1, math.ceil(len(ordered)/4))]
    return dict(fit_recordings=sorted(set(names)-set(held)), held_recordings=sorted(held))


def prepare(args, partition):
    arrays = args[:-1]
    source = args[-1]['training_site']
    records = args[5].astype(str)
    if partition != recording_partition(source, records):
        raise ValueError('Fixed input-only recording partition changed')
    fit = np.isin(records, partition['fit_recordings'])
    held = np.isin(records, partition['held_recordings'])
    if not fit.any() or not held.any() or (fit & held).any() or not (fit | held).all():
        raise ValueError('Whole TRAIN partition must be exhaustive and disjoint')
    subsets = [tuple(a[mask] for a in arrays) for mask in (fit,held)]
    x,env,y,series,sites,recs,frames = subsets[0]
    pr = old.core.preprocess(x,env,y,sites,recs,frames,training_site=source)
    return (*subsets[0],pr), (*subsets[1],pr)


def eligible(args):
    return moving_from_features(args[0]) & old.parent.forest.causal_inputs(args[0],args[1],args[-1])[1]


def shared_preprocess(packets):
    names = sorted(packets)
    targets,envs,sites,recs,frames = [],[],[],[],[]
    for name in names:
        x,env,y,_,site,rec,frame,pr = packets[name]
        targets.append(y/pr['scale']); envs.append(env/pr['scale'])
        sites.append(site); recs.append(rec); frames.append(frame)
    y,env = np.concatenate(targets),np.concatenate(envs)
    site,rec,frame = map(np.concatenate,(sites,recs,frames))
    known = np.isfinite(y).all(1)
    w,_ = old.core.weights(site,rec,frame,known)
    means = w[known]@y[known]
    mean_env = float(w@env)
    fractions = np.maximum([means[0]/max(mean_env,1e-8),means[1]/max(mean_env,1e-8),
                           1-(means[0]+means[1])/max(mean_env,1e-8)],1e-5)
    fractions /= fractions.sum()
    cond = np.clip([means[3]/max(means[2],1e-8),means[4]/max(means[1],1e-8)],1e-5,1-1e-5)
    bias = np.r_[np.log(fractions[:2]/fractions[2]),np.log(np.expm1(means[2])),
                 np.log(cond/(1-cond))].astype(np.float32)
    ext = np.column_stack((y[known],old.core.signed(y[known])))
    rms = np.sqrt((w[known,None]*ext**2).sum(0)).clip(.01).astype(np.float32)
    return dict(mean=np.zeros(packets[names[0]][0].shape[1]),bias=bias,rms=rms)


def fit_shared(packets, *, mode, settings, objective_config, seed, identity,
               experiment_sha256, path, heartbeat, resume=False, stop_at=None, guard=None):
    if mode not in ('shared_global','shared_source') or len(packets) != 4:
        raise ValueError('Four same-context TRAIN sources and registered shared objective required')
    old.validate_objective(objective_config)
    names = sorted(packets); pr = shared_preprocess(packets)
    data = {}; hashes = {}; norms = {}; scales = []
    for name in names:
        x,env,y,series,sites,recs,frames,p = packets[name]
        known = np.isfinite(y).all(1)
        groups,sg,keys = old.core.sampling.query_groups(sites,recs,frames,known)
        z = np.clip((x-p['mean'])/p['std'],-p['clip'],p['clip']).astype(np.float32)
        data[name] = dict(z=torch.from_numpy(z),env=torch.from_numpy((env/p['scale']).astype(np.float32)),
            truth=torch.from_numpy(np.where(known[:,None],y/p['scale'],0).astype(np.float32)),
            eligible=torch.from_numpy(eligible(packets[name])),groups=groups,sg=sg,keys=keys,
            rng=torch.Generator().manual_seed(seed+7919),draw_hash='0'*64,row_draws=0)
        hashes[name] = {k:old.parent.forest.fingerprint(a) for k,a in zip(
            ('x','envelope','primary','temporal','sites','recordings','frames'),packets[name][:-1])}
        norms[name] = p
        scales.append(local.scaling.train_budget_scales(y,sites,recs,frames,p['scale']))
    scale = torch.from_numpy(np.stack(scales))
    model = old.parent.initialize(pr,settings,seed)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    dual = torch.zeros((4,2) if mode == 'shared_source' else (2,))
    path = Path(path); step=0; seconds=0.; trace=[]
    binding = dict(mode=mode,settings=settings,objective_config=objective_config,seed=seed,
        identity=identity,experiment_sha256=experiment_sha256,preprocess=pr,
        source_preprocess=norms,input_hashes=hashes,initial_model=initial,scales=scale)
    if path.exists():
        if not resume: raise ValueError('Explicit resume required')
        state = old.core.read_checkpoint(path)
        for k,v in binding.items(): old.core.exact(state[k],v)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        step,seconds,trace,dual = (state[k] for k in ('step','seconds','trace','dual'))
        for name in names:
            data[name]['rng'].set_state(state['samplers'][name]['rng'])
            for k in ('draw_hash','row_draws'): data[name][k] = state['samplers'][name][k]
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Fixed nondecreasing update budget required')
    began = time.monotonic(); rms = torch.from_numpy(pr['rms'])
    def save():
        if guard: guard()
        old.core.save_checkpoint(path,dict(**binding,step=step,seconds=seconds+time.monotonic()-began,
            trace=trace,dual=dual,model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            samplers={name:dict(rng=d['rng'].get_state(),draw_hash=d['draw_hash'],row_draws=d['row_draws'])
                      for name,d in data.items()},new_forecaster=False,inner_held_scored=False))
    path.parent.mkdir(parents=True,exist_ok=True)
    if step == 0: save()
    while step < limit:
        opt.zero_grad(set_to_none=True); primary=[]; regret=[]; risks=[]; action=[]
        for name in names:
            d = data[name]
            ix,seg,qids = old.core.sampling.draw_queries(d['groups'],d['sg'],settings['query_batch_size'],d['rng'])
            p,_ = model(d['z'][ix],d['env'][ix]); seg=torch.from_numpy(seg)
            primary.append(old.core.losses(p,d['truth'][ix],seg,settings['query_batch_size'],rms)['total'])
            task = old.decision.terms(p,d['truth'][ix],seg,settings['query_batch_size'],d['eligible'][ix],rms,
                                     old.temperature(step,settings,objective_config))
            regret.append(task['regret']);risks.append(task['risk_excess']);action.append(task['action_mean'])
            d['row_draws'] += len(ix)
            d['draw_hash'] = hashlib.sha256(bytes.fromhex(d['draw_hash'])+qids.tobytes()+ix.tobytes()).hexdigest()
        risk = torch.stack(risks)
        excess = risk/scale if mode == 'shared_source' else risk.mean(0)/scale.mean(0)
        penalty = (dual.detach()*excess).sum()/(4 if mode == 'shared_source' else 1)
        loss = torch.stack(primary).mean()+torch.stack(regret).mean()+penalty
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite shared risk loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
        opt.step()
        dual = (dual.detach()+objective_config['dual_learning_rate']*excess.detach()).clamp(0,objective_config['dual_maximum'])
        step += 1
        if step == 1 or step%settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),
                primary_loss=float(torch.stack(primary).mean().detach()),
                regret=float(torch.stack(regret).mean().detach()),
                mean_soft_action=float(torch.stack(action).mean().detach()),
                source_raw_excess=risk.detach().tolist(),dual=dual.tolist())
            trace.append(row);heartbeat(state='training',mode=mode,**row)
        if step%settings['checkpoint_every'] == 0 or step == limit: save()
    return old.core.read_checkpoint(path)


def predict_shared(state, source, x, envelope):
    pr = state['source_preprocess'][source]
    old.parent.forest.causal_inputs(x,envelope,pr)
    model = old.parent.initialize(state['preprocess'],state['settings'],state['seed'])
    model.load_state_dict(state['model']);model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(x),4096):
            z = np.clip((x[start:start+4096]-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32)
            env = (envelope[start:start+4096]/pr['scale']).astype(np.float32)
            p,_ = model(torch.from_numpy(z),torch.from_numpy(env))
            out.append(p.numpy().astype(float)*pr['scale'])
    return np.concatenate(out)
