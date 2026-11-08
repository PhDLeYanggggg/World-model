"""Train new deterministic forecast trajectories with past-only road context."""
import argparse
from collections import defaultdict
from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_cv_abstention as parent

PUBLIC = parent.PUBLIC.parent/'opendd_velocity_candidate_v1'
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
ARMS = ('self','pedestrian','traffic')
STEPS, BATCH, SEEDS = 600,128,parent.SEEDS


def beat(phase, **extra):
    msg = dict(phase=phase,pid=os.getpid(),**extra)
    parent.prior.write_json(PRIVATE/'heartbeat.json',msg)
    print(json.dumps(msg),flush=True)


def register():
    assets = json.loads(parent.prior.SOURCE.read_text())['opendd']['sites']
    for a in assets:
        p = parent.prior.ARCHIVES/a['asset']['archive']
        if p.stat().st_size!=a['asset']['bytes'] or parent.prior.sha(p)!=a['asset']['sha256']:
            raise ValueError(f'Source changed: {p.name}')
    paths = (Path(__file__).resolve(),PUBLIC/'protocol.md',parent.prior.SOURCE,
        parent.prior.PUBLIC/'registration.json',parent.PUBLIC/'metrics.json',
        PUBLIC.parent/'opendd_query_rank_v1/analysis.json',Path(parent.__file__),
        ROOT/'src/world_model/m3w_opendd_velocity_candidate.py', ROOT/'src/world_model/m3w_ols_replacement.py',
        ROOT/'src/evaluation/m3w_opendd_cv_abstention.py', ROOT/'src/evaluation/m3w_opendd_location.py')
    p = PUBLIC/'registration.json'
    parent.prior.write_json(p,dict(files={str(a.relative_to(ROOT)):parent.prior.sha(a) for a in paths},
        arms=ARMS,seeds=SEEDS,steps=STEPS,batch=BATCH,checkpoint_every=100,
        official_risk_budget=.02,dronecrowd_confirmation_access=False,stage5c_executed=False,smc_enabled=False),immutable=True)
    return parent.prior.sha(p),assets


def read_traffic(conn, table, ticks):
    if not re.fullmatch(r'rdb[1-7]_\d+',table):
        raise ValueError('Unverified recording identifier')
    ticks = sorted(set(ticks))
    out = defaultdict(dict)
    # Restrict the database request to past timestamps before extracting positions.
    placeholders = ','.join('?' for _ in ticks)
    sql = (f'SELECT OBJID,TIMESTAMP,UTM_X,UTM_Y FROM "{table}" '
           "WHERE CLASS IN ('Car','Van','Truck','Motorcycle') "
           f'AND CAST(ROUND(TIMESTAMP*30000.0/1001.0) AS INTEGER) IN ({placeholders})')
    allowed = set(ticks)
    for ident,t,x,y in conn.execute(sql,ticks):
        index = round(t*30000./1001.)
        if index not in allowed or abs(index*1001./30000.-t)>6e-7 or not np.isfinite([x,y]).all():
            raise ValueError('Off-grid or non-past vehicle input')
        if ident in out[index]:
            raise ValueError(f'Duplicate vehicle at {table}/{index}')
        out[index][ident] = (x,y)
    return out


def build_context(site, archive, raw, registration):
    path = PRIVATE/f'{site}_context.npz'
    receipt = path.with_suffix('.json')
    if path.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        if old['registration']!=registration or old['sha256']!=parent.prior.sha(path):
            raise ValueError(f'Invalid context cache for {site}')
        with np.load(path,allow_pickle=False) as f:
            return f['context']
    beat('extract_past_traffic',site=site)
    h = raw['history']
    context = np.full((len(h),2,2,8,2),np.nan)
    groups = defaultdict(list)
    for k,rec in enumerate(raw['recording']):
        groups[str(rec)].append(k)
    with ZipFile(archive) as z, TemporaryDirectory(dir=PRIVATE) as tmp:
        members = [n for n in z.namelist() if Path(n).name==f'trajectories_{site}_v3.sqlite']
        if len(members)!=1:
            raise ValueError('Unverified trajectory DB layout')
        db = Path(tmp)/'tracks.sqlite'
        with z.open(members[0]) as src, db.open('wb') as dest:
            shutil.copyfileobj(src,dest,1024*1024)
        with closing(sqlite3.connect(f'{db.as_uri()}?mode=ro&immutable=1',uri=True)) as conn:
            conn.execute('PRAGMA trusted_schema=OFF')
            conn.execute('PRAGMA query_only=ON')
            for rec,query in groups.items():
                ticks = raw['start'][query,None]+12*np.arange(8)[None]
                road = read_traffic(conn,rec,ticks.ravel().tolist())
                for q,t in zip(query,ticks,strict=True):
                    lo,hi = raw['offsets'][q:q+2]
                    current = road[int(t[-1])]
                    ids = sorted(current)
                    car_xy = np.asarray([current[k] for k in ids],dtype=float).reshape(-1,2)
                    ego = h[lo:hi,-1]
                    peer_distance = np.linalg.norm(ego[:,None]-ego[None],axis=-1)
                    np.fill_diagonal(peer_distance,np.inf)
                    car_dist = np.linalg.norm(ego[:,None]-car_xy[None],axis=-1)
                    for i in range(hi-lo):
                        n = min(2,hi-lo-1)
                        peers = np.argsort(peer_distance[i],kind='stable')[:n]
                        context[lo+i,0,:n] = h[lo+peers]
                        for j,k in enumerate(np.argsort(car_dist[i],kind='stable')[:2]):
                            if car_dist[i,k] > 15.:
                                break
                            context[lo+i,1,j] = [road[int(f)].get(ids[k],(np.nan,np.nan)) for f in t]
    tmp = path.with_suffix('.pending')
    with tmp.open('wb') as f:
        np.savez(f,context=context)
    tmp.replace(path)
    parent.prior.write_json(receipt,dict(registration=registration,sha256=parent.prior.sha(path)),immutable=True)
    return context


def prepare(row, original, registration):
    site = row['site']
    d = parent.prepare(site,original)
    path = parent.prior.PRIVATE/f'{site}_cohort.npz'
    with np.load(path,allow_pickle=False) as source:
        raw = {k:source[k] for k in ('history','offsets','start','recording')}
    ctx = build_context(site,parent.prior.ARCHIVES/row['asset']['archive'],raw,registration)
    x,rot,scale = model.design(raw['history'],ctx)
    support = np.isfinite(ctx[:,1,:,-1]).all(-1).any(1)
    beat('past_context_verified',site=site,rows=len(x),motor_context_rows=int(support.sum()))
    return {**d,'design':torch.from_numpy(x),'rot':rot,'scale':scale,'motor':support,
        'future_local':torch.from_numpy(((d['y']@rot)/scale[:,None,None]).astype(np.float32)),
        'cv_local':torch.from_numpy(((d['cv']@rot)/scale[:,None,None]).astype(np.float32)),
        'valid_torch':torch.from_numpy(d['valid'])}


def features(d,arm):
    x = d['design'].clone()
    if arm=='self':
        x[:,17:] = 0.
    if arm=='pedestrian':
        x[:,-48:] = 0.
    return x


def fit(sources, held, seed, arm, registration):
    train = tuple(k for k in sources if k!=held)
    d = [sources[k] for k in train]
    xs = [features(a,arm) for a in d]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1776)
    net = model.network(xs[0].shape[1])
    opt = torch.optim.AdamW(net.parameters(),lr=.001)
    path = PRIVATE/f'{held}_{seed}_{arm}.pt'
    ident = dict(registration=registration,train=list(train),held=held,seed=seed,arm=arm)
    step,trace = 0,[]
    if path.exists():
        ck = torch.load(path,map_location='cpu',weights_only=True)
        if ck['identity']!=ident or ck['step'] not in range(100,STEPS+1,100):
            raise ValueError('Checkpoint identity mismatch')
        step,trace = ck['step'],ck['trace']
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state = ck['rng']
    for i in range(step,STEPS):
        draw = rng.integers(len(train),size=BATCH)
        picks = [(j,rng.choice(a['pool'],size=int((draw==j).sum()))) for j,a in enumerate(d)]
        x = torch.cat([xs[j][at] for j,at in picks])
        y,mask,cv = (torch.cat([d[j][name][at] for j,at in picks])
                     for name in ('future_local','valid_torch','cv_local'))
        loss = model.objective(model.decode(net(x)),y,mask,cv)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite trajectory objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i+1)%100==0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident,model=net.state_dict(),optimizer=opt.state_dict(),
                            step=i+1,trace=trace,rng=rng.bit_generator.state),tmp)
            tmp.replace(path)
            beat('checkpoint',held=held,arm=arm,seed=seed,step=i+1,loss=trace[-1])
    return net.eval(),dict(**ident,trace=trace)


def predict(net,d,arm):
    with torch.no_grad():
        x = model.decode(net(features(d,arm))).numpy().astype(float)
    return (x*d['scale'][:,None,None])@d['rot'].transpose(0,2,1)


def oracle(d,bank):
    y,valid,cv = d['y'],d['valid'],d['cv']
    bound = np.where(valid[:,None],np.linalg.norm(cv-y,axis=-1)[:,None]-np.linalg.norm(bank-y[:,None],axis=-1),
                     -np.linalg.norm(bank-cv[:,None],axis=-1)).sum(2)
    # Do not enlarge the causal query roster when adding a forecast family.
    fixed = np.maximum.reduceat(d['legal'].any(1),d['offsets'][:-1])
    legal = (np.linalg.norm(bank-cv[:,None],axis=-1)>1e-9).any(2)
    return score.edit({**d,'bank':bank,'legal':legal},bound,allowed_queries=fixed)


def compare(d,pred,ref,cut,subset=None):
    if subset is not None:
        if not d['valid'][subset].any():
            return None
        d = {k:d[k][subset] for k in ('cv','y','valid')}
        pred,ref = pred[subset],ref[subset]
        if np.nansum(np.linalg.norm(d['cv']-d['y'],axis=-1)) <= 0.:
            return None
    row = score.metrics(d,pred,np.zeros(0,dtype=bool),cut,ref=ref)
    if not np.isfinite(row['fde']):
        row['fde'] = None
    return row


def evaluate(d,nets,cut):
    bank = d['bank']
    ref, support = oracle(d,bank)
    output = {'cv':compare(d,d['cv'],d['cv'],cut),'ols4':compare(d,d['ols'],d['cv'],cut),
              'oracle_old':score.metrics(d,ref,support,cut)}
    predictions = {arm:predict(net,d,arm) for arm,net in nets.items()}
    for arm,p in predictions.items():
        output[arm] = compare(d,p,d['cv'],cut)
        output[f'{arm}_vs_ols4'] = compare(d,p,d['ols'],cut)
        output[f'{arm}_quarter'] = compare(d,d['cv']+.25*(p-d['cv']),d['cv'],cut)
        p_new,new_support = oracle(d,np.concatenate((bank,p[:,None]),axis=1))
        if not np.array_equal(new_support,support):
            raise ValueError('Expanded menu changed oracle roster')
        output[f'oracle_add_{arm}'] = score.metrics(d,p_new,new_support,cut,ref=ref)
    for arm,control in (('traffic','pedestrian'),('pedestrian','self')):
        output[f'{arm}_vs_{control}'] = compare(d,predictions[arm],predictions[control],cut)
        output[f'{arm}_vs_{control}_motor_supported'] = compare(d,predictions[arm],predictions[control],cut,d['motor'])
    return output


def summary(sites):
    out = {}
    for key in next(iter(sites.values()))[0]['arms']:
        values = [[r['arms'][key] for r in seeds if r['arms'][key] is not None] for seeds in sites.values()]
        flat = [r for rs in values for r in rs]
        v = np.asarray([np.mean([r['lower_pp'] for r in rs]) for rs in values if rs])
        if not len(v):
            out[key] = None
            continue
        draws = np.random.default_rng(817).integers(len(v),size=(3000,len(v)))
        fde = [r['fde'] for r in flat if r['fde'] is not None]
        out[key] = dict(sites=len(v),lower_pp=float(v.mean()),
            ci95_pp=np.quantile(v[draws].mean(1),[.025,.975]).tolist(),positive_sites=int((v>0).sum()),
            ade=float(np.mean([r['ade'] for r in flat])),fde=float(np.mean(fde)) if fde else None,
            worst_site_pp=float(v.min()),worst_easy_pct=max((r['signed_easy_pct'] for r in flat
                     if r['signed_easy_pct'] is not None),default=None))
    return out


def run(registration,assets):
    begin = monotonic()
    prior_id = parent.prior.sha(parent.prior.PUBLIC/'registration.json')
    sources = {a['site']:prepare(a,prior_id,registration) for a in assets}
    fits,nets = [],{}
    for held in sources:
        for seed in SEEDS:
            nets[held,seed] = {}
            for arm in ARMS:
                net,meta = fit(sources,held,seed,arm,registration)
                fits.append(meta)
                nets[held,seed][arm] = net
    result = {held:[dict(seed=seed,arms=evaluate(d,nets[held,seed],
        parent.easy_cut(sources,[s for s in sources if s!=held]))) for seed in SEEDS] for held,d in sources.items()}
    p = PUBLIC/'analysis.json'
    replay = p.exists()
    parent.prior.write_json(p,dict(registration=registration,fits=fits,sites=result,summary=summary(result),
        support={s:dict(rows=len(d['y']),motor_rows=int(d['motor'].sum())) for s,d in sources.items()},
        runtime=dict(architecture=platform.machine(),torch=str(torch.__version__),threads=4,workers=0),
        scope='development',independent_confirmation=False,official_risk_pass=False),immutable=True)
    beat('complete',elapsed_s=round(monotonic()-begin,3),analysis_sha256=parent.prior.sha(p),exact_replay=replay)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only',action='store_true')
    args = parser.parse_args()
    if sys.platform=='darwin' and platform.machine()!='arm64':
        raise SystemExit('Run native arm64 .venv-pytorch/bin/python, not Rosetta')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registration,assets = register()
        if args.register_only:
            print(registration)
        else:
            import numpy as np
            import torch
            from src.evaluation import m3w_opendd_cv_abstention as score
            from src.world_model import m3w_ols_replacement as motion, m3w_opendd_velocity_candidate as model

            parent.np,parent.torch,parent.motion = np,torch,motion
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            run(registration,assets)
