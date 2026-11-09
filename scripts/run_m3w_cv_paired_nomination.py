"""Train matched pointwise and within-agent conservative ranking objectives."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_screen_order as screen

base, prior, model = screen.base, screen.prior, screen.model
np, torch, read, sha, once, stable, artifact, checked = (
    base.np, base.torch, base.read, base.sha, base.once, base.stable, base.artifact, base.checked)
OUT = screen.OUT.parent/'moving_cv_paired_nomination_v1'
PRIVATE = screen.PRIVATE.parent/OUT.name
STEPS = 1000
ARMS = ('point', 'paired')
STRATEGIES = ('frozen_match','point_match','paired_match','point_calibrated','paired_calibrated')
RANK_WEIGHT, TEMPERATURE = .25, .15


def signed_preference(a, b, target, valid, scale):
    """Positive only when a wins for every completion of the missing steps."""
    y = np.where(valid[..., None], target, 0.)
    known = np.where(valid, np.linalg.norm(b-y, axis=-1)-np.linalg.norm(a-y, axis=-1), 0.).sum(1)
    debit = np.where(valid, 0., np.linalg.norm(b-a, axis=-1)).sum(1)
    return np.sign(known)*np.maximum(np.abs(known)-debit, 0.)/(valid.shape[1]*scale)


def features(past, cv, ols, candidate, scale):
    return np.concatenate((model.features(past,candidate-cv,scale),
        np.arcsinh((candidate-ols)/scale[:,None,None]).reshape(len(past),-1)), axis=1).astype(np.float32)


def sampler(cv, ols, bank, d):
    past, _ = prior.study.motion.features(d['history'], 'prefix')
    scale = prior.model.scale(d['history'])
    groups = [np.flatnonzero(d['sites'] == k) for k in sorted(set(d['sites']))]

    def sample(rng):
        i = prior.draw(groups, rng)
        a = rng.integers(bank.shape[1],size=len(i))
        b = (a+rng.integers(1,bank.shape[1],size=len(i))) % bank.shape[1]
        p = bank[i[:,None],np.column_stack((a,b))]
        c,o,y,v,s = cv[i],ols[i],d['target'][i],d['valid'][i],scale[i]
        _, _, gain, _ = prior.utility.candidate_terms(c,o,p,y,v)
        x = np.stack([features(past[i],c,o,p[:,j],s) for j in range(2)],axis=1)
        preference = signed_preference(p[:,0],p[:,1],y,v,s)
        return tuple(map(torch.from_numpy, (x,(gain/(v.shape[1]*s[:,None,None])).clip(-8.,8.).astype(np.float32),
                                            preference.astype(np.float32))))
    return sample, past.shape[1]+49


def objective(net, x, gain, preference, weight):
    f = torch.nn.functional
    pred = net(x)
    point = f.smooth_l1_loss(pred,gain)
    delta = (pred[:,0]-pred[:,1]).mean(-1)
    paired = (preference.abs().clamp(max=1.)*f.softplus(-preference.sign()*delta/TEMPERATURE)).mean()
    return point + weight*paired


@torch.inference_mode()
def forecast(net, cv, ols, bank, h):
    past, _ = prior.study.motion.features(h,'prefix')
    scale = prior.model.scale(h)
    score = np.empty(bank.shape[:2])
    for j in range(bank.shape[1]):
        for start in range(0,len(h),4096):
            ix = slice(start,start+4096)
            x = features(past[ix],cv[ix],ols[ix],bank[ix,j],scale[ix])
            score[ix,j] = net(torch.from_numpy(x)).mean(-1).numpy()
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite candidate score')
    return score


def nomination(score, pred, gate):
    gain, risk, easy, legal = pred
    eligible = model.admissible(gain,risk,easy,legal,*gate)
    return screen.nominate((score,risk,easy,eligible))


def calibration(score, pred, d, cv, ols, bank, cut):
    groups = prior.query_groups(d['query'])
    sources = sorted(set(d['sites']))
    ladder = []
    for t,p in base.GRID:
        action = screen.choose(nomination(score,pred,(t,p)),groups)
        rows = base.metrics(cv,ols,bank,d,{'strict':action},cut)
        gain = [[prior.prior.parent.readout.merge_comparisons([r['comparisons'][f'strict_vs_{k}']
                 for r in rows if r['source'] == s])['completion_lower_pp'] for k in ('cv','ols4')] for s in sources]
        ladder.append(dict(tolerance=t,min_easy=p,edits=int((action >= 0).sum()),
            passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),
            gain_pp=np.mean(gain,axis=0).tolist()))
    return dict(setting=model.choose_setting(ladder,len(np.unique(d['recording']))),ladder=ladder)


def train_cell(old, old_choices, d, donor, reg, beat, replay):
    ident = old['identity']
    fold,seed = ident['fold'],ident['seed']
    roles = [set(ident[k]) for k in ('train','calibration','held')]
    if set.union(*roles) != set(d['sites']) or sum(map(len,roles)) != len(set(d['sites'])):
        raise ValueError('Fitting roles must be disjoint and exhaustive')
    _,train = base.subset(d,ident['train'])
    _,cal = base.subset(d,ident['calibration'])
    hi,held = base.subset(d,ident['held'])
    h = train['history']
    a = prior.study.motion.bank(h)
    draw,width = sampler(a[:,0],a[:,1],base.expanded(a[:,0],a[:,1:]),train)
    del a
    cut = old['cut']
    gate = screen.load_head(old['fits'][0],width-24,8)
    c = prior.study.motion.bank(cal['history'])
    cb = base.expanded(c[:,0],c[:,1:])
    cp = screen.predictions(gate,c[:,0],cb,cal['history'])
    cv,ols,raw = prior.bank(held['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    bank = base.expanded(cv,raw)
    pred = screen.predictions(gate,cv,bank,held['history'])
    groups = prior.query_groups(held['query'])
    with np.load(checked(old_choices['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],held['ids'])
        control = z['screen_match']
        np.testing.assert_array_equal(control,screen.choose(screen.nominate(pred,screen.RELAXED),groups,z['before']))
    actions, fits, selections, rng_state, updates = {'frozen_match':control}, [], {}, None, 0
    for arm, weight in zip(ARMS,(0.,RANK_WEIGHT),strict=True):
        name = f'{arm}_{fold}_{seed}'
        identity = dict(ident, arm=arm, registration=reg, weight=weight)
        net, fit, u = prior.prefix.trainer.fit(identity,name,seed,lambda:prior.model.head(width,2),STEPS,
            draw,lambda net,x,g,p:objective(net,x,g,p,weight),replay,beat)
        state = torch.load(checked(fit['artifact']),weights_only=True,map_location='cpu')['rng']
        if rng_state is not None and state != rng_state:
            raise ValueError('Matched training draws diverged')
        rng_state = state
        fits.append(fit)
        updates += u
        selected = calibration(forecast(net,c[:,0],c[:,1],cb,cal['history']),cp,cal,c[:,0],c[:,1],cb,cut)
        selections[arm] = selected
        held_score = forecast(net,cv,ols,bank,held['history'])
        actions[arm+'_match'] = screen.choose(nomination(held_score,pred,screen.RELAXED),groups,control)
        setting = selected['setting']
        actions[arm+'_calibrated'] = (np.full_like(control,-1) if setting is None
            else screen.choose(nomination(held_score,pred,setting),groups))
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',dict(ids=held['ids'],**actions))
    return dict(identity=ident,cut=cut,actions=path,fits=fits,calibration=selections), updates


def evaluate(cells, d, donor, beat):
    scorer = prior.prior.parent.readout
    pairs = {f'{k}_vs_{label}':(v,k) for k in STRATEGIES for label,v in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(paired_vs_point=('point_match','paired_match'),point_vs_frozen=('frozen_match','point_match'),
                 paired_vs_frozen=('frozen_match','paired_match'))
    rows = []
    local = np.zeros(4,np.int64)
    for cell in cells:
        ident = cell['identity']
        hi,held = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(held['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        with np.load(checked(cell['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],held['ids'])
            actions = {k:z[k] for k in STRATEGIES}
        for group in prior.query_groups(held['query']):
            count = np.count_nonzero(actions['frozen_match'][group]>=0)
            if any(np.count_nonzero(actions[k][group]>=0) != count for k in STRATEGIES[1:3]):
                raise ValueError('Query budget changed')
        same = (actions['point_match']>=0) & (actions['paired_match']>=0)
        diff = same & (actions['point_match'] != actions['paired_match'])
        shared = np.flatnonzero(diff)
        preference = signed_preference(bank[shared,actions['paired_match'][shared]],
            bank[shared,actions['point_match'][shared]],held['target'][shared],held['valid'][shared],np.ones(len(shared)))
        local += (same.sum(),diff.sum(),(preference>0).sum(),(preference<0).sum())
        for rec in np.unique(held['recording']):
            ix = held['recording']==rec
            q,y,v = (held[k][ix] for k in ('query','target','valid'))
            c = cv[ix]
            paths = {base.CV:c,base.OLS:ols[ix],**{k:prior.apply(c,bank[ix],a[ix]) for k,a in actions.items()}}
            r = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=cell['cut'][0],hard_cut=cell['cut'][1])
            _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
            comparisons = {k:prior.study.motion.full_masses(paths[a],paths[b],c,y,v,1./n[inv])
                           for k,(a,b) in pairs.items()}
            source, = np.unique(held['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=comparisons))
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    result = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
    if (result['views'],result['source_groups'],result['risk']['frozen_match']['switches']) != (72,9,2317):
        raise ValueError('Expected the frozen fitting-source cohort')
    return result, dict(zip(('shared_recipients','changed_shared_paths','paired_definite_wins','point_definite_wins'),
                            map(int,local),strict=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

        def beat(**v):
            info = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',info)
            if v.get('phase') and v.get('step',STEPS) in (100,STEPS):
                print(info,flush=True)

        parent_reg,donor,_ = base.register()
        parent, choices = base.OUT/'freeze.json',screen.OUT/'freeze.json'
        a,b = read(parent),read(choices)
        if (a['registration'] != parent_reg or not a['before_held_scoring'] or not b['before_held_scoring']
                or b['registration'] != sha(screen.OUT/'registration.json')
                or len(a['cells']) != 9 or len(b['cells']) != 9):
            raise ValueError('Expected both parent freezes')
        registration = OUT/'registration.json'
        once(registration,stable(dict(parents=[artifact(parent),artifact(choices)],
            files=[artifact(p) for p in (Path(__file__),OUT/'protocol.md',Path(screen.__file__),
                                         Path(prior.prefix.trainer.__file__))],
            steps=STEPS,seeds=base.SEEDS,weight=RANK_WEIGHT,temperature=TEMPERATURE,
            original_2pct=True,independent_confirmation=False)))
        reg = sha(registration)
        data = base.fit_population(beat)
        cells,updates = [],0
        for old,old_choice in zip(a['cells'],b['cells'],strict=True):
            if old['identity'] != old_choice['identity']:
                raise ValueError('Parent cells do not align')
            cell,u = train_cell(old,old_choice,data,donor,reg,beat,args.replay)
            cells.append(cell)
            updates += u
            beat(phase='cell_frozen',fold=old['identity']['fold'],seed=old['identity']['seed'],updates=updates)
        freeze = OUT/'freeze.json'
        once(freeze,stable(dict(registration=reg,cells=cells,before_held_scoring=True,optimizer_steps=18*STEPS)))
        summary,local = evaluate(read(freeze)['cells'],data,donor,beat)
        qualification = {k:sum(c['calibration'][k]['setting'] is not None for c in cells) for k in ARMS}
        maxima = {k:[max(v['passes'] for v in c['calibration'][k]['ladder']) for c in cells] for k in ARMS}
        once(OUT/'readout.json',stable(dict(freeze=sha(freeze),summary=summary,shared_path_comparison=local,
            qualifying_calibration_cells=qualification,calibration_max_passes=maxima,
            original_2pct=True,a_labels_opened=False,independent_confirmation=False)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__ == '__main__':
    main()
