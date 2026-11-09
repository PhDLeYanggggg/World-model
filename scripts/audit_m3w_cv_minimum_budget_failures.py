"""Separate recipient and displacement failures of the frozen teacher ablation."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_minimum_budget as study

exp,base,prior,np,torch = study.exp,study.base,study.prior,study.np,study.torch
ARMS = ('cost','matched_teacher','fixed_recipients','clipped_radius')


def align(cv,old,new,selected):
    radius = lambda x:np.linalg.norm(x-cv,axis=-1).max(1)
    a,b = radius(old),radius(new)
    if (a[selected]<=0.).any() or (b[selected]<=0.).any():
        raise ValueError('Matched recipient requires a nontrivial action')
    f = np.ones(len(cv))
    f[selected] = np.minimum(1.,a[selected]/b[selected])
    path = cv+f[:,None,None]*(new-cv)
    if (radius(path)[selected]>a[selected]+1e-9).any():
        raise ValueError('Clipped path exceeded matched reference displacement')
    return path


def load_head(fit,width):
    z = torch.load(exp.checked(fit['artifact']),map_location='cpu',weights_only=True)
    if z['identity']!=fit['identity'] or z['step']!=study.STEPS:
        raise ValueError('Unexpected teacher checkpoint identity')
    net = prior.model.head(width,6)
    net.load_state_dict(z['net'])
    return net.eval()


def main():
    torch.set_num_threads(4)
    f,r = [study.OUT/k for k in ('training_freeze.json','training_readout.json')]
    frozen,registered = exp.read(f),exp.read(r)
    if (registered['freeze']!=exp.sha(f) or registered['a_labels_opened']
            or registered['independent_confirmation'] or not registered['original_2pct']):
        raise ValueError('Require frozen development-only ablation')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    scorer = prior.prior.parent.readout
    rows,masses = [],{}
    for c in frozen['cells']:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        w = study.context(h)[0].shape[1]+37
        net = load_head(c['fits'][1],w)
        p,legal,s = study.infer({'teacher':net},h,cv,ols,bank)
        with np.load(exp.checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a,b = z['cost'],z['teacher_matched']
        chosen = a>=0
        g = study.decode(p,legal,[np.array([i]) for i in np.flatnonzero(chosen)],s,
                         'teacher',.25,1.,False,match=a)
        np.testing.assert_array_equal(g>=0,chosen)
        paths = {k:exp.slate.path_for(cv,bank,x) for k,x in zip(ARMS[:3],(a,b,g),strict=True)}
        paths['clipped_radius'] = align(cv,paths['cost'],paths['fixed_recipients'],chosen)
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            local = {k:x[ix] for k,x in paths.items()}
            local.update({base.CV:cv[ix],base.OLS:ols[ix]})
            weight = exp.weight(q)
            score = scorer.score_recording(local,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            terms = {k:prior.study.motion.full_masses(local['cost'],local[k],cv[ix],y,v,weight) for k in ARMS[1:]}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(score,source=str(source),seed=seed,comparisons=terms))
            width = np.where(v,0.,np.linalg.norm(local['cost']-local['matched_teacher'],axis=-1))
            change = chosen[ix]^(b[ix]>=0)
            m = masses.setdefault((str(source),seed),np.zeros(4))
            m += [terms['matched_teacher']['reference'],weight[change]@width[change].sum(1),
                  weight[~change]@width[~change].sum(1),np.count_nonzero(change)]
    summary = prior.prefix.grouped.grouped_readout(rows,{z['recording']:z['source'] for z in rows},base.SEEDS)
    old = registered['summary']['effects']['teacher_at_match']
    if summary['effects']['matched_teacher']!=old:
        raise ValueError('Unchanged teacher contrast failed exact replay')
    m = np.array(list(masses.values()))
    parts = (100*m[:,1:3]/m[:,:1]).mean(0)
    if not np.isclose(parts.sum(),old['known_pp']['point']-old['completion_lower_pp']['point'],atol=1e-9):
        raise ValueError('Recipient and shape debits do not sum to paired difference')
    out = dict(parent=exp.artifact(r),code=exp.artifact(Path(__file__)),summary=summary,
        debit_recipient_change_pp=float(parts[0]),debit_shared_recipient_pp=float(parts[1]),
        recipient_symmetric_difference=int(m[:,3].sum()),diagnostic_only=True,
        a_labels_opened=False,independent_confirmation=False,original_2pct=True)
    exp.once(study.OUT/'failure_diagnosis.json',exp.stable(out))
    print({k:(v['known_pp']['point'],v['completion_lower_pp']['point']) for k,v in summary['effects'].items()},flush=True)


if __name__=='__main__':
    main()
