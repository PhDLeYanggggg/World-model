"""Membership probability diagnostics; no deployment threshold selection."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from src.evaluation.m3w_harm_tail_diagnostics import weights_for_sites


def summarize(prob,y,sites,subset=None):
    prob,y,sites=map(np.asarray,(prob,y,sites)); known=np.isfinite(y)
    if prob.shape!=y.shape or sites.shape!=y.shape or not np.isfinite(prob).all() or (prob<0).any() or (prob>1).any():
        raise ValueError('Aligned probabilities and labels required')
    if not np.isin(y[known],[0,1]).all(): raise ValueError('Binary/unknown targets required')
    if subset is not None: known &= np.asarray(subset,bool)
    if not known.any(): return dict(status='not_estimable',reason='no_known_rows')
    p,t,s=prob[known].astype(float),y[known].astype(float),sites[known]
    w=weights_for_sites(np.ones(len(s),bool),s); rate=float(w@t)
    q=np.clip(p,1e-7,1-1e-7); index=np.minimum((p*10).astype(int),9); bins=[]
    for i in range(10):
        use=index==i; mass=float(w[use].sum())
        bins.append(dict(index=i,rows=int(use.sum()),mass=mass,
            predicted=float(w[use]@p[use]/mass) if mass else None,observed=float(w[use]@t[use]/mass) if mass else None))
    return dict(rows=len(t),positive=int(t.sum()),negative=int((1-t).sum()),positive_rate=rate,
        stable_support=int(t.sum())>=20 and int((1-t).sum())>=20,
        Brier=float(w@((p-t)**2)),log_loss=float(-w@(t*np.log(q)+(1-t)*np.log1p(-q))),
        AUROC=float(roc_auc_score(t,p,sample_weight=w)) if 0<rate<1 else None,
        AUPRC=float(average_precision_score(t,p,sample_weight=w)) if rate>0 else None,
        ECE=sum(b['mass']*abs(b['predicted']-b['observed']) for b in bins if b['mass']),
        mean_probability=float(w@p),bins=bins)


def paired(rows,cfg):
    out={}
    for pair in cfg['pairs']:
        out[pair]={}
        for a,b in sorted({(r['producer'],r['controller']) for r in rows}):
            group=[r for r in rows if r['pair']==pair and (r['producer'],r['controller'])==(a,b)]
            assert len(group)==3 and sorted(r['seed'] for r in group)==cfg['seeds']
            sites=sorted({f['held'] for r in group for f in r['folds']}); assert len(sites)==4
            result={}
            for arm in cfg['arms']:
                for subset in ('all','envelope_positive'):
                    for key in ('Brier_skill_percent','AUROC_above_chance','log_loss_gain','ECE','AUROC_over_reference_proxy'):
                        values=[]
                        for site in sites:
                            vals=[]
                            for r in group:
                                f=next(f for f in r['folds'] if f['held']==site); m=f['metrics'][arm][subset]
                                ref=f['metrics']['train_constant'][subset]; proxy=f['metrics']['reference_ratio'][subset]
                                if any(v.get('status')=='not_estimable' for v in (m,ref,proxy)):
                                    vals.append(None); continue
                                if key=='Brier_skill_percent': value=100*(1-m['Brier']/ref['Brier']) if ref['Brier']>0 else None
                                elif key=='AUROC_above_chance': value=m['AUROC']-.5 if m['AUROC'] is not None else None
                                elif key=='AUROC_over_reference_proxy': value=m['AUROC']-proxy['AUROC'] if m['AUROC'] is not None and proxy['AUROC'] is not None else None
                                elif key=='log_loss_gain': value=ref['log_loss']-m['log_loss']
                                else: value=m['ECE']
                                vals.append(value)
                            if any(v is None or not np.isfinite(v) for v in vals): break
                            values.append(float(np.mean(vals)))
                        name=arm+'__'+subset+'__'+key
                        if len(values)!=4: result[name]=dict(status='not_estimable',reason='missing_support_not_dropped'); continue
                        v=np.array(values); rng=np.random.default_rng(cfg['bootstrap_seed'])
                        draws=v[rng.integers(0,4,size=(cfg['bootstrap_resamples'],4))].mean(1)
                        result[name]=dict(point=float(v.mean()),CI=np.quantile(draws,[.025,.975]).tolist(),locality_points=dict(zip(sites,values)))
            out[pair][f'producer{a}_controller{b}']=result
    return out


def gates(contrasts):
    arms={}
    for arm in ('linear','mlp'):
        arms[arm]={}
        for metric in ('Brier_skill_percent','AUROC_above_chance','log_loss_gain'):
            vs=[r[arm+'__envelope_positive__'+metric] for r in contrasts['full'].values()]
            arms[arm][metric]=dict(positive=sum(v.get('CI',[0,0])[0]>0 for v in vs),
                negative=sum(v.get('CI',[0,0])[1]<0 for v in vs),not_estimable=sum('CI' not in v for v in vs))
        arms[arm]['membership_signal_gate']=all(arms[arm][k]['positive']==6 for k in ('Brier_skill_percent','AUROC_above_chance','log_loss_gain'))
    return dict(arms=arms,membership_signal_established=arms['mlp']['membership_signal_gate'],
        policy_advance=False,deployment_changed=False,independent_confirmation=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
