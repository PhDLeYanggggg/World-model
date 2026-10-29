"""Joint harm moments and donor-record-excluded blending on frozen queries."""
import hashlib
import statistics

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from src.evaluation import m3w_local_cost_analogue as previous

base = previous.base
JOINT_ARMS = ('coupled_cost', 'record_mean_blend', 'shuffled_alpha_blend', 'nested_support_blend')
MODES = ('independent_cost',) + JOINT_ARMS
ARMS = ('origin',) + MODES
PRIMARY = 'nested_support_blend'


def joint(mean, cap):
    """Exact equal-curvature quadratic optimum: 0 <= easy <= all <= cap."""
    mean, cap = np.asarray(mean, float), np.asarray(cap, float)
    if (mean.shape[-1] != 2 or cap.shape != mean.shape[:-1]
            or not np.isfinite(mean).all() or not np.isfinite(cap).all()
            or (cap < 0).any() or (cap > 1).any()):
        raise ValueError('Finite two-channel means and bounded causal caps required')
    ordered = np.where((mean[...,1] > mean[...,0])[...,None], mean.mean(-1)[...,None], mean)
    return np.minimum(np.maximum(ordered, 0), cap[...,None])


def nested_fit(target, squared_scale, source_count, record_count, recs, sources, caps):
    """Hold a donor's ENTIRE recording out of the local training subset."""
    different = recs[:,:,None] != recs[:,None,:]
    same_source = sources[:,:,None] == sources[:,None,:]
    counts = source_count[:,None,:] - same_source*record_count[:,:,None]
    if np.any(counts[different] <= 0):
        raise ValueError('Nested source count must exclude the full held recording')
    weights = np.divide(squared_scale[:,None,:], counts,
        out=np.zeros(counts.shape,float), where=different & (counts > 0))
    denominator = weights.sum(-1)
    means = np.divide(np.einsum('bij,bjc->bic',weights,target),denominator[...,None],
        out=np.zeros(target.shape,float),where=denominator[...,None] > 0)
    return joint(means,caps), (denominator > 0).all(-1), weights


def blend_coefficient(error, delta, recs, supported):
    counts = (recs[:,:,None] == recs[:,None,:]).sum(-1)
    weight = 1./counts; weight /= weight.sum(-1,keepdims=True)
    numerator = -(weight[...,None]*error*delta).sum((1,2))
    denominator = (weight[...,None]*delta**2).sum((1,2))
    alpha = np.clip(np.divide(numerator,denominator,out=np.zeros_like(numerator),
        where=(denominator > 1e-24) & supported),0,1)
    before = (weight[...,None]*error**2).sum((1,2))
    after = (weight[...,None]*(error+alpha[:,None,None]*delta)**2).sum((1,2))
    if np.any(after > before+1e-10*np.maximum(1,before)):
        raise AssertionError('Blend optimum failed its donor objective')
    return alpha, before, after


def retrieve(bank, query, frozen, *, k=32, heartbeat=lambda **_: None):
    old = previous.retrieve(bank,query,frozen,k=k,heartbeat=heartbeat)
    x, env, y, recs, eligible, scales, sources = (np.asarray(bank[key]) for key in
        ('x','envelope','labels','recordings','eligible','scale','sources'))
    query, frozen = np.asarray(query), np.asarray(frozen)
    if (np.any(frozen[eligible,4] > frozen[eligible,1]+1e-7)
            or np.any(frozen[eligible,0]+frozen[eligible,1] > env[eligible]+1e-7)
            or np.any(frozen[eligible][:,[0,1,4]] < -1e-7)):
        raise ValueError('Frozen moments violate joint constraints; do not silently repair')
    n=len(x); indices=old['indices']; position=np.full(n,-1,int);position[indices]=np.arange(len(indices))
    origin = frozen[indices][:,[1,4]]
    preds = {a:origin.copy() for a in JOINT_ARMS}
    preds['independent_cost']=old['predictions']['local_signed32']
    names=('alpha','mean_alpha','shuffled_alpha','record_ESS','max_record_weight','distinct_records',
           'nested_supported','nested_loss_before','nested_loss_after')
    diag={key:np.zeros(len(indices)) for key in names}
    diag.update(old['diagnostics'])
    causal=eligible & (env > 0); known=np.isfinite(y).all(1); selected=causal & known & query
    caps=np.zeros(n); caps[causal]=np.clip((env[causal]-frozen[causal,0])/env[causal],0,1)
    signed=np.zeros((n,2))
    signed[selected]=(y[selected][:,[1,4]]+.02*(frozen[selected][:,[2,3]]-y[selected][:,[2,3]]))/env[selected,None]
    _,rec_ids=np.unique(recs,return_inverse=True); _,source_ids=np.unique(sources,return_inverse=True)
    with threadpool_limits(limits=4):
        for receipt in old['references']:
            rec=receipt['recording']; ix=np.flatnonzero(query & (recs == rec)); p=position[ix]
            if not receipt['supported']: continue
            norm=np.flatnonzero(causal & (recs != rec)); donors=np.flatnonzero(selected & (recs != rec))
            mean=x[norm].astype(float).mean(0);std=np.maximum(x[norm].astype(float).std(0),1e-6)
            z=np.clip((x[donors]-mean)/std,-8,8).astype(np.float32)
            search=NearestNeighbors(n_neighbors=k,algorithm='brute',metric='euclidean',n_jobs=1).fit(z)
            counts=np.bincount(source_ids[donors],minlength=len(set(sources)))
            record_counts=np.bincount(rec_ids[donors],minlength=len(set(recs)))
            trace=hashlib.sha256()
            for start in range(0,len(ix),256):
                q=ix[start:start+256]; pos=position[q]
                _,near=search.kneighbors(np.clip((x[q]-mean)/std,-8,8).astype(np.float32))
                chosen=donors[near]; trace.update(chosen.tobytes())
                rs=rec_ids[chosen]; ss=source_ids[chosen]
                square=(env[chosen]/scales[chosen])**2; sc=counts[ss]
                w=square/sc; wn=w/w.sum(-1,keepdims=True)
                raw=(w[...,None]*signed[chosen]).sum(1)/w.sum(1)[:,None]
                # Compare raw arithmetic to the frozen independent solution before coupling.
                inherited=env[q,None]*np.clip(raw,0,1)
                if not np.array_equal(inherited,preds['independent_cost'][pos]):
                    raise AssertionError('Independent fit changed')
                coupled=env[q,None]*joint(raw,caps[q]); preds['coupled_cost'][pos]=coupled
                nested,support,_=nested_fit(signed[chosen],square,sc,record_counts[rs],rs,ss,caps[chosen])
                nested*=env[chosen,None]
                e=((frozen[chosen][:,:,[1,4]]-.02*frozen[chosen][:,:,[2,3]])-
                   (y[chosen][:,:,[1,4]]-.02*y[chosen][:,:,[2,3]]))/scales[chosen,None]
                delta=(nested-frozen[chosen][:,:,[1,4]])/scales[chosen,None]
                a,before,after=blend_coefficient(e,delta,rs,support)
                diag['alpha'][pos]=a;diag['nested_loss_before'][pos]=before;diag['nested_loss_after'][pos]=after
                diag['nested_supported'][pos]=support
                same=rs[:,:,None] == rs[:,None,:]
                masses=(same*wn[:,None,:]).sum(-1)
                diag['record_ESS'][pos]=1/(wn*masses).sum(-1)
                diag['max_record_weight'][pos]=masses.max(-1)
                diag['distinct_records'][pos]=(1/same.sum(-1)).sum(-1).round()
                preds[PRIMARY][pos]=origin[pos]+a[:,None]*(coupled-origin[pos])
                heartbeat(state='analogue_queries',recording=rec,completed=min(start+256,len(ix)),total=len(ix))
            if trace.hexdigest()!=receipt['neighbor_sha256']:
                raise AssertionError('Nested query neighbors changed')
            diag['mean_alpha'][p]=diag['alpha'][p].mean()
            salt=int.from_bytes(hashlib.sha256(('20261029|'+rec).encode()).digest()[:8],'little')
            diag['shuffled_alpha'][p]=diag['alpha'][np.random.default_rng(salt).permutation(p)]
            for arm,key in (('record_mean_blend','mean_alpha'),('shuffled_alpha_blend','shuffled_alpha')):
                preds[arm][p]=origin[p]+diag[key][p,None]*(preds['coupled_cost'][p]-origin[p])
    return dict(indices=indices,predictions=preds,supported=old['supported'],
        references=old['references'],diagnostics=diag,prior_donor_loss=old['donor_loss'])


def summarize(rows, expected):
    keys=[(r['context'],r['source'],r['seed']) for r in rows]
    if len(keys)!=72 or len(set(keys))!=72 or set(keys)!=set(expected):
        raise ValueError('Complete72 frozen views required')
    means={a:{t:statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in rows)
        if all(r['scores'][a][t]['full_cohort_MSE'] is not None for r in rows) else None
        for t in ('all','easy')} for a in ARMS}
    wins=0
    for context,seed in sorted({(r['context'],r['seed']) for r in rows}):
        group=[r for r in rows if (r['context'],r['seed'])==(context,seed)]
        if all(means[a]['easy'] is not None for a in ('origin','independent_cost',PRIMARY)):
            vals={a:statistics.fmean(r['scores'][a]['easy']['full_cohort_MSE'] for r in group)
                  for a in ('origin','independent_cost',PRIMARY)}
            wins+=vals[PRIMARY] < min(vals['origin'],vals['independent_cost'])-1e-12
    n,o,c=(means[a] for a in (PRIMARY,'origin','independent_cost'))
    uniform={t:statistics.fmean(r['prior_uniform_MSE'][t] for r in rows) for t in ('all','easy')}
    checks=dict(all_views_local_support=all(r['diagnostics']['local_supported']==r['scores']['origin']['rows'] for r in rows),
        easy_MSE_beats_origin=n['easy'] is not None and n['easy']<o['easy']-1e-12,
        easy_MSE_beats_independent=n['easy'] is not None and c['easy'] is not None and n['easy']<c['easy']-1e-12,
        easy_MSE_beats_uniform=n['easy'] is not None and n['easy']<uniform['easy']-1e-12,
        all_MSE_not_worse_than_independent=n['all'] is not None and c['all'] is not None and n['all']<=c['all']+1e-12,
        all_MSE_not_worse_than_uniform=n['all'] is not None and n['all']<=uniform['all']+1e-12,
        at_least12_of18_pair_wins=wins>=12,
        joint_moments_valid=all(r['diagnostics']['arms'][a]['easy_exceeds_all']==0 and
            r['scores'][a]['benefit_harm_envelope_exceedances']==0 for r in rows for a in JOINT_ARMS))
    attribution={a:n['easy'] is not None and n['easy']<means[a]['easy']-1e-12
                 for a in ('record_mean_blend','shuffled_alpha_blend')}
    return dict(views=72,equal_view_MSE=means,prior_uniform_MSE=uniform,pair_wins=int(wins),
        magnitude_screen=checks,magnitude_screen_pass=all(checks.values()),conditional_assignment_screen=attribution,
        occurrences={k:sum(r['scores'][PRIMARY][k] for r in rows) for k in ('rows','known','unknown','supported','unsupported')},
        fitting_diagnostic_only=True,whole_pipeline_OOF=False,held_inference=False,independent_roles_read=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)
