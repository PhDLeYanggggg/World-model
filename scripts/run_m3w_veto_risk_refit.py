"""Compare all-action versus veto-targeted risk residuals with nested exclusions."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_expanded_risk_support as base
from src.world_model import m3w_veto_risk_refit as model

run,np,prior = base.run,base.np,base.prior
PUBLIC,PRIVATE = prior.PUBLIC/'veto_refit',prior.PRIVATE/'veto_refit'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_veto_risk_refit.py')
ARMS = ('original','all_actions','veto_only')


def register():
    parents = (base.REG,base.PUBLIC/'analysis.json',base.PUBLIC/'freeze.json',
        base.PUBLIC/'replay_execution.json',base.PUBLIC/'observed_reference_opportunities.json',
        Path(base.__file__),prior.REG,prior.PUBLIC/'freeze.json',prior.PUBLIC/'analysis.json',
        run.REG,run.PUBLIC/'freeze.json',Path(prior.model.__file__),Path(run.model.__file__),
        Path(prior.nested.base.__file__),Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],arms=ARMS,seeds=run.kin.prior.SEEDS,
        paired_draws=3000,original_budget=.02,matched_counts=True,
        nested_double_exclusion=True,raw_count_witness_preserved=True,
        independent_outcomes_opened=False)))


def fit(d,refs,records,beat):
    ids,sites = d['global_ids'],d['sites']
    splits = {s:np.flatnonzero(sites==s) for s in np.unique(sites)}
    by_source = {s:[] for s in splits}
    for ri,g in records.values():
        s = sites[ri[0]]
        by_source[s].append((ri,g,np.searchsorted(splits[s],ri)))
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    original = {c['seed']:c for c in run.read(prior.PUBLIC/'freeze.json')['seeds']}
    heads = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    cells = []
    for seed in run.kin.prior.SEEDS:
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(original[seed]['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            raw,old = z['choices'][[0,2]]
        with np.load(run.checked(heads[seed]['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            old_coef = dict(zip(z['names'],z['coefficients']))
        data,m = {},{}
        for s,ix in splits.items():
            b,c = bank[ix],cv[ix]
            x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            y,valid = labels['target_eval'][ix]-d['origin'][ix,None],labels['valid'][ix]
            _,a,e = base.nested.base.teacher.actual(b,c,y,valid,cuts[s]['easy'])
            data[s] = (b,c,x,scale,a,e)
            m[s] = run.model.fitting_moments(x,b,c,y,valid,scale,1./d['crowd'][ix],cuts[s]['easy'])
        choices = np.full((len(ARMS),len(ids)),-1,np.int16)
        choices[0] = old
        releases = np.zeros((2,*bank.shape[:2]),bool)
        summary = []
        for held,ix in splits.items():
            mark = PRIVATE/f'{seed}_{held}.json'
            ident = dict(seed=seed,held=str(held),registration=run.sha(REG))
            if not mark.exists():
                donor_stats = []
                counts = np.zeros(2,dtype=int)
                for donor in splits:
                    if donor==held:
                        continue
                    b,c,x,s,true_a,true_e = data[donor]
                    coef = run.model.coefficients([v for t,v in m.items() if t not in (held,donor)])
                    u,a,e,legal = run.model.decision_costs(x,coef,b,c,s)
                    slack = np.empty((len(c),2))
                    for _,g,j in by_source[donor]:
                        w = prior.nested.base.select(g,u[j],a[j],e[j],s[j],legal[j])
                        n = [int((w[h]>=0).sum()) for h in g]
                        ledger,_ = prior.model.stream(g,u[j]/s[j,None],a[j],e[j],legal[j],
                                                      counts=n,witness=w)
                        slack[j] = model.prefix(g,a[j],e[j],ledger)
                    z,veto = model.descriptors(x,s,a,e,slack,legal)
                    y = np.stack((true_a-a,true_e-e),axis=-1)/s[:,None,None]
                    w = 1./d['crowd'][splits[donor]]
                    donor_stats.append((model.moments(z,y,legal,w),model.moments(z,y,veto,w)))
                    counts += [legal.sum(),veto.sum()]
                coef = np.stack([model.coefficients([v[j] for v in donor_stats]) for j in range(2)])
                ch = np.full((2,len(ix)),-1,np.int16)
                released = np.zeros((2,len(ix),bank.shape[1]),bool)
                for ri,g,j in by_source[held]:
                    b,c = bank[ri],cv[ri]
                    x,scale = run.model.descriptors(d['history'][ri],b,c,d['crowd'][ri])
                    u,a,e,ok = run.model.decision_costs(x,old_coef[held],b,c,scale)
                    slack = model.prefix(g,a,e,old[ri])
                    x,veto = model.descriptors(x,scale,a,e,slack,ok)
                    radius = np.linalg.norm(b-c[:,None],axis=-1).max(2)
                    n = [(raw[ri[h]]>=0).sum() for h in g]
                    for k,w in enumerate(coef):
                        aa,ee,r = model.reopen(x,w,scale,radius,a,e,slack,veto,raw[ri])
                        ch[k,j],_ = prior.model.stream(g,u/scale[:,None],aa,ee,ok,counts=n,witness=raw[ri])
                        released[k,j] = r
                path = mark.with_suffix('.npz')
                base.nested.base.parent.base.write_array(path,ids=ids[ix],choices=ch,released=released,
                    coefficients=coef)
                run.once(mark,dict(identity=ident,cache=run.artifact(path),donor_counts=counts.tolist()))
            c = run.read(mark)
            if c['identity']!=ident:
                raise ValueError('Saved fold identity mismatch')
            with np.load(run.checked(c['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids[ix])
                choices[1:,ix],releases[:,ix] = z['choices'],z['released']
            summary.append(c['donor_counts'])
            beat(state='held_source_frozen',seed=seed,fold=len(summary))
        path = PRIVATE/f'{seed}_all.npz'
        if not path.exists():
            base.nested.base.parent.base.write_array(path,ids=ids,choices=choices,releases=releases)
        else:
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                np.testing.assert_array_equal(z['choices'],choices)
                np.testing.assert_array_equal(z['releases'],releases)
        cells.append(dict(seed=seed,cache=run.artifact(path),donor_counts=summary))
    run.once(PUBLIC/'freeze.json',dict(cells=cells,decisions_frozen_before_scoring=True))


def score(d,refs,records,beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['decisions_frozen_before_scoring']:
        raise ValueError('All three frozen seed choices required')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(veto_vs_all=('all_actions','veto_only',p.CV),
        veto_vs_original=('original','veto_only',p.CV),all_vs_original=('original','all_actions',p.CV))
    old = {(r['seed'],r['recording']):r for r in run.read(base.PUBLIC/'analysis.json')['records']}
    results = []
    for c0 in f['cells']:
        seed = c0['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c0['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            ch,released = z['choices'],z['releases']
        for r,(ix,g) in records.items():
            s = str(d['sites'][ix[0]])
            c,b,v = cv[ix],bank[ix],ch[:,ix]
            for j in g:
                n = (v[:,j]>=0).sum(1)
                if not (n==n[0]).all():
                    raise ValueError('Exact count changed')
            paths = {a:run.model.selected_paths(z,b,c) for a,z in zip(ARMS,v)}
            out = p.readout.score_recording({p.CV:c,**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            parent = old[seed,r]
            if out['metrics']['original']!=parent['metrics']['original']:
                raise ValueError('The original control failed replay')
            limited = 'no_easy_reference_in_any_option' in parent['fixed_prefix_bound']['failed_necessary_conditions']
            active = np.zeros(len(ix),bool)
            for j in g:
                active[j] = (v[0,j]>=0).any()
            _,a,e = base.nested.base.teacher.actual(b,c,y[ix],valid[ix],cuts[s]['easy'])
            ref,_ = run.model.errors(c,y[ix],valid[ix])
            easy = ((ref>0.)&(ref<=cuts[s]['easy']))[:,None]
            locally_ok = (a<=1e-10)&(e<=1e-10)
            r_stats = {}
            for k,arm in enumerate(ARMS[1:]):
                re = released[k,ix]
                j = np.flatnonzero(v[k+1]>=0)
                idx = j,v[k+1,j]
                selected_release = re[idx]
                r_stats[arm] = dict(releases=int(re.sum()),locally_admissible=int((re&locally_ok).sum()),
                    observed_positive_easy_reference=int((re&locally_ok&easy).sum()),
                    selected_releases=int(selected_release.sum()),
                    selected_not_locally_admissible=int((selected_release&~locally_ok[idx]).sum()),
                    limited_view_with_new_local_easy=bool(limited and (re&locally_ok&easy&active[:,None]).any()))
            results.append(dict(seed=seed,source=s,**out,release=r_stats,previously_limited=limited))
        beat(state='seed_scored',seed=seed)
    if len(results)!=102 or sum(r['previously_limited'] for r in results)!=33:
        raise ValueError('Complete view roster and predefined limited-view slice required')
    sources = np.unique(d['sites'])
    groups = [[r for r in results if r['source']==s and r['seed']==k]
              for s in sources for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def ci(x):
        if any(v is None for v in x):
            return None
        z = np.asarray(x).reshape(len(sources),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for key in pairs:
        v = [p.readout.merge_comparisons([r['comparisons'][key] for r in g]) for g in groups]
        comparisons[key] = {k:ci([x[k] for x in v]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risk = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in results]
        risk[a] = dict(passes=sum(x['supported'] for x in v),switches=sum(x['selected'] for x in v),
            all_over_2=sum(x['completion_all_risk_upper'] is not None and x['completion_all_risk_upper']>.02 for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in v))
    release = {a:{k:sum(r['release'][a][k] for r in results) for k in results[0]['release'][a]} for a in ARMS[1:]}
    run.once(PUBLIC/'analysis.json',run.stable(dict(records=results,risk=risk,comparisons=comparisons,
        release=release,prior_limited_views=33,original_exact=True,original_rules_unchanged=True,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','fit','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit the design before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from saved held-source checkpoints')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            print(msg,flush=True)
        beat(state='start')
        d,refs = run.population(),run.register()
        records = prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
        (fit if phase=='fit' else score)(d,refs,records,beat)
        name = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
                                                     artifact=run.artifact(name)))


if __name__=='__main__':
    main()
