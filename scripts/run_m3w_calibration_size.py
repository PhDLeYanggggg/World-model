"""Three-versus-six source calibration at identical fitted costs."""
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
from scripts import run_m3w_context_calibration as context
from src.world_model import m3w_calibration_size as model

base,run,np,prior = context.base,context.run,context.np,context.prior
PUBLIC,PRIVATE = base.PUBLIC/'size',base.PRIVATE/'size'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_calibration_size.py')
ARMS = ('base','unguarded','three_a','three_b','utility6_a','utility6_b','six')


def register():
    parents = (context.REG,context.PUBLIC/'analysis.json',context.PUBLIC/'replay_execution.json',
        base.REG,base.PUBLIC/'analysis.json',Path(base.__file__),Path(base.guard.__file__),
        Path(base.parent.model.__file__),run.REG,Path(run.model.__file__),prior.REG,Path(prior.model.__file__),
        Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],arms=ARMS,seeds=run.kin.prior.SEEDS,
        fit_sources=7,calibration_sources=6,held_sources=1,
        grid=base.guard.GRIDS['radius'],paired_draws=3000,original_budget=.02,
        independent_outcomes_opened=False)))


def fit(d,refs,records,beat):
    ids,sites = d['global_ids'],d['sites']
    splits = {str(s):np.flatnonzero(sites==s) for s in np.unique(sites)}
    by_source = {s:[] for s in splits}
    for ri,g in records.values():
        s = str(sites[ri[0]])
        by_source[s].append((g,np.searchsorted(splits[s],ri)))
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    cells = []
    for seed in run.kin.prior.SEEDS:
        cv,bank = run.candidates(d,refs,seed)
        data,m = {},{}
        for s,ix in splits.items():
            b,c = bank[ix],cv[ix]
            x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            _,a,e = base.parent.base.nested.base.teacher.actual(b,c,y[ix],valid[ix],cuts[s]['easy'])
            data[s] = (b,c,x,scale,a,e)
            m[s] = run.model.fitting_moments(x,b,c,y[ix],valid[ix],scale,1./d['crowd'][ix],cuts[s]['easy'])
        choices = np.full((len(ARMS),len(ids)),-1,np.int16)
        released = np.zeros_like(choices,bool)
        summary = []
        for held,ix in splits.items():
            mark = PRIVATE/f'{seed}_{held}.json'
            learn,a,b = model.roles(splits,held)
            identity = dict(seed=seed,held=held,fit=list(learn),cal_a=list(a),cal_b=list(b),registration=run.sha(REG))
            if not mark.exists():
                moments = []
                for donor in learn:
                    p,c,x,scale,true_a,true_e = data[donor]
                    co = run.model.coefficients([m[k] for k in learn if k!=donor])
                    z = np.empty((*x.shape[:2],x.shape[2]+4))
                    v = np.zeros(x.shape[:2],bool)
                    r = np.empty((*x.shape[:2],2))
                    for g,j in by_source[donor]:
                        _,pa,pe,_,_,_,_,zz,veto = base.original_context(g,x[j],scale[j],p[j],c[j],co)
                        z[j],v[j] = zz,veto
                        r[j] = np.stack((true_a[j]-pa,true_e[j]-pe),axis=-1)/scale[j,None,None]
                    moments.append(base.parent.model.moments(z,r,v,1./d['crowd'][splits[donor]]))
                coef = run.model.coefficients([m[s] for s in learn])
                correction = base.parent.model.coefficients(moments)
                unused_harm = np.zeros(correction.shape[:2])
                ca,cb = [base.calibrate(d,records,v,cv,bank,coef,correction,unused_harm,
                    y,valid,cuts)['radius']['candidates'] for v in (a,b)]
                sel,both = model.comparisons(ca,cb)
                ch = np.full((len(ARMS),len(ix)),-1,np.int16)
                used = np.zeros_like(ch,bool)
                for ri,g in records.values():
                    if str(sites[ri[0]])!=held:
                        continue
                    ctx,orig,re,scores = base.context(d,ri,g,cv,bank,coef,correction,unused_harm)
                    j = np.searchsorted(ix,ri)
                    ch[0,j] = orig
                    for k,index in enumerate((7,*sel),1):
                        keep = base.guard.keep('radius',index,scores['radius'],re)
                        t = base.guard.allocation(*ctx,keep) if keep.any() else orig
                        ch[k,j] = t
                        take = np.flatnonzero(t>=0)
                        used[k,j[take]] = keep[take,t[take]]
                path = mark.with_suffix('.npz')
                base.parent.base.nested.base.parent.base.write_array(path,ids=ids[ix],choices=ch,used=used,
                    baseline_coefficients=coef,residual_coefficients=correction)
                run.once(mark,dict(identity=identity,cache=run.artifact(path),
                    thresholds=dict(zip(ARMS[2:],sel)),cal_a=ca,cal_b=cb,cal_six=both))
            c = run.read(mark)
            if c['identity']!=identity:
                raise ValueError('Frozen role identity differs')
            with np.load(run.checked(c['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids[ix])
                choices[:,ix],released[:,ix] = z['choices'],z['used']
            summary.append(c)
            beat(state='held_frozen',seed=seed,fold=len(summary),thresholds=c['thresholds'])
        path = PRIVATE/f'{seed}_all.npz'
        if not path.exists():
            base.parent.base.nested.base.parent.base.write_array(path,ids=ids,choices=choices,used=released)
        else:
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                np.testing.assert_array_equal(z['choices'],choices)
                np.testing.assert_array_equal(z['used'],released)
        cells.append(dict(seed=seed,cache=run.artifact(path),selection=summary))
    run.once(PUBLIC/'freeze.json',dict(cells=cells,decisions_frozen_before_readout=True))


def score(d,refs,records,beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['decisions_frozen_before_readout']:
        raise ValueError('Require the full frozen outer matrix')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    pairs = {a+'_vs_base':('base',a,p.CV) for a in ARMS[1:]}
    pairs.update(six_vs_a=('three_a','six',p.CV),six_vs_b=('three_b','six',p.CV),
        utility_vs_a=('three_a','utility6_a',p.CV),utility_vs_b=('three_b','utility6_b',p.CV),
        six_vs_utility_a=('utility6_a','six',p.CV),six_vs_utility_b=('utility6_b','six',p.CV))
    composite = dict(six_vs_three_mean=('six_vs_a','six_vs_b'),
        utility_only_pooling=('utility_vs_a','utility_vs_b'),
        additional_constraints=('six_vs_utility_a','six_vs_utility_b'))
    out = []
    for c in f['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            ch,re = z['choices'],z['used']
        for r,(ix,g) in records.items():
            v = ch[:,ix]
            for j in g:
                n = (v[:,j]>=0).sum(1)
                if not (n==n[0]).all():
                    raise ValueError('Every per-query count must match')
            s = str(d['sites'][ix[0]])
            c0,b0 = cv[ix],bank[ix]
            paths = {a:run.model.selected_paths(z,b0,c0) for a,z in zip(ARMS,v)}
            res = p.readout.score_recording({p.CV:c0,**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            for k,(a,b) in composite.items():
                res['comparisons'][k] = model.mean_pair(res['comparisons'][a],res['comparisons'][b])
            out.append(dict(seed=seed,source=s,**res,released={a:int(z[ix].sum()) for a,z in zip(ARMS,re)}))
        beat(state='seed_scored',seed=seed)
    if len(out)!=102:
        raise ValueError('The complete 102 recording views are required')
    sites = sorted({r['source'] for r in out})
    grouped = [[r for r in out if r['source']==s and r['seed']==k]
               for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    def ci(v):
        if any(x is None for x in v):
            return None
        z = np.asarray(v).reshape(len(sites),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparison = {}
    for key in (*pairs,*composite):
        v = [p.readout.merge_comparisons([r['comparisons'][key] for r in g]) for g in grouped]
        comparison[key] = {k:ci([r[k] for r in v]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risk = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in out]
        risk[a] = dict(passes=sum(x['supported'] for x in v),switches=sum(x['selected'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in v),
            outer_preservation_failures=sum(not base.guard.preserves(r['metrics']['base']['risk'],
                r['metrics'][a]['risk']) for r in out),selected_releases=sum(r['released'][a] for r in out))
    selection = [r for c in f['cells'] for r in c['selection']]
    run.once(PUBLIC/'analysis.json',run.stable(dict(records=out,risk=risk,comparisons=comparison,
        no_reopening_folds={a:sum(r['thresholds'][a]==0 for r in selection) for a in ARMS[2:]},
        three_a_b_disagreement=sum(r['thresholds']['three_a']!=r['thresholds']['three_b'] for r in selection),
        outer_outcomes_not_used_for_selection=True,original_rules_unchanged=True,
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
            raise ValueError('Commit the design first')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from saved source checkpoints')
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
