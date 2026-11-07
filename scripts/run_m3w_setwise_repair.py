"""Test set-level supervision on the fixed, double-excluded donor proposals."""
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
from scripts import run_m3w_nested_feasible_teacher as nested
from src.world_model import m3w_setwise_repair as model

run,np,prior = nested.run,nested.np,nested.prior
PUBLIC,PRIVATE = nested.base.PUBLIC/'setwise',nested.base.PRIVATE/'setwise'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_setwise_repair.py')
ARMS = ('original','membership','additive','interaction')


def register():
    paths = (nested.REG,nested.PUBLIC/'freeze.json',nested.PUBLIC/'analysis.json',
        nested.PUBLIC/'replay_execution.json',Path(nested.__file__),Path(nested.base.teacher.__file__),
        prior.REG,prior.PUBLIC/'freeze.json',prior.PUBLIC/'analysis.json',Path(prior.model.__file__),
        run.REG,run.PUBLIC/'freeze.json',Path(run.model.__file__),Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths],arms=ARMS,seeds=run.kin.prior.SEEDS,
        paired_source_draws=3000,predicted_spend_dominance=True,shortlist_per_signal=16,
        original_budget=.02,independent_outcomes_opened=False)))


def fit(d,refs,records,beat):
    ids,sites = d['global_ids'],d['sites']
    splits = {s:np.flatnonzero(sites==s) for s in np.unique(sites)}
    cut = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    static = {c['seed']:c for c in run.read(prior.PUBLIC/'freeze.json')['seeds']}
    heads = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    cells = []
    for seed in run.kin.prior.SEEDS:
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(static[seed]['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            original = z['choices'][2]
        with np.load(run.checked(heads[seed]['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            reference_coef = dict(zip(z['names'],z['coefficients']))
        data,statistics = {},{}
        for s,ix in splits.items():
            b,c = bank[ix],cv[ix]
            x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            y,valid = labels['target_eval'][ix]-d['origin'][ix,None],labels['valid'][ix]
            _,a,e = nested.base.teacher.actual(b,c,y,valid,cut[s]['easy'])
            data[s] = (b,c,d['origin'][ix],x,scale,a,e,run.old.prior.groups(d['queries'][ix]))
            statistics[s] = run.model.fitting_moments(x,b,c,y,valid,scale,1./d['crowd'][ix],cut[s]['easy'])
        ch = np.full((len(ARMS),len(ids)),-1,np.int16)
        ch[0] = original
        summaries = []
        for held,ix in splits.items():
            mark = PRIVATE/f'{seed}_{held}.json'
            identity = dict(seed=seed,held=str(held),registration=run.sha(REG))
            if not mark.exists():
                old = run.read(nested.PRIVATE/f'{seed}_{held}.json')
                if old['identity']!=dict(seed=seed,held=str(held),registration=run.sha(nested.REG)):
                    raise ValueError('Require the matching nested teacher checkpoint')
                with np.load(run.checked(old['cache']),allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],ids[ix])
                    membership_coef = z['coefficients'][1]
                old_counts = {p['donor']:p for p in old['donors']}
                differences = []
                for donor in splits:
                    if donor==held:
                        continue
                    b,c,origin,x,s,a,e,g = data[donor]
                    coef = run.model.coefficients([v for t,v in statistics.items() if t not in (held,donor)])
                    u,_,_,legal = run.model.decision_costs(x,coef,b,c,s)
                    y,_,info = nested.base.teacher.examples(g,u/s[:,None],a,e,legal)
                    if dict(donor=str(donor),**info)!=old_counts[donor]:
                        raise ValueError('The frozen donor counts and outcomes must match')
                    diff = model.demonstration_differences(g,x,b,c,origin,s,y)
                    if len(diff)!=info['changed_queries']:
                        raise ValueError('Every changed set contributes exactly once')
                    differences.append(diff)
                additive,interact = model.fit(differences,membership_coef.size)
                choices = np.full((3,len(ix)),-1,np.int16)
                counts = dict(active=0,nontrivial=0,options=0,two_switch_nontrivial=0)
                for ri,gs in records.values():
                    if sites[ri[0]]!=held:
                        continue
                    b,c = bank[ri],cv[ri]
                    x,s = run.model.descriptors(d['history'][ri],b,c,d['crowd'][ri])
                    u,a,e,legal = run.model.decision_costs(x,reference_coef[held],b,c,s)
                    models = (membership_coef,additive.reshape(x.shape[1:]),
                              interact[:-model.PAIR_DIM].reshape(x.shape[1:]))
                    values = [nested.base.parent.ranker.score(x,w) for w in models]
                    for g in gs:
                        w = original[ri[g]]
                        if not (w>=0).any():
                            continue
                        z = model.slate(u[g]/s[g,None],a[g],e[g],legal[g],w)
                        counts['active'] += 1
                        counts['nontrivial'] += len(z)>1
                        counts['options'] += len(z)
                        counts['two_switch_nontrivial'] += len(z)>1 and z.shape[1]==2
                        bonus = model.mixed(b[g],c[g],d['origin'][ri[g]],s[g],z)@interact[-model.PAIR_DIM:]
                        for k,v in enumerate(values):
                            take = model.choose(z,v[g],bonus if k==2 else None)
                            rows = np.searchsorted(ix,ri[g[take[:,0]]])
                            choices[k,rows] = take[:,1]
                path = mark.with_suffix('.npz')
                nested.base.parent.base.write_array(path,ids=ids[ix],choices=choices,
                    additive_coefficients=additive,interaction_coefficients=interact)
                run.once(mark,dict(identity=identity,cache=run.artifact(path),counts=counts,
                                   changed_donor_pairs=sum(len(v) for v in differences)))
            cell = run.read(mark)
            if cell['identity']!=identity:
                raise ValueError('Saved held-source identity mismatch')
            with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids[ix])
                ch[1:,ix] = z['choices']
            summaries.append(dict(held=str(held),**cell['counts'],changed_donor_pairs=cell['changed_donor_pairs']))
            beat(state='held_source_frozen',seed=seed,fold=len(summaries),options=cell['counts']['options'])
        path = PRIVATE/f'{seed}_all.npz'
        if not path.exists():
            nested.base.parent.base.write_array(path,ids=ids,choices=ch)
        else:
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                np.testing.assert_array_equal(z['choices'],ch)
        cells.append(dict(seed=seed,cache=run.artifact(path),summaries=summaries))
    run.once(PUBLIC/'freeze.json',dict(cells=cells,before_outcomes=True,no_teacher_actions_exported=True))


def score(d,refs,by_record,beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['before_outcomes']:
        raise ValueError('The complete decisions must freeze first')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(interaction_vs_additive=('additive','interaction',p.CV),
        additive_vs_membership=('membership','additive',p.CV),
        interaction_vs_original=('original','interaction',p.CV),
        additive_vs_original=('original','additive',p.CV),membership_vs_original=('original','membership',p.CV))
    old = {(r['seed'],r['recording']):r['metrics']['original']
           for r in run.read(nested.PUBLIC/'analysis.json')['records']}
    results = []
    for c in f['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            ch = z['choices']
        for r,(ix,g) in by_record.items():
            s = str(d['sites'][ix[0]])
            v = ch[:,ix]
            for j in g:
                n = (v[:,j]>=0).sum(1)
                if not (n==n[0]).all():
                    raise ValueError('Exact count match required')
            paths = {a:run.model.selected_paths(z,bank[ix],cv[ix]) for a,z in zip(ARMS,v)}
            out = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            if out['metrics']['original']!=old[seed,r]:
                raise ValueError('The original does not replay exactly')
            local = {}
            for a in ARMS[2:]:
                z = [run.model.pair_error(d['history'][ix[j]],paths[a][j],y[ix[j]],valid[ix[j]]) for j in g]
                local[a] = [float(sum(t for t,_ in z if t is not None)),sum(t is not None for t,_ in z)]
            results.append(dict(seed=seed,source=s,**out,local=local,
                pair_changed_actions=int((v[2]!=v[3]).sum()),pair_changed_queries=sum(bool((v[2,j]!=v[3,j]).any()) for j in g)))
        beat(state='seed_scored',seed=seed)
    if len(results)!=102:
        raise ValueError('All recording views are required')
    sources = np.unique(d['sites'])
    grouped = [[r for r in results if r['source']==s and r['seed']==k]
               for s in sources for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def ci(x):
        if any(v is None for v in x):
            return None
        z = np.asarray(x).reshape(len(sources),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for key in pairs:
        b = [p.readout.merge_comparisons([r['comparisons'][key] for r in g]) for g in grouped]
        comparisons[key] = {k:ci([v[k] for v in b]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    changed = sum(r['pair_changed_queries'] for r in results)
    if not changed:
        comparisons['interaction_vs_additive'] = None
    risks = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in results]
        risks[a] = dict(passes=sum(x['supported'] for x in v),switches=sum(x['selected'] for x in v),
            all_over_2=sum(x['completion_all_risk_upper'] is not None and x['completion_all_risk_upper']>.02 for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in v))
    def relative(g):
        n = sum(r['local']['additive'][1] for r in g)
        return sum(r['local']['additive'][0]-r['local']['interaction'][0] for r in g)/n if n else None
    def hard(g,a):
        z = [r['metrics'][a]['subsets']['hard'] for r in g if r['metrics'][a]['subsets']['hard']['known_rows']]
        mass = sum(v['known_rows']*v['floor_ADE_pixel'] for v in z)
        return 100*sum(v['known_rows']*(v['floor_ADE_pixel']-v['model_ADE_pixel']) for v in z)/mass if mass else None
    slate = {key:sum(r[key] for c in f['cells'] for r in c['summaries']) for key in
             ('active','nontrivial','options','two_switch_nontrivial','changed_donor_pairs')}
    run.once(PUBLIC/'analysis.json',run.stable(dict(records=results,risk=risks,comparisons=comparisons,
        pair_error_reduction=ci([relative(g) for g in grouped]),slate=slate,
        hard_gain={a:ci([hard(g,a) for g in grouped]) for a in ARMS},
        interaction_changed_queries=changed,interaction_changed_rows=sum(r['pair_changed_actions'] for r in results),
        original_exact=True,original_risk_rules_unchanged=True,
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
            raise ValueError('Commit this design before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from saved held sources')
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
