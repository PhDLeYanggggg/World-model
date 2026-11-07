"""Expand chronological decision support and test a causal reference bonus."""
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
from scripts import run_m3w_setwise_repair as base
from src.world_model import m3w_expanded_risk_support as model

run,np,prior,nested = base.run,base.np,base.prior,base.nested
PUBLIC,PRIVATE = prior.PUBLIC/'expanded_support',prior.PRIVATE/'expanded_support'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_expanded_risk_support.py')
ARMS = ('original','restricted_membership','full_membership','reference_bonus','easy_caution')


def register():
    parents = (base.REG,base.PUBLIC/'analysis.json',base.PUBLIC/'freeze.json',
        base.PUBLIC/'slate_support.json',base.PUBLIC/'replay_execution.json',
        nested.REG,nested.PUBLIC/'freeze.json',nested.PUBLIC/'analysis.json',
        prior.REG,prior.PUBLIC/'freeze.json',prior.PUBLIC/'analysis.json',run.REG,
        run.PUBLIC/'freeze.json',Path(prior.model.__file__),Path(run.model.__file__),
        Path(prior.nested.cal.__file__),Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],arms=ARMS,seeds=run.kin.prior.SEEDS,
        fixed_query_counts=True,reference_bonus=.25,easy_excess_penalty=.25,
        original_budget=.02,bootstrap_draws=3000,no_new_training=True,
        independent_outcomes_opened=False)))


def freeze(d,refs,by_record,beat):
    ids,sites = d['global_ids'],d['sites']
    static = {c['seed']:c for c in run.read(prior.PUBLIC/'freeze.json')['seeds']}
    heads = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    teacher = {c['seed']:c for c in run.read(nested.PUBLIC/'freeze.json')['cells']}
    narrow = {c['seed']:c for c in run.read(base.PUBLIC/'freeze.json')['cells']}
    cells = []
    for seed in run.kin.prior.SEEDS:
        mark = PRIVATE/f'{seed}.json'
        if not mark.exists():
            cv,bank = run.candidates(d,refs,seed)
            with np.load(run.checked(static[seed]['choices']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                raw,original = z['choices'][[0,2]]
            with np.load(run.checked(heads[seed]['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                coef = dict(zip(z['names'],z['coefficients']))
            with np.load(run.checked(narrow[seed]['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                restricted = z['choices'][1]
            with np.load(run.checked(teacher[seed]['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                full = z['choices'][2]
            membership = {}
            for s in np.unique(sites):
                cell = run.read(nested.PRIVATE/f'{seed}_{s}.json')
                if cell['identity']!=dict(seed=seed,held=str(s),registration=run.sha(nested.REG)):
                    raise ValueError('Wrong held-source membership checkpoint')
                with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],ids[sites==s])
                    membership[s] = z['coefficients'][1]
            choices = np.full((len(ARMS),len(ids)),-1,np.int16)
            choices[1] = restricted
            for ix,groups in by_record.values():
                s = sites[ix[0]]
                c,b = cv[ix],bank[ix]
                x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
                u,a,e,legal = run.model.decision_costs(x,coef[s],b,c,scale)
                radius = np.linalg.norm(b-c[:,None],axis=-1).max(2)
                easy_ref = prior.nested.cal.predicted_components(x,coef[s],scale,radius)[...,3]
                basic,ref,caution = model.score_arms(u,e,easy_ref,scale)
                member = nested.base.parent.ranker.score(x,membership[s])
                n = [(raw[ix[g]]>=0).sum() for g in groups]
                for j,v in ((0,basic),(2,member),(3,ref),(4,caution)):
                    choices[j,ix],_ = prior.model.stream(groups,v,a,e,legal,counts=n,witness=raw[ix])
            np.testing.assert_array_equal(choices[0],original)
            np.testing.assert_array_equal(choices[2],full)
            path = PRIVATE/f'{seed}.npz'
            nested.base.parent.base.write_array(path,ids=ids,choices=choices)
            run.once(mark,dict(seed=seed,registration=run.sha(REG),cache=run.artifact(path),
                               exact_original_and_full_membership=True))
        c = run.read(mark)
        if c['seed']!=seed or c['registration']!=run.sha(REG):
            raise ValueError('Saved seed identity mismatch')
        run.checked(c['cache'])
        cells.append(c)
        beat(state='seed_frozen',seed=seed)
    run.once(PUBLIC/'freeze.json',dict(cells=cells,outcomes_not_used_for_choices=True))


def score(d,refs,by_record,beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['outcomes_not_used_for_choices']:
        raise ValueError('All choices must freeze before readout')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(full_vs_restricted=('restricted_membership','full_membership',p.CV),
        reference_vs_original=('original','reference_bonus',p.CV),
        caution_vs_original=('original','easy_caution',p.CV),
        reference_vs_caution=('easy_caution','reference_bonus',p.CV))
    old = {(r['seed'],r['recording']):r['metrics'] for r in run.read(base.PUBLIC/'analysis.json')['records']}
    full = {(r['seed'],r['recording']):r['metrics']['feasible_teacher']
            for r in run.read(nested.PUBLIC/'analysis.json')['records']}
    heads = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    results = []
    for c0 in f['cells']:
        seed = c0['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c0['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            choices = z['choices']
        with np.load(run.checked(heads[seed]['cache']),allow_pickle=False) as z:
            coef = dict(zip(z['names'],z['coefficients']))
        for r,(ix,g) in by_record.items():
            s = str(d['sites'][ix[0]])
            c,b,ch = cv[ix],bank[ix],choices[:,ix]
            for j in g:
                n = (ch[:,j]>=0).sum(1)
                if not (n==n[0]).all():
                    raise ValueError('Exact per-query count match required')
            paths = {a:run.model.selected_paths(z,b,c) for a,z in zip(ARMS,ch)}
            out = p.readout.score_recording({p.CV:c,**paths},pairs,y[ix],valid[ix],
                d['queries'][ix],r,easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            cache = (old[seed,r]['original'],old[seed,r]['membership'],full[seed,r])
            for arm,expected in zip(ARMS,cache):
                if out['metrics'][arm]!=expected:
                    raise ValueError('Cached original and matched controls must replay')
            x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            _,a,e,legal = run.model.decision_costs(x,coef[s],b,c,scale)
            _,real_a,real_e = nested.base.teacher.actual(b,c,y[ix],valid[ix],cuts[s]['easy'])
            ref,_ = run.model.errors(c,y[ix],valid[ix])
            ref = np.where((ref>0.)&(ref<=cuts[s]['easy']),ref,0.)
            oracle = model.bound(g,a,e,legal,ch[0],real_a,real_e,ref)
            results.append(dict(seed=seed,source=s,**out,observed_easy_reference={
                a:float(ref[z>=0].sum()) for a,z in zip(ARMS,ch)},fixed_prefix_bound=oracle))
        beat(state='seed_scored',seed=seed)
    if len(results)!=102:
        raise ValueError('Every recording view is required')
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
        b = [p.readout.merge_comparisons([r['comparisons'][key] for r in g]) for g in groups]
        comparisons[key] = {k:ci([v[k] for v in b]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risk = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in results]
        risk[a] = dict(passes=sum(x['supported'] for x in v),switches=sum(x['selected'] for x in v),
            all_over_2=sum(x['completion_all_risk_upper'] is not None and x['completion_all_risk_upper']>.02 for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in v),
            pooled_easy_reference=float(sum(r['observed_easy_reference'][a] for r in results)))
    bound = [r['fixed_prefix_bound']['failed_necessary_conditions'] for r in results]
    labels = sorted({k for ks in bound for k in ks})
    run.once(PUBLIC/'analysis.json',run.stable(dict(records=results,risk=risk,comparisons=comparisons,
        full_fixed_prefix_infeasible=sum(bool(v) for v in bound),
        full_fixed_prefix_not_ruled_out=sum(not v for v in bound),
        full_prefix_reasons={k:sum(k in v for v in bound) for k in labels},
        narrow_infeasible=run.read(base.PUBLIC/'slate_support.json')['proven_infeasible'],
        original_and_cached_controls_exact=True,original_rules_unchanged=True,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','freeze','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit the design before running it')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from frozen seed outputs')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            print(msg,flush=True)
        beat(state='start')
        d,refs = run.population(),run.register()
        records = prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
        (freeze if phase=='freeze' else score)(d,refs,records,beat)
        name = PUBLIC/('freeze.json' if phase=='freeze' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
                                                     artifact=run.artifact(name)))


if __name__=='__main__':
    main()
