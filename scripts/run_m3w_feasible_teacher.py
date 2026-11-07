"""Matched test of learning risk-constrained donor demonstrations."""
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
from scripts import run_m3w_query_centered_ranker as parent
from src.world_model import m3w_feasible_teacher as teacher

run,np,prior = parent.run,parent.np,parent.prior
PUBLIC,PRIVATE = prior.PUBLIC/'feasible_teacher',prior.PRIVATE/'feasible_teacher'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(teacher.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_feasible_teacher.py')
ARMS = ('original','gain_teacher','feasible_teacher')


def register():
    parents = (parent.REG,parent.PUBLIC/'replay_execution.json',Path(parent.__file__),
        Path(parent.ranker.__file__),prior.REG,prior.PUBLIC/'freeze.json',prior.PUBLIC/'analysis.json',
        run.REG,run.PUBLIC/'freeze.json',Path(prior.model.__file__),Path(run.model.__file__),
        Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],arms=ARMS,seeds=run.kin.prior.SEEDS,
        original_budget=.02,query_local_zero_credit_teacher=True,matched_training_counts=True,
        matched_evaluation_counts=True,bootstrap_draws=3000,independent_outcomes_opened=False)))


def fit(d,refs,by_record,beat):
    ids,sites = d['global_ids'],d['sites']
    splits = {s:np.flatnonzero(sites==s) for s in np.unique(sites)}
    cuts = run.cuts_by_source()
    label = run.old.prior.arrays(('target_eval','valid'))
    old = {c['seed']:c for c in run.read(prior.PUBLIC/'freeze.json')['seeds']}
    old_fits = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    cells = []
    for seed in run.kin.prior.SEEDS:
        mark = PRIVATE/f'{seed}.json'
        if not mark.exists():
            cv,bank = run.candidates(d,refs,seed)
            with np.load(run.checked(old[seed]['choices']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                raw,original = z['choices'][[0,2]]
            with np.load(run.checked(old_fits[seed]['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                static_coef = dict(zip(z['names'],z['coefficients']))
            moments, stats = {}, []
            for s,ix in splits.items():
                b,c,q = bank[ix],cv[ix],d['queries'][ix]
                x,_ = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
                legal = np.any(b!=c[:,None],axis=(2,3))
                u,a,e = teacher.actual(b,c,label['target_eval'][ix]-d['origin'][ix,None],
                                       label['valid'][ix],cuts[s]['easy'])
                g = run.old.prior.groups(q)
                z,mask,info = teacher.examples(g,u,a,e,legal)
                moments[s] = [parent.ranker.moments(x,v,mask,q)[1] for v in z]
                stats.append(dict(queries=len(g),**info))
            weights = [{s:parent.ranker.fit([v[j] for t,v in moments.items() if s!=t],bank.shape[1])
                        for s in splits} for j in range(2)]
            ch = np.full((len(ARMS),len(ids)),-1,np.int16)
            ch[0] = original
            for ix,groups in by_record.values():
                s = sites[ix[0]]
                x,scale = run.model.descriptors(d['history'][ix],bank[ix],cv[ix],d['crowd'][ix])
                _,a,e,legal = run.model.decision_costs(x,static_coef[s],bank[ix],cv[ix],scale)
                counts = [(raw[ix[g]]>=0).sum() for g in groups]
                for j,w in enumerate(weights,1):
                    ch[j,ix],_ = prior.model.stream(groups,parent.ranker.score(x,w[s]),a,e,legal,
                                                    counts=counts,witness=raw[ix])
            path = PRIVATE/f'{seed}.npz'
            parent.base.write_array(path,ids=ids,choices=ch,sources=np.asarray(list(splits)),
                coefficients=np.stack([[w[s] for s in splits] for w in weights]))
            totals = {k:sum(v[k] for v in stats) for k in
                      ('queries','active_queries','switches','changed_queries','saturated_queries')}
            totals['utility_all_easy_excess'] = np.sum([v['utility_all_easy_excess'] for v in stats],axis=0).tolist()
            run.once(mark,dict(seed=seed,registration=run.sha(REG),cache=run.artifact(path),
                               fits=2*len(splits),donor_summary=totals))
        c = run.read(mark)
        if c['registration']!=run.sha(REG) or c['seed']!=seed:
            raise ValueError('Saved fit identity mismatch')
        run.checked(c['cache'])
        cells.append(c)
        beat(state='seed_frozen',seed=seed)
    run.once(PUBLIC/'freeze.json',dict(cells=cells,decisions_frozen_before_readout=True,
                                      donor_actions_not_exported=True))


def score(d,refs,by_record,beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['decisions_frozen_before_readout']:
        raise ValueError('Require all frozen decisions')
    p = run.parent
    cuts = run.cuts_by_source()
    label = run.old.prior.arrays(('target_eval','valid'))
    y,valid = label['target_eval']-d['origin'][:,None],label['valid']
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(feasible_vs_gain=('gain_teacher','feasible_teacher',p.CV),
                 feasible_vs_original=('original','feasible_teacher',p.CV),
                 gain_vs_original=('original','gain_teacher',p.CV))
    old = {(r['seed'],r['recording']):r['metrics']['ledger_matched']
           for r in run.read(prior.PUBLIC/'analysis.json')['records']}
    records = []
    for c in f['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            choice = z['choices']
        for r,(ix,groups) in by_record.items():
            ch = choice[:,ix]
            for g in groups:
                count = (ch[:,g]>=0).sum(1)
                if not (count==count[0]).all():
                    raise ValueError('Every query count must match')
            s = str(d['sites'][ix[0]])
            paths = {a:run.model.selected_paths(z,bank[ix],cv[ix]) for a,z in zip(ARMS,ch)}
            out = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],
                d['queries'][ix],r,easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            if out['metrics']['original']!=old[seed,r]:
                raise ValueError('Original control differs')
            pair = {}
            for a in ARMS:
                v = [run.model.pair_error(d['history'][ix[g]],paths[a][g],y[ix[g],:],
                                        valid[ix[g]]) for g in groups]
                pair[a] = [float(sum(t for t,_ in v if t is not None)),sum(t is not None for t,_ in v)]
            records.append(dict(seed=seed,source=s,**out,pair_error=pair,
                replaced_selected=int(((ch[1]>=0)&(ch[2]<0)).sum()),
                changed_forecast_same_row=int(((ch[1]>=0)&(ch[2]>=0)&(ch[1]!=ch[2])).sum())))
        beat(state='seed_scored',seed=seed)
    if len(records)!=102:
        raise ValueError('Every recording and forecast-bank seed required')
    sources = np.unique(d['sites'])
    grouped = [[r for r in records if r['source']==s and r['seed']==k]
               for s in sources for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def ci(x):
        if any(v is None for v in x):
            return None
        z = np.asarray(x).reshape(len(sources),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for k in pairs:
        b = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in grouped]
        comparisons[k] = {key:ci([v[key] for v in b]) for key in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risks = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in records]
        risks[a] = dict(passes=sum(x['supported'] for x in v),switches=sum(x['selected'] for x in v),
            unknown_switches=sum(x['unknown_selected'] for x in v),
            all_over_2=sum(x['completion_all_risk_upper'] is not None and x['completion_all_risk_upper']>.02 for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in v))
    def pair_delta(g):
        n = sum(r['pair_error']['gain_teacher'][1] for r in g)
        return sum(r['pair_error']['gain_teacher'][0]-r['pair_error']['feasible_teacher'][0] for r in g)/n if n else None
    def hard(a,g):
        v = [r['metrics'][a]['subsets']['hard'] for r in g if r['metrics'][a]['subsets']['hard']['known_rows']]
        ref = sum(z['floor_ADE_pixel']*z['known_rows'] for z in v)
        return sum((z['floor_ADE_pixel']-z['model_ADE_pixel'])*z['known_rows']*100 for z in v)/ref if ref else None
    run.once(PUBLIC/'analysis.json',run.stable(dict(records=records,comparisons=comparisons,risk=risks,
        donor_summary=[c['donor_summary'] for c in f['cells']],pair_error_reduction=ci([pair_delta(g) for g in grouped]),
        hard_known_gain={a:ci([hard(a,g) for g in grouped]) for a in ARMS},
        replaced_selected=sum(r['replaced_selected'] for r in records),
        changed_forecast_same_row=sum(r['changed_forecast_same_row'] for r in records),
        original_control_exact=True,original_risk_rules_unchanged=True,independent_confirmation=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','fit','score','replay'))
    phase = parser.parse_args().phase
    register()
    if phase=='register':
        return
    for file in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(file.relative_to(ROOT))],cwd=ROOT)!=file.read_bytes():
            raise ValueError('Commit the design before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from saved seed checkpoints')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            print(msg,flush=True)
        beat(state='start')
        d,refs = run.population(),run.register()
        by_record = prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
        (fit if phase=='fit' else score)(d,refs,by_record,beat)
        target = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
                                                      artifact=run.artifact(target)))


if __name__=='__main__':
    main()
