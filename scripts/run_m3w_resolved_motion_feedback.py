"""Source-cross-fitted causal feedback from already resolved endpoint forecasts."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_causal_record_ledger as prior
from src.world_model import m3w_resolved_motion_feedback as model

run, np = prior.run, prior.np
PUBLIC = prior.PUBLIC/'resolved_feedback'
PRIVATE = prior.PRIVATE/'resolved_feedback'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_resolved_motion_feedback.py')
SCOPES = ('static', 'own', 'own_scene')
ARMS = SCOPES + tuple(s+'_free' for s in SCOPES)


def register():
    parents = (prior.REG, prior.PUBLIC/'freeze.json', prior.PUBLIC/'analysis.json',
               prior.PUBLIC/'replay_execution.json', Path(prior.__file__),
               Path(prior.model.__file__), Path(run.__file__), Path(run.model.__file__),
               run.REG, run.PUBLIC/'freeze.json', run.old.prior.PARENT/'population.json')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, seeds=run.kin.prior.SEEDS,
        maturity_steps=12, half_life_steps=24, query_cap=2, ridge=.002,
        bootstraps=3000, original_risk_budget=.02, development_only=True,
        fixed_matched_risk_costs=True, independent_outcomes_opened=False)))


def write_array(path, **arrays):
    tmp = path.with_suffix('.pending.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def features(d, refs, by_record, meta, beat):
    ids, origin = d['global_ids'], d['origin']
    scale = 1. + 6.5*np.median(np.linalg.norm(np.diff(d['history'],axis=1),axis=-1),axis=1)
    cells = []
    for seed in run.kin.prior.SEEDS:
        mark = PRIVATE/f'{seed}_features.json'
        if not mark.exists():
            cv, bank = run.candidates(d,refs,seed)
            f = np.zeros((*bank.shape[:2],model.OWN+model.SCENE),np.float32)
            counts = {}
            for r,(ix,groups) in by_record.items():
                f[ix], counts[r] = model.features(groups,meta['frames'][ix],meta['agents'][ix],
                    origin[ix],scale[ix],cv[ix],bank[ix])
            cache = PRIVATE/f'{seed}_features.npz'
            write_array(cache,ids=ids,feedback=f)
            run.once(mark,dict(seed=seed,registration=run.sha(REG),cache=run.artifact(cache),
                               counts=counts))
        c = run.read(mark)
        if c['registration'] != run.sha(REG) or c['seed'] != seed:
            raise ValueError('Feature checkpoint identity mismatch')
        run.checked(c['cache'])
        if sum(v['exact_maturity_matches'] for v in c['counts'].values()) != 74821:
            raise ValueError('Maturity roster differs from geometry-only audit')
        cells.append(c)
        beat(state='features_frozen',seed=seed)
    run.once(PUBLIC/'features.json',dict(cells=cells,no_target_arrays_opened=True))


def fit(d, refs, by_record, beat):
    feature_freeze = run.read(PUBLIC/'features.json')
    if not feature_freeze['no_target_arrays_opened'] or len(feature_freeze['cells']) != 3:
        raise ValueError('Freeze all causal features first')
    labels = run.old.prior.arrays(('target_eval','valid'))
    ids, sites = d['global_ids'], d['sites']
    old_cells = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    old_choices = {c['seed']:c for c in run.read(prior.PUBLIC/'freeze.json')['seeds']}
    cuts = run.cuts_by_source()
    splits = {s:np.flatnonzero(sites==s) for s in np.unique(sites)}
    cells = []
    for c in feature_freeze['cells']:
        seed = c['seed']
        mark = PRIVATE/f'{seed}_decisions.json'
        if not mark.exists():
            cv, bank = run.candidates(d,refs,seed)
            with np.load(run.checked(c['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                f = z['feedback']
            with np.load(run.checked(old_cells[seed]['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                static = dict(zip(z['names'],z['coefficients']))
            with np.load(run.checked(old_choices[seed]['choices']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                raw, matched, free = z['choices'][[0,2,3]]
            moments = {s:{} for s in SCOPES[1:]}
            for s,ix in splits.items():
                x, scale = run.model.descriptors(d['history'][ix],bank[ix],cv[ix],d['crowd'][ix])
                for scope in SCOPES[1:]:
                    moments[scope][s] = run.model.fitting_moments(model.augment(x,f[ix],scope),
                        bank[ix],cv[ix],labels['target_eval'][ix]-d['origin'][ix,None],
                        labels['valid'][ix],scale,1./d['crowd'][ix],cuts[s]['easy'])
            fitted = {s:{held:run.model.coefficients([m for site,m in moments[s].items() if site!=held])
                         for held in splits} for s in SCOPES[1:]}
            choice = np.full((len(ARMS),len(ids)),-1,np.int16)
            diagnostic = np.zeros((len(SCOPES),len(ids)))
            for r,(ix,groups) in by_record.items():
                s = sites[ix[0]]
                x, scale = run.model.descriptors(d['history'][ix],bank[ix],cv[ix],d['crowd'][ix])
                fixed = run.model.decision_costs(x,static[s],bank[ix],cv[ix],scale)
                counts = [(raw[ix[g]]>=0).sum() for g in groups]
                j = np.flatnonzero(matched[ix]>=0)
                for a,scope in enumerate(SCOPES):
                    u, all_cost, easy_cost, legal = fixed if not a else run.model.decision_costs(
                        model.augment(x,f[ix],scope),fitted[scope][s],bank[ix],cv[ix],scale)
                    np.testing.assert_array_equal(legal,fixed[3])
                    value = u/scale[:,None]
                    choice[a,ix],_ = prior.model.stream(groups,value,*fixed[1:],
                        counts=counts,witness=raw[ix])
                    choice[a+len(SCOPES),ix],_ = prior.model.stream(groups,value,all_cost,easy_cost,legal)
                    diagnostic[a,ix[j]] = easy_cost[j,matched[ix[j]]]
            np.testing.assert_array_equal(choice[0],matched)
            np.testing.assert_array_equal(choice[3],free)
            cache = PRIVATE/f'{seed}_decisions.npz'
            write_array(cache,ids=ids,choices=choice,predicted_easy_excess=diagnostic,
                own_coefficients=np.stack(list(fitted['own'].values())),
                own_scene_coefficients=np.stack(list(fitted['own_scene'].values())),
                coefficient_sources=np.asarray(list(splits)))
            run.once(mark,dict(seed=seed,registration=run.sha(REG),cache=run.artifact(cache),
                               fits=2*len(splits),static_controls_exact=True))
        m = run.read(mark)
        if m['registration'] != run.sha(REG) or m['seed'] != seed:
            raise ValueError('Decision checkpoint identity mismatch')
        run.checked(m['cache'])
        cells.append(m)
        beat(state='decisions_frozen',seed=seed)
    run.once(PUBLIC/'freeze.json',dict(cells=cells,features=run.artifact(PUBLIC/'features.json'),
                                      before_scoring=True,held_source_excluded=True))


def score(d, refs, by_record, beat):
    f = run.read(PUBLIC/'freeze.json')
    if not f['before_scoring'] or len(f['cells'])!=3:
        raise ValueError('Freeze all actions before scoring')
    labels = run.old.prior.arrays(('target_eval','valid'))
    y = labels['target_eval']-d['origin'][:,None]
    valid = labels['valid']
    p = run.parent
    cuts = run.cuts_by_source()
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(own_vs_static=('static','own',p.CV),
                 own_scene_vs_static=('static','own_scene',p.CV),
                 own_scene_vs_own=('own','own_scene',p.CV),
                 own_free_vs_static_free=('static_free','own_free',p.CV),
                 own_scene_free_vs_static_free=('static_free','own_scene_free',p.CV))
    records = []
    for c in f['cells']:
        seed = c['seed']
        cv, bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            choices, pred = z['choices'], z['predicted_easy_excess']
        for r,(ix,groups) in by_record.items():
            s = str(d['sites'][ix[0]])
            c0 = cuts[s]
            choice = choices[:,ix]
            for g in groups:
                n = (choice[:len(SCOPES),g]>=0).sum(1)
                if not (n == n[0]).all():
                    raise ValueError('Matched query count changed')
            paths = {a:run.model.selected_paths(z,bank[ix],cv[ix]) for a,z in zip(ARMS,choice)}
            out = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],
                d['queries'][ix],r,easy_cut=c0['easy'],hard_cut=c0['hard'])
            j = np.flatnonzero(choice[0]>=0)
            old = paths['static'][j]
            base = cv[ix[j]]
            target, mask = y[ix[j]], valid[ix[j]]
            ref,_ = run.model.errors(base,target,mask)
            alt,_ = run.model.errors(old,target,mask)
            radius = np.linalg.norm(old-base,axis=-1).max(1)
            known = mask.any(1)
            easy = known & (ref>0.) & (ref<=c0['easy'])
            excess = np.where(known,np.where(easy,np.maximum(alt-ref,0.)-.02*ref,0.),radius)
            weight = 1./d['crowd'][ix[j]]
            residual = (pred[:,ix[j]]-excess)/radius
            records.append(dict(seed=seed,source=s,**out,
                risk_forecast=dict(weight=float(weight.sum()),square=(residual**2 @ weight).tolist(),
                                   bias=(residual @ weight).tolist()),
                action_disagreements={a:int((choice[0]!=choice[i]).sum()) for i,a in enumerate(SCOPES)}))
        beat(state='scored_seed',seed=seed)
    if len(records)!=102:
        raise ValueError('Require every recording view')
    sources = np.unique(d['sites'])
    grouped = [[r for r in records if r['source']==s and r['seed']==seed]
               for s in sources for seed in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def ci(v):
        if any(x is None for x in v):
            return None
        z = np.asarray(v).reshape(len(sources),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for k in pairs:
        m = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in grouped]
        comparisons[k] = {key:ci([v[key] for v in m]) for key in ('known_pp','completion_lower_pp')}
    risks = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in records]
        risks[a] = dict(passes=sum(x['supported'] for x in v),switches=sum(x['selected'] for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v))
    mass = sum(r['risk_forecast']['weight'] for r in records)
    diag = {k:dict(zip(SCOPES,np.sum([r['risk_forecast'][k] for r in records],axis=0)/mass))
            for k in ('square','bias')}
    run.once(PUBLIC/'analysis.json',run.stable(dict(comparisons=comparisons,risk=risks,records=records,
        easy_excess_forecast=diag,static_controls_exact=True,original_rules_unchanged=True,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','features','fit','score','replay'))
    phase = parser.parse_args().phase
    register()
    if phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT)!=f.read_bytes():
            raise ValueError('Commit the design before execution')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**status):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from the last saved seed')
            message = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**status)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',message)
            print(message,flush=True)
        beat(state='start')
        d, refs = run.population(), run.register()
        meta = run.old.prior.arrays(('frames','agents'))
        by_record = prior.blocks(d,meta['frames'])
        if phase == 'features':
            features(d,refs,by_record,meta,beat)
        elif phase == 'fit':
            fit(d,refs,by_record,beat)
        else:
            score(d,refs,by_record,beat)
        name = {'features':'features','fit':'freeze','score':'analysis','replay':'analysis'}[phase]
        run.once(PUBLIC/f'{phase}_execution.json',dict(seconds=time.monotonic()-start,
            pid=os.getpid(),artifact=run.artifact(PUBLIC/f'{name}.json')))


if __name__ == '__main__':
    main()
