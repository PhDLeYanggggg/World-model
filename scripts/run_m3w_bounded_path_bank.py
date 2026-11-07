"""Fit and evaluate bounded substitution on the frozen full action bank."""
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
from scripts import run_m3w_resolved_motion_feedback as feedback
from src.world_model import m3w_bounded_path_bank as model

prior, run, np = feedback.prior, feedback.run, feedback.np
PUBLIC, PRIVATE = prior.PUBLIC/'bounded_bank', prior.PRIVATE/'bounded_bank'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_bounded_path_bank.py')
ARMS = ('full','fixed_half','fixed_quarter','bounded','full_free','bounded_free')


def register():
    parents = (prior.REG,prior.PUBLIC/'freeze.json',prior.PUBLIC/'analysis.json',
               prior.PUBLIC/'replay_execution.json',run.REG,run.PUBLIC/'freeze.json',
               feedback.PUBLIC/'replay_execution.json',Path(feedback.__file__),
               Path(prior.model.__file__),Path(run.model.__file__),Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],arms=ARMS,levels=model.LEVELS,
        seeds=run.kin.prior.SEEDS,original_risk_budget=.02,source_bootstrap_draws=3000,
        no_future_input=True,development_only=True,independent_outcomes_opened=False)))


def fit(d, refs, by_record, beat):
    ids, sites = d['global_ids'], d['sites']
    split = {s:np.flatnonzero(sites==s) for s in np.unique(sites)}
    controls = {r['seed']:r for r in run.read(prior.PUBLIC/'freeze.json')['seeds']}
    control_heads = {r['seed']:r for r in run.read(run.PUBLIC/'freeze.json')['seeds']}
    label = run.old.prior.arrays(('target_eval','valid'))
    cuts = run.cuts_by_source()
    cells = []
    for seed in run.kin.prior.SEEDS:
        mark = PRIVATE/f'{seed}.json'
        if not mark.exists():
            cv, bank = run.candidates(d,refs,seed)
            width = bank.shape[1]
            with np.load(run.checked(controls[seed]['choices']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                raw, full, free = z['choices'][[0,2,3]]
            with np.load(run.checked(control_heads[seed]['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                weights = dict(zip(z['names'],z['coefficients']))
            coef = {1.:weights}
            for level in model.LEVELS[1:]:
                moments = {}
                for s,ix in split.items():
                    b = model.blend(bank[ix],cv[ix],level)
                    x, scale = run.model.descriptors(d['history'][ix],b,cv[ix],d['crowd'][ix])
                    moments[s] = run.model.fitting_moments(x,b,cv[ix],label['target_eval'][ix]-d['origin'][ix,None],
                        label['valid'][ix],scale,1./d['crowd'][ix],cuts[s]['easy'])
                coef[level] = {s:run.model.coefficients([m for t,m in moments.items() if s!=t]) for s in split}
            ch = np.full((len(ARMS),len(ids)),-1,np.int16)
            ch[0],ch[4] = full,free
            ch[1] = model.attenuate(full,width,.5)
            ch[2] = model.attenuate(full,width,.25)
            for ix,groups in by_record.values():
                costs = []
                s = sites[ix[0]]
                for level in model.LEVELS:
                    b = model.blend(bank[ix],cv[ix],level)
                    x, scale = run.model.descriptors(d['history'][ix],b,cv[ix],d['crowd'][ix])
                    u,a,e,ok = run.model.decision_costs(x,coef[level][s],b,cv[ix],scale)
                    costs.append((u/scale[:,None],a,e,ok))
                v = [np.concatenate(z,axis=1) for z in zip(*costs)]
                count = [(raw[ix[g]]>=0).sum() for g in groups]
                ch[3,ix],_ = prior.model.stream(groups,*v,counts=count,witness=raw[ix])
                ch[5,ix],_ = prior.model.stream(groups,*v)
            path = PRIVATE/f'{seed}.npz'
            feedback.write_array(path,ids=ids,choices=ch,sources=np.asarray(list(split)),
                coefficients=np.stack([[coef[a][s] for s in split] for a in model.LEVELS[1:]]))
            run.once(mark,dict(seed=seed,registration=run.sha(REG),cache=run.artifact(path),fits=2*len(split)))
        cell = run.read(mark)
        if cell['seed']!=seed or cell['registration']!=run.sha(REG):
            raise ValueError('Saved fit identity mismatch')
        run.checked(cell['cache'])
        cells.append(cell)
        beat(state='seed_frozen',seed=seed)
    run.once(PUBLIC/'freeze.json',dict(cells=cells,decisions_frozen_before_readout=True))


def score(d, refs, by_record, beat):
    frozen = run.read(PUBLIC/'freeze.json')
    if len(frozen['cells'])!=3 or not frozen['decisions_frozen_before_readout']:
        raise ValueError('All decisions must be frozen before evaluation')
    label = run.old.prior.arrays(('target_eval','valid'))
    y, valid = label['target_eval']-d['origin'][:,None],label['valid']
    p = run.parent
    cuts = run.cuts_by_source()
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update({f'bounded_vs_{a}':(a,'bounded',p.CV) for a in ARMS[:3]})
    pairs.update({f'{a}_vs_full':('full',a,p.CV) for a in ARMS[1:3]})
    pairs['bounded_free_vs_full_free'] = ('full_free','bounded_free',p.CV)
    records = []
    original = {(r['seed'],r['recording']):r['metrics'] for r in run.read(prior.PUBLIC/'analysis.json')['records']}
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d,refs,seed)
        with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            ch = z['choices']
        for r,(ix,groups) in by_record.items():
            z = ch[:,ix]
            for g in groups:
                counts = (z[:4,g]>=0).sum(1)
                if not (counts==counts[0]).all():
                    raise ValueError('Matched count changed')
            s = str(d['sites'][ix[0]])
            paths = {a:model.selected(bank[ix],cv[ix],c) for a,c in zip(ARMS,z)}
            for a,c in zip(ARMS,z):
                np.testing.assert_array_equal(np.any(paths[a]!=cv[ix],axis=(1,2)),c>=0)
            out = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            for a,b in (('full','ledger_matched'),('full_free','ledger_free')):
                if out['metrics'][a]!=original[seed,r][b]:
                    raise ValueError('Original policy does not replay')
            width = bank.shape[1]
            frequency = {a:np.bincount(c[c>=0]//width,minlength=len(model.LEVELS)).tolist()
                         for a,c in zip(ARMS,z)}
            records.append(dict(seed=seed,source=s,amplitude_counts=frequency,**out))
        beat(state='seed_scored',seed=seed)
    if len(records)!=102:
        raise ValueError('The complete recording roster is required')
    sources = np.unique(d['sites'])
    grouped = [[r for r in records if r['source']==s and r['seed']==seed]
               for s in sources for seed in run.kin.prior.SEEDS]
    draws = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def ci(xs):
        if any(x is None for x in xs):
            return None
        v = np.asarray(xs).reshape(len(sources),3).mean(1)
        return dict(point=float(v.mean()),interval95=np.quantile(v[draws].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for k in pairs:
        m = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in grouped]
        comparisons[k] = {key:ci([v[key] for v in m]) for key in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risks = {}
    for a in ARMS:
        risk = [r['metrics'][a]['risk'] for r in records]
        risks[a] = dict(passes=sum(x['supported'] for x in risk),switches=sum(x['selected'] for x in risk),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in risk),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in risk),
            amplitude_counts=np.sum([r['amplitude_counts'][a] for r in records],axis=0).tolist())
    run.once(PUBLIC/'analysis.json',run.stable(dict(comparisons=comparisons,risk=risks,records=records,
        exact_original_controls=True,original_risk_rules_unchanged=True,independent_confirmation=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','fit','score','replay'))
    phase = parser.parse_args().phase
    register()
    if phase=='register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT)!=f.read_bytes():
            raise ValueError('Commit the registered design before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from saved seeds')
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
