"""Compare first-query matched and context-blind calibration for release."""
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
from scripts import run_m3w_release_guard as base
from src.world_model import m3w_context_calibration as model

run,np,prior = base.run,base.np,base.prior
PUBLIC,PRIVATE = base.PUBLIC/'context',base.PRIVATE/'context'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_context_calibration.py')
ARMS = ('base','unguarded','pooled','blind','near','far')


def register():
    paths = (base.REG,base.PUBLIC/'freeze.json',base.PUBLIC/'analysis.json',
        base.PUBLIC/'replay_execution.json',Path(base.__file__),Path(base.guard.__file__),
        run.REG,Path(run.model.__file__),prior.REG,Path(prior.model.__file__),
        Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths],arms=ARMS,seeds=run.kin.prior.SEEDS,
        first_query_only=True,source_roles_reused=True,recordings_per_calibration_source=1,
        paired_source_draws=3000,original_budget=.02,independent_outcomes_opened=False)))


def freeze(d,refs,records,beat):
    ids,sites = d['global_ids'],d['sites']
    profiles = {r:model.profile(d['history'][ix[g[0]]]) for r,(ix,g) in records.items()}
    groups = {}
    for r,(ix,_) in records.items():
        groups.setdefault(str(sites[ix[0]]),[]).append(r)
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    cells = []
    for c0 in run.read(base.PUBLIC/'freeze.json')['cells']:
        seed = c0['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c0['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            cached = z['choices'][[0,1,3]]
            old_release = z['selected_release'][[0,1,3]]
        choices = np.full((len(ARMS),len(ids)),-1,np.int16)
        releases = np.zeros_like(choices,bool)
        choices[:3],releases[:3] = cached,old_release
        outcomes = []
        for cell in c0['selection']:
            ident = cell['identity']
            held,fit,cal = ident['held'],ident['fitting'],ident['calibration']
            if (held in (*fit,*cal) or set(fit)&set(cal) or len(cal)!=3
                    or ident['registration']!=run.sha(base.REG)):
                raise ValueError('Keep the frozen source separation')
            ix = np.flatnonzero(sites==held)
            mark = PRIVATE/f'{seed}_{held}.json'
            identity = dict(seed=seed,held=held,registration=run.sha(REG))
            if not mark.exists():
                with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],ids[ix])
                    coef,correction,harm = z['baseline_coefficients'],z['residual_coefficients'],z['guard_coefficients']
                norms = model.fitting_scale([profiles[r] for s in fit for r in groups[s]])
                peers = {s:groups[s] for s in cal}
                tables = {r:base.calibrate(d,{r:records[r]},(s,),cv,bank,coef,correction,harm,
                    y,valid,cuts)['radius']['candidates'] for s in cal for r in groups[s]}
                ch = np.full((3,len(ix)),-1,np.int16)
                used = np.zeros_like(ch,bool)
                details = []
                for r in groups[held]:
                    ri,g = records[r]
                    j = np.searchsorted(ix,ri)
                    ctx,original,re,scores = base.context(d,ri,g,cv,bank,coef,correction,harm)
                    np.testing.assert_array_equal(original,cached[0,ri])
                    row = {}
                    for k,arm in enumerate(ARMS[3:]):
                        selection = model.peers(profiles[r],profiles,peers,norms,arm)
                        threshold = model.threshold([tables[p] for p,_ in selection])
                        allowed = base.guard.keep('radius',threshold,scores['radius'],re)
                        v = base.guard.allocation(*ctx,allowed) if allowed.any() else original
                        ch[k,j] = v
                        a = np.flatnonzero(v>=0)
                        used[k,j[a]] = allowed[a,v[a]]
                        row[arm] = dict(distance=float(np.mean([x for _,x in selection])),
                                        threshold=threshold,peer_records=[p for p,_ in selection])
                    details.append(dict(recording=r,**row))
                path = mark.with_suffix('.npz')
                base.parent.base.nested.base.parent.base.write_array(path,ids=ids[ix],choices=ch,used=used)
                run.once(mark,dict(identity=identity,cache=run.artifact(path),details=details))
            c = run.read(mark)
            if c['identity']!=identity:
                raise ValueError('Saved calibration fold differs')
            with np.load(run.checked(c['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids[ix])
                choices[3:,ix],releases[3:,ix] = z['choices'],z['used']
            outcomes.extend(c['details'])
            beat(state='held_recordings_frozen',seed=seed,held=held)
        path = PRIVATE/f'{seed}_all.npz'
        if not path.exists():
            base.parent.base.nested.base.parent.base.write_array(path,ids=ids,choices=choices,used=releases)
        else:
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                np.testing.assert_array_equal(z['choices'],choices)
                np.testing.assert_array_equal(z['used'],releases)
        cells.append(dict(seed=seed,cache=run.artifact(path),details=outcomes))
    run.once(PUBLIC/'freeze.json',dict(cells=cells,outer_choices_before_scoring=True,uses_only_first_query=True))


def score(d,refs,records,beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['outer_choices_before_scoring']:
        raise ValueError('Require all outer choices frozen')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    pairs = {a+'_vs_base':('base',a,p.CV) for a in ARMS[1:]}
    pairs.update(near_vs_blind=('blind','near',p.CV),near_vs_far=('far','near',p.CV),
                 near_vs_pooled=('pooled','near',p.CV))
    old = {(r['seed'],r['recording']):r['metrics'] for r in run.read(base.PUBLIC/'analysis.json')['records']}
    results = []
    for c in f['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            ch,re = z['choices'],z['used']
        details = {v['recording']:v for v in c['details']}
        for r,(ix,g) in records.items():
            v = ch[:,ix]
            for j in g:
                n = (v[:,j]>=0).sum(1)
                if not (n==n[0]).all():
                    raise ValueError('Per-query count differs')
            s = str(d['sites'][ix[0]])
            b,c0 = bank[ix],cv[ix]
            paths = {a:run.model.selected_paths(z,b,c0) for a,z in zip(ARMS,v)}
            out = p.readout.score_recording({p.CV:c0,**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            for a,b0 in zip(ARMS[:3],('base','unguarded','radius')):
                if out['metrics'][a]!=old[seed,r][b0]:
                    raise ValueError('Parent control failed exact replay')
            results.append(dict(seed=seed,source=s,**out,used={a:int(z[ix].sum()) for a,z in zip(ARMS,re)},
                distance={a:details[r][a]['distance'] for a in ARMS[3:]},
                threshold={a:details[r][a]['threshold'] for a in ARMS[3:]},
                near_blind_changed_rows=int((v[4]!=v[3]).sum())))
        beat(state='seed_scored',seed=seed)
    if len(results)!=102:
        raise ValueError('All original views required')
    sites = sorted({r['source'] for r in results})
    grouped = [[r for r in results if r['source']==s and r['seed']==k]
               for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    def ci(x):
        if any(v is None for v in x):
            return None
        z = np.asarray(x).reshape(len(sites),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for key in pairs:
        v = [p.readout.merge_comparisons([r['comparisons'][key] for r in g]) for g in grouped]
        comparisons[key] = {k:ci([x[k] for x in v]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risks = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in results]
        risks[a] = dict(passes=sum(x['supported'] for x in v),switches=sum(x['selected'] for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in v),
            selected_released=sum(r['used'][a] for r in results))
    summary = {a:dict(mean_distance=float(np.mean([r['distance'][a] for r in results])),
        no_reopening_threshold=sum(r['threshold'][a]==0 for r in results)) for a in ARMS[3:]}
    run.once(PUBLIC/'analysis.json',run.stable(dict(records=results,risk=risks,comparisons=comparisons,
        context=summary,near_blind_changed_rows=sum(r['near_blind_changed_rows'] for r in results),
        original_controls_exact=True,original_rules_unchanged=True,outer_outcomes_not_used_for_threshold=True,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','freeze','score','replay'))
    phase = parser.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit the design before inference')
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
        (freeze if phase=='freeze' else score)(d,refs,records,beat)
        name = PUBLIC/('freeze.json' if phase=='freeze' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
                                                     artifact=run.artifact(name)))


if __name__=='__main__':
    main()
