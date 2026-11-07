"""Compare exact matched-volume hindsight with frozen past-only retiming."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_prefix_capacity as previous
from src.evaluation import m3w_timing_capacity as model

run, np, prior = previous.run, previous.np, previous.prior
PUBLIC, PRIVATE = prior.PUBLIC/'timing_capacity', prior.PRIVATE/'timing_capacity'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_timing_capacity.py')
ARMS = ('direct','local','banked')
LIMIT = 8.
PAST = previous.directional


def register():
    analysis = previous.PUBLIC/'analysis.json'
    proof = [previous.PUBLIC/f'{phase}_execution.json' for phase in ('score','replay')]
    if any(run.read(p)['artifact']['sha256'] != run.sha(analysis) for p in proof):
        raise ValueError('Require the exact earlier fixed-count readout')
    inputs = (analysis,*proof,previous.REG,Path(previous.model.__file__),
              PAST.PUBLIC/'freeze.json',PAST.PUBLIC/'analysis.json',
              PAST.size.PUBLIC/'freeze.json',Path(run.model.__file__),Path(prior.model.__file__),
              Path(prior.nested.base.__file__),Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        inputs=[run.artifact(p) for p in inputs], limit_seconds=LIMIT,
        modes=model.MODES, causal_arms=ARMS, risk_budget=.02, test_outcomes_opened=False)))


def parents(d,seed_cell,controls):
    seed = seed_cell['seed']
    original = {c['identity']['held']:c for c in controls[seed]}
    dc = np.full(len(d['sites']),-1,np.int16)
    coefs = {}
    for fold in seed_cell['folds']:
        s = fold['identity']['held']
        at = d['sites']==s
        with np.load(run.checked(fold['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'][at])
            dc[at] = z['choices'][0]
        with np.load(run.checked(original[s]['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'][at])
            np.testing.assert_array_equal(z['choices'][0],dc[at])
            coefs[s] = z['baseline_coefficients'].copy()
    return dc, coefs


def costs(d, ix, b, floor, coef):
    x, scale = run.model.descriptors(d['history'][ix],b,floor,d['crowd'][ix])
    u,a,e,ok = run.model.decision_costs(x,coef,b,floor,scale)
    return u, a, e, ok, scale


def freeze(d,refs,records,seed_cells,controls,beat):
    frozen = []
    for cell in seed_cells:
        seed = cell['seed']
        mark = PRIVATE/f'{seed}_freeze.json'
        identity = dict(seed=seed,registration=run.sha(REG))
        if not mark.exists():
            cv,bank = run.candidates(d,refs,seed)
            direct, coef = parents(d,cell,controls)
            ch = np.full((3,len(cv)),-1,np.int16)
            ch[0] = direct
            for ix,g in records.values():
                s = str(d['sites'][ix[0]])
                u,a,e,ok,scale = costs(d,ix,bank[ix],cv[ix],coef[s])
                mint = prior.nested.base.select(g,u,a,e,scale,ok)
                count = np.array([(mint[j]>=0).sum() for j in g])
                np.testing.assert_array_equal(count,[(direct[ix[j]]>=0).sum() for j in g])
                for k,carry in enumerate((False,True),1):
                    q = model.retime(g,u/scale[:,None],a,e,ok,mint,carry=carry)
                    spent = np.array([(q[j]>=0).sum() for j in g])
                    if (np.cumsum(spent)>np.cumsum(count)).any() or not carry and (spent>count).any():
                        raise ValueError('Causal policy used an unearned switch slot')
                    ch[k,ix] = q
            cache = mark.with_suffix('.npz')
            PAST.size.base.parent.base.nested.base.parent.base.write_array(cache,ids=d['global_ids'],choices=ch)
            run.once(mark,dict(identity=identity,cache=run.artifact(cache)))
        doc = run.read(mark)
        if doc['identity']!=identity:
            raise ValueError('Wrong seed or freeze registration')
        frozen.append(doc)
        beat(state='seed_frozen',seed=seed)
    run.once(PUBLIC/'freeze.json',dict(cells=frozen,no_development_targets_loaded=True))


def evaluate(d,refs,records,cells,controls,replay,beat):
    frozen = run.read(PUBLIC/'freeze.json')
    if not frozen['no_development_targets_loaded'] or len(frozen['cells'])!=3:
        raise ValueError('All causal policies must freeze before scoring')
    old = {(r['identity']['seed'],r['identity']['recording']):r
           for r in run.read(previous.PUBLIC/'analysis.json')['records']}
    before = {(r['seed'],r['recording']):r['metrics']['direct']
              for r in run.read(PAST.PUBLIC/'analysis.json')['records']}
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    cuts = run.cuts_by_source()
    pairs = dict(local_vs_direct=('direct','local',run.parent.CV),
                 banked_vs_direct=('direct','banked',run.parent.CV),
                 banked_vs_local=('local','banked',run.parent.CV))
    out = []
    for parent,f in zip(cells,frozen['cells']):
        seed = parent['seed']
        if f['identity']!=dict(seed=seed,registration=run.sha(REG)):
            raise ValueError('Wrong frozen identity')
        if not replay:
            cv,bank = run.candidates(d,refs,seed)
            dc,coef = parents(d,parent,controls)
            with np.load(run.checked(f['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['global_ids'])
                ch = z['choices']
                np.testing.assert_array_equal(ch[0],dc)
        for rec,(ix,g) in records.items():
            s = str(d['sites'][ix[0]])
            identity = dict(seed=seed,recording=rec,source=s,registration=run.sha(REG))
            path = PRIVATE/f'{seed}_{rec}.json'
            if not path.exists():
                if replay:
                    raise ValueError('Replay requires all checkpoints')
                c,b = cv[ix],bank[ix]
                easy,hard = (cuts[s][k] for k in ('easy','hard'))
                pred = {k:run.model.selected_paths(v[ix],b,c) for k,v in zip(ARMS,ch)}
                score = run.parent.readout.score_recording({run.parent.CV:c,**pred},pairs,y[ix],valid[ix],
                        d['queries'][ix],rec,easy_cut=easy,hard_cut=hard)
                if score['metrics']['direct']!=before[seed,rec]:
                    raise ValueError('The direct control failed exact replay')
                _,a,e,legal,_ = costs(d,ix,b,c,coef[s])
                p = previous.model.prepare(c,b,pred['direct'],dc[ix],y[ix],valid[ix],
                                           g,easy,np.stack((a,e)),legal)
                past = old[seed,rec]
                if p['fixed'].sum()!=past['direct_switches'] or p['reference']!=past['reference']:
                    raise ValueError('Matching recording count or objective denominator changed')
                arms = {}
                blocked = None
                for mode,ref in zip(model.MODES,previous.model.MODES[1:]):
                    beat(state='solving',seed=seed,complete=len(out),mode=mode)
                    z = (dict(status='infeasible',reason='inherited_'+blocked) if blocked else
                         model.solve(p,c,b,pred['direct'],y[ix],valid[ix],rec,easy,hard,mode,LIMIT))
                    ref_result = past['modes'][ref]
                    if z['status']=='infeasible' and ref_result['status'] in previous.FEASIBLE:
                        raise ValueError('A superset cannot rule out a verified witness')
                    if (z['status']=='optimal' and ref_result['status'] in previous.FEASIBLE and
                            z['paired_lower_mass'] < ref_result['paired_lower_mass']-1e-7):
                        raise ValueError('A relaxed optimum cannot be worse')
                    arms[mode] = z
                    if z['status']=='infeasible':
                        blocked = mode
                run.once(path,run.stable(dict(identity=identity,total=int(p['fixed'].sum()),
                    modes=arms,prior=past['modes'],online={k:score['metrics'][k]['risk'] for k in ARMS},
                    contrasts=score['comparisons'])))
            item = run.read(path)
            if item['identity']!=identity:
                raise ValueError('Wrong recording checkpoint identity')
            out.append(item)
            beat(state='recording_done',seed=seed,complete=len(out))
    run.once(PUBLIC/'analysis.json',summarize(out,pairs))


def summarize(rows,pairs):
    sources = sorted({r['identity']['source'] for r in rows})
    if len(rows)!=102 or len(sources)!=14 or sum(r['total'] for r in rows)!=6952:
        raise ValueError('Incomplete study roster')
    oracle = {}
    for mode,ref in zip(model.MODES,previous.model.MODES[1:]):
        arms = [r['modes'][mode] for r in rows]
        ok = [x for x in arms if x['status'] in previous.FEASIBLE]
        oracle[mode] = dict(states=dict(Counter(x['status'] for x in arms)),
            positive_witnesses=sum(x['paired_lower_mass']>1e-9 for x in ok),
            switches_retimed=sum(x['retimed_switches'] for x in ok),
            infeasible_recovered=sum(r['prior'][ref]['status']=='infeasible' and
                r['modes'][mode]['status'] in previous.FEASIBLE for r in rows),
            optimal_pairs_improved=sum(r['prior'][ref]['status']==r['modes'][mode]['status']=='optimal' and
                r['modes'][mode]['paired_lower_mass']>r['prior'][ref]['paired_lower_mass']+1e-8 for r in rows))
    online = {}
    for a in ARMS:
        v = [r['online'][a] for r in rows]
        online[a] = dict(switches=sum(x['selected'] for x in v),passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and
                                x['known_easy_net_degradation']>.02 for x in v))
    groups = [[r for r in rows if r['identity']['source']==s and r['identity']['seed']==k]
              for s in sources for k in run.kin.prior.SEEDS]
    draws = np.random.default_rng(73107).integers(14,size=(3000,14))
    contrasts = {}
    for name in pairs:
        vals = [run.parent.readout.merge_comparisons([r['contrasts'][name] for r in g]) for g in groups]
        x = np.asarray([r['completion_lower_pp'] for r in vals],dtype=float).reshape(14,3).mean(1)
        if not np.isfinite(x).all():
            raise ValueError('Required paired utility is not estimable')
        contrasts[name] = dict(point=float(x.mean()),interval95=np.quantile(x[draws].mean(1),[.025,.975]).tolist())
    return run.stable(dict(oracle=oracle,online=online,contrasts=contrasts,records=rows,
        original_rules_unchanged=True,independent_confirmation=False,deployment_changed=False,
        stage5c_executed=False,smc_enabled=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','freeze','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit the diagnostic definition first')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**fields):
            if time.monotonic()-start>43200 or shutil.disk_usage(ROOT).free<2*1024**3:
                raise OSError('Resume from verified checkpoints')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**fields)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            if fields['state']=='seed_frozen' or fields['state']=='recording_done' and fields['complete']%8==0:
                print(msg,flush=True)
        d,refs = run.population(),run.register()
        records = prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
        cells = run.read(PAST.PUBLIC/'freeze.json')['cells']
        controls = {c['seed']:c['selection'] for c in run.read(PAST.size.PUBLIC/'freeze.json')['cells']}
        if phase=='freeze':
            freeze(d,refs,records,cells,controls,beat)
        else:
            evaluate(d,refs,records,cells,controls,phase=='replay',beat)
        path = PUBLIC/('freeze.json' if phase=='freeze' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            aggregate_replay_only=phase=='replay',artifact=run.artifact(path)))


if __name__ == '__main__':
    main()
