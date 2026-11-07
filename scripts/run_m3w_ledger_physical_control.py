"""Reoptimize the physical-only and full ledgers at common causal counts."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_causal_record_ledger as ledger

run, np = ledger.run, ledger.np
PUBLIC = ledger.PUBLIC/'physical_control'; PRIVATE = ledger.PRIVATE/'physical_control'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md')
ARMS = ('physical_filter','physical_ledger','full_ledger')
PHYSICAL = tuple(k in run.parent.BASELINES for k in run.NAMES)


def register():
    a, e = (ledger.PUBLIC/n for n in ('analysis.json','score_execution.json'))
    if run.read(e)['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require verified parent ledger results')
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],parents=[run.artifact(p) for p in
        (a,e,ledger.REG,Path(ledger.__file__),Path(ledger.model.__file__),run.REG,run.PUBLIC/'freeze.json')],
        physical_actions=[k for k,b in zip(run.NAMES,PHYSICAL) if b], arms=ARMS, matched_query_counts=True,
        unchanged_budget=.02, external_outcomes=False)))


def freeze(d, refs, blocks):
    ids = d['global_ids']; files = []
    for cell in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = cell['seed']; cv, paths = run.candidates(d,refs,seed)
        with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids); coef = dict(zip(z['names'],z['coefficients']))
        choice = np.full((len(ARMS),len(ids)),-1,dtype=np.int16)
        for _,(ix,g) in blocks.items():
            s = str(d['sites'][ix[0]])
            x, scale = run.model.descriptors(d['history'][ix],paths[ix],cv[ix],d['crowd'][ix])
            u,a,e,legal = run.model.decision_costs(x,coef[s],paths[ix],cv[ix],scale)
            physical = legal & np.asarray(PHYSICAL)[None]
            parent = ledger.nested.base.select(g,u,a,e,scale,physical)
            counts = [(parent[j]>=0).sum() for j in g]; value = u/scale[:,None]
            opt = [ledger.model.stream(g,value,a,e,ok,counts=counts,witness=parent)[0] for ok in (physical,legal)]
            choice[:,ix] = np.stack((parent,*opt))
        p = PRIVATE/f'{seed}.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids); np.testing.assert_array_equal(z['choices'],choice)
        else:
            tmp = p.with_suffix('.pending.npz'); np.savez_compressed(tmp,ids=ids,choices=choice); tmp.replace(p)
        files.append(dict(seed=seed,choices=run.artifact(p)))
        print(dict(seed=seed,frozen_switches=(choice>=0).sum(1).tolist()),flush=True)
    run.once(PUBLIC/'freeze.json',dict(seeds=files,before_outcome_readout=True))


def score(d, refs, blocks):
    frozen = run.read(PUBLIC/'freeze.json')
    if not frozen['before_outcome_readout'] or len(frozen['seeds'])!=3:
        raise ValueError('All seed policies must be frozen first')
    lab = run.old.prior.arrays(('target_eval','valid')); y,valid = lab['target_eval']-d['origin'][:,None],lab['valid']
    p = run.parent; cuts = run.cuts_by_source(); records = []; actions = np.zeros((len(ARMS),len(run.NAMES)),dtype=np.int64)
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs['full_vs_physical'] = ('physical_ledger','full_ledger',p.CV)
    for v in frozen['seeds']:
        seed = v['seed']; cv, bank = run.candidates(d,refs,seed)
        with np.load(run.checked(v['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids']); choice = z['choices']
        for k,c in enumerate(choice):
            actions[k] += np.bincount(c[c>=0],minlength=len(run.NAMES))
        if actions[:2,~np.asarray(PHYSICAL)].any():
            raise ValueError('Physical control must be independently reoptimized without learned forecasts')
        for r,(ix,groups) in blocks.items():
            ch = choice[:,ix]; s = str(d['sites'][ix[0]]); cut = cuts[s]
            for g in groups:
                n = (ch[:,g]>=0).sum(1)
                if not (n == n[0]).all():
                    raise ValueError('Mismatched counts in the bank ablation')
            paths = {a:run.model.selected_paths(c,bank[ix],cv[ix]) for a,c in zip(ARMS,ch)}
            out = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cut['easy'],hard_cut=cut['hard'])
            records.append(dict(seed=seed,source=s,**out))
    if len(records)!=102:
        raise ValueError('All repeated recording views required')
    sources = np.unique(d['sites']); draws = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    groups = [[r for r in records if r['source']==s and r['seed']==seed] for s in sources for seed in run.kin.prior.SEEDS]
    def ci(v):
        if any(z is None for z in v):
            return None
        v = np.asarray(v).reshape(len(sources),3).mean(1)
        return dict(point=float(v.mean()),interval95=np.quantile(v[draws].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for k in pairs:
        v = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in groups]
        comparisons[k] = {f:ci([z[f] for z in v]) for f in ('known_pp','completion_lower_pp')}
    risks = {}
    for k in ARMS:
        v = [r['metrics'][k]['risk'] for r in records]
        risks[k] = dict(passes=sum(z['supported'] for z in v),switches=sum(z['selected'] for z in v),
            undefined_easy=sum(z['completion_easy_risk_upper'] is None for z in v),
            easy_over_2=sum(z['completion_easy_risk_upper'] is not None and z['completion_easy_risk_upper']>.02 for z in v))
    run.once(PUBLIC/'analysis.json',run.stable(dict(comparisons=comparisons,risks=risks,records=records,
        action_counts=dict(zip(ARMS,actions.tolist())),action_order=run.NAMES,matched_query_counts=True,
        original_rules_unchanged=True,independent_confirmation=False,deployment_changed=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('register','freeze','score','replay'))
    phase = p.parse_args().phase; register()
    if phase=='register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit matched physical control before scoring')
    PRIVATE.mkdir(parents=True,exist_ok=True); start = time.monotonic(); refs = run.register(); d = run.population()
    b = ledger.blocks(d,run.old.prior.arrays(('frames',))['frames'])
    if phase=='freeze':
        freeze(d,refs,b)
    else:
        score(d,refs,b)
    run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
        artifact=run.artifact(PUBLIC/('freeze.json' if phase=='freeze' else 'analysis.json'))))


if __name__=='__main__':
    main()
