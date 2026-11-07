"""Freeze and evaluate deliberate one-slot waiting on the exposed development roster."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_timing_capacity as study
from src.world_model import m3w_one_slot_reservation as model

run, np, prior = study.run, study.np, study.prior
PUBLIC, PRIVATE = study.PUBLIC/'reservation', study.PRIVATE/'reservation'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_one_slot_reservation.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','freeze','score'))
    phase = p.parse_args().phase
    action_proof = study.PUBLIC/'frozen_agreement.json'
    proof = run.read(action_proof)
    if proof['freeze_sha256']!=run.sha(study.PUBLIC/'freeze.json') or any(
            v['local_vs_direct_changed'] or v['banked_vs_local_changed'] for v in proof['seeds']):
        raise ValueError('Need the causal greedy no-change diagnosis')
    run.once(REG,run.stable(dict(files=[run.artifact(v) for v in FILES],
        inputs=[run.artifact(v) for v in (study.REG,action_proof,study.PUBLIC/'freeze.json',
            study.PAST.PUBLIC/'freeze.json',study.PAST.size.PUBLIC/'freeze.json',
            Path(study.__file__),Path(prior.model.__file__),Path(run.parent.readout.__file__))],
        original_rules_unchanged=True,protected_outcomes_opened=False)))
    if phase=='register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT)!=f.read_bytes():
            raise ValueError('Commit the design before running')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    d,refs = run.population(),run.register()
    records = prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
    cells = run.read(study.PAST.PUBLIC/'freeze.json')['cells']
    controls = {c['seed']:c['selection'] for c in run.read(study.PAST.size.PUBLIC/'freeze.json')['cells']}
    if phase=='freeze':
        frozen = []
        for cell in cells:
            seed = cell['seed']
            mark = PRIVATE/f'{seed}.json'
            identity = dict(seed=seed,registration=run.sha(REG))
            if not mark.exists():
                cv,bank = run.candidates(d,refs,seed)
                dc,coefs = study.parents(d,cell,controls)
                ch = np.full(len(dc),-1,np.int16)
                reports = {}
                for rec,(ix,g) in records.items():
                    s = str(d['sites'][ix[0]])
                    u,a,e,ok,scale = study.costs(d,ix,bank[ix],cv[ix],coefs[s])
                    mint = prior.nested.base.select(g,u,a,e,scale,ok)
                    original = np.array([(dc[ix[j]]>=0).sum() for j in g])
                    np.testing.assert_array_equal(original,[(mint[j]>=0).sum() for j in g])
                    ch[ix],stat = model.allocate(g,u/scale[:,None],a,e,ok,mint)
                    balance = np.cumsum(original-np.array([(ch[ix[j]]>=0).sum() for j in g]))
                    if ((balance<0)|(balance>1)).any() or balance[-1]!=stat['unused']:
                        raise ValueError('A reservation must hold at most one earned switch')
                    reports[rec] = stat
                cache = mark.with_suffix('.npz')
                study.PAST.size.base.parent.base.nested.base.parent.base.write_array(
                    cache,ids=d['global_ids'],choices=np.stack((dc,ch)))
                run.once(mark,dict(identity=identity,cache=run.artifact(cache),records=reports))
            row = run.read(mark)
            if row['identity']!=identity:
                raise ValueError('Wrong frozen seed')
            frozen.append(row)
            print(dict(seed=seed,state='frozen'),flush=True)
        run.once(PUBLIC/'freeze.json',dict(cells=frozen,labels_opened=False))
        artifact = PUBLIC/'freeze.json'
    else:
        freeze = PUBLIC/'freeze.json'
        f = run.read(freeze)
        if f['labels_opened'] or run.read(PUBLIC/'freeze_execution.json')['artifact']['sha256']!=run.sha(freeze):
            raise ValueError('Decisions must freeze first')
        labels = run.old.prior.arrays(('target_eval','valid'))
        y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
        prior_risk = {(r['seed'],r['recording']):r['metrics']['direct'] for r in
                      run.read(study.PAST.PUBLIC/'analysis.json')['records']}
        cuts = run.cuts_by_source()
        bkey = run.parent.CV
        out = []
        for c in f['cells']:
            seed = c['identity']['seed']
            cv,bank = run.candidates(d,refs,seed)
            with np.load(run.checked(c['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['global_ids'])
                ch = z['choices']
            for rec,(ix,_) in records.items():
                s = str(d['sites'][ix[0]])
                floor,b = cv[ix],bank[ix]
                paths = {k:run.model.selected_paths(v[ix],b,floor)
                         for k,v in zip(('direct','reservation'),ch)}
                q = run.parent.readout.score_recording({bkey:floor,**paths},
                    dict(primary=('direct','reservation',bkey)),y[ix],valid[ix],d['queries'][ix],
                    rec,easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
                if q['metrics']['direct']!=prior_risk[seed,rec]:
                    raise ValueError('Exact direct control replay required')
                out.append(dict(seed=seed,source=s,recording=rec,actions=c['records'][rec],
                    direct=q['metrics']['direct']['risk'],reservation=q['metrics']['reservation']['risk'],
                    contrast=q['comparisons']['primary']))
        sources = sorted({r['source'] for r in out})
        if len(out)!=102 or len(sources)!=14:
            raise ValueError('Incomplete recording roster')
        groups = [[r['contrast'] for r in out if r['source']==s and r['seed']==k]
                  for s in sources for k in (17,29,43)]
        x = np.asarray([run.parent.readout.merge_comparisons(v)['completion_lower_pp'] for v in groups],
                        dtype=float).reshape(14,3).mean(1)
        draw = np.random.default_rng(73107).integers(14,size=(3000,14))
        metrics = {}
        for arm in ('direct','reservation'):
            v = [r[arm] for r in out]
            metrics[arm] = dict(switches=sum(x['selected'] for x in v),
                passes=sum(x['supported'] for x in v),
                undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
                easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and
                                    x['known_easy_net_degradation']>.02 for x in v))
        artifact = PUBLIC/'analysis.json'
        run.once(artifact,run.stable(dict(views=len(out),metrics=metrics,
            actions={k:sum(r['actions'][k] for r in out) for k in ('held','released','unused')},
            equal_count_views=sum(r['direct']['selected']==r['reservation']['selected'] for r in out),
            paired_lower_pp=dict(point=float(x.mean()),interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist()),
            direct_control_exact=True,original_rules_unchanged=True,independent_confirmation=False,
            deployment_changed=False,stage5c_executed=False,smc_enabled=False)))
        print(run.read(artifact),flush=True)
    run.once(PUBLIC/f'{phase}_execution.json',dict(seconds=time.monotonic()-start,artifact=run.artifact(artifact)))


if __name__=='__main__':
    main()
