"""Test nested frozen action banks with the unchanged recording constraints."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_recording_switch_ceiling as prior
from src.evaluation import m3w_recording_bank_ceiling as api

old,np,read,sha,artifact,once,stable=(prior.old,prior.np,prior.read,prior.sha,prior.artifact,prior.once,prior.stable)
parent=prior.parent
PUBLIC=old.HOME/'moving_recording_bank_ceiling_v1'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),Path(api.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_recording_bank_ceiling.py')
ARMS={'binary':['eqmotion'],'two_neural':['eqmotion','pct'],
      'full':['eqmotion','pct',*(v for v in parent.BASELINES if v!=parent.CV)]}


def registration():
    p=prior.PUBLIC/'analysis.json'
    run=prior.PUBLIC/'run_execution.json'
    if read(run)['artifact']['sha256']!=sha(p):raise ValueError('Frozen binary diagnostic required')
    once(REG,stable(dict(files=[artifact(f) for f in FILES],parents=[artifact(f) for f in
        (prior.REG,p,run,Path(prior.api.__file__),parent.REG,Path(parent.__file__))],
        arms=ARMS,expected_views=102,budget=.02,labels_for_ceiling_only=True,external_label_access=False)))
    return read(parent.REG)['forecasts'],read(p)


def run(banks,binary):
    a=old.prior.arrays(('global_ids','history','origin','sites','queries','recordings'))
    y=old.prior.arrays(('target_eval','valid'))
    ids,sites,recs,q=(a[k] for k in ('global_ids','sites','recordings','queries'))
    cuts=read(old.PARENTS[4])['cuts']
    cut={s:cuts[str(ref['job']['producer_fold'])] for ref in read(old.PARENTS[3])['forecasts'] for s in ref['job']['donor_sites']}
    old_views={(v['identity']['seed'],v['identity']['recording']):v for v in binary['recording_views']}
    rows=[]
    for seed in old.SEEDS:
        preds=parent.read_banks(banks,seed,ids,sites)
        for rec in np.unique(recs):
            ix=np.flatnonzero(recs==rec); site=str(sites[ix[0]]); ref=old_views[seed,str(rec)]
            if not np.all(sites[ix]==site) or ref['identity']['source']!=site or len(ix)!=ref['rows']:
                raise ValueError('No changed source or recording roster')
            h=a['history'][ix].astype(float);o=a['origin'][ix,None]
            floor=np.arange(1,13)[None,:,None]*(h[:,-1]-h[:,-2])[:,None]
            paths={k:v[ix] for k,v in preds.items()}
            paths.update({k:parent.baseline_numpy(h,i)-o for i,k in enumerate(parent.BASELINES) if k!=parent.CV})
            _,inverse,counts=np.unique(q[ix],return_inverse=True,return_counts=True);w=1./counts[inverse]
            scores={k:api.solve(floor,np.stack([paths[m] for m in names]),y['target_eval'][ix]-o,
                y['valid'][ix],w,str(rec),cut[site]['easy'],cut[site]['hard']) for k,names in ARMS.items()}
            value=scores['binary']; previous=ref['free_count']
            if value['feasible']!=(previous is not None) or previous and (
                value['completion_gain_mass']!=previous['completion_gain_mass'] or value['risk']!=previous['risk']):
                raise ValueError('Original binary ceiling did not replay exactly')
            rows.append(dict(seed=seed,recording=str(rec),source=site,reference_mass=ref['reference_mass'],arms=scores))
        print(dict(seed=seed,complete=len(rows),total=102),flush=True)
    if len(rows)!=102:raise ValueError('Complete fixed roster required')
    names=sorted(set(sites));draw=np.random.default_rng(73107).integers(len(names),size=(3000,len(names)))
    summary={}
    for k in ARMS:
        supported=[v for v in rows if v['arms'][k]['feasible']]; gain=None
        if len(supported)==len(rows):
            values=[]
            for s in names:
                view=[]
                for seed in old.SEEDS:
                    subset=[v for v in rows if v['source']==s and v['seed']==seed]
                    view.append(100*sum(v['arms'][k]['completion_gain_mass'] for v in subset)/sum(v['reference_mass'] for v in subset))
                values.append(np.mean(view))
            v=np.array(values);gain=dict(point=float(v.mean()),interval95=np.quantile(v[draw].mean(1),[.025,.975]).tolist())
        summary[k]=dict(feasible=len(supported),positive=sum(v['arms'][k]['completion_gain_mass']>0 for v in supported),
            completion_lower_pp=gain,action_counts=np.sum([v['arms'][k]['action_counts'] for v in supported],axis=0).tolist(),
            remaining_infeasible=[dict(seed=v['seed'],recording=v['recording'],source=v['source']) for v in rows if not v['arms'][k]['feasible']])
    once(PUBLIC/'analysis.json',stable(dict(arms=ARMS,summary=summary,recording_views=rows,
        binary_oracle_exact=True, original_rules_unchanged=True,labels_used_only_for_diagnostic=True,
        oracle_row_choices_exported=False,independent_confirmation=False,new_training=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','replay'))
    phase=parser.parse_args().phase;banks,ref=registration()
    if phase=='register':return
    for p in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT)!=p.read_bytes():
            raise ValueError('Commit frozen diagnostic before scoring')
    start=time.monotonic();run(banks,ref)
    once(PUBLIC/(phase+'_execution.json'),dict(pid=os.getpid(),seconds=time.monotonic()-start,
        artifact=artifact(PUBLIC/'analysis.json'),exact_replay=phase=='replay'))


if __name__=='__main__':main()
