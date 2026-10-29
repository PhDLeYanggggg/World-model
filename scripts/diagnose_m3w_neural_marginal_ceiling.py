"""New SOURCE diagnostic: does the frozen neural path add to a physical oracle?"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_balanced_region_cost as parent
from src.evaluation.m3w_neural_marginal_ceiling import candidate_ceiling

PUBLIC=parent.PUBLIC.parent/'european_neural_marginal_ceiling_v1'
PRIVATE=parent.PRIVATE.parent/'european_neural_marginal_ceiling_v1';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_neural_marginal_ceiling.py',
    ROOT/'tests/test_m3w_neural_marginal_ceiling.py',PUBLIC/'protocol.md')


def register():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    parent.once(REG,dict(bindings={str(p.relative_to(ROOT)):parent.sha(p) for p in FILES},
        forecaster_contract_sha256=parent.sha(parent.parent.REG),
        candidate_readout_sha256=parent.sha(parent.PUBLIC/'summary.json'),
        contexts=6,source_groups=48,controller_seeds_not_repeated=True,
        result_role='posthoc_exposed_SOURCE_future_label_diagnostic_not_primary',
        independent_roles_opened=False,optimizer_updates=0,deployment_changed=False))


def execute(verify=False):
    cfg,values,controls=parent.admission();binding=json.loads(REG.read_text())
    for rel,digest in binding['bindings'].items():
        if parent.sha(ROOT/rel)!=digest:raise ValueError('Registered oracle diagnosis changed')
        parent.parent.parent.committed(ROOT/rel)
    parent.parent.parent.committed(REG)
    if binding['candidate_readout_sha256']!=parent.sha(parent.PUBLIC/'summary.json'):
        raise ValueError('Frozen candidate readout changed')
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed ceiling immutable;use verify')
    parent.parent.torch.set_num_threads(4);parent.parent.torch.set_num_interop_threads(1)
    started=time.monotonic();rows=[];references=[];pc=dict(values[0],seeds=[43])
    expected=parent.parent.aggregates.aggregate_groups(parent.NAME)
    def beat(**event):
        if time.monotonic()-started>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12hhealthy boundary')
        value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        parent.json_write(PRIVATE/'heartbeat.json',value);print(json.dumps(value),flush=True)
    for context,items in sorted(values[1]['contexts'].items()):
        # No cost-head inference,scene feature extraction or controller-seed duplication.
        scopes=parent.parent.parent.prepare_context(context,items,values[2],values[3],values[4],beat,include_held=True)
        for scope,packets in scopes.items():
            for name,packet in sorted(packets.items()):
                known=parent.parent.np.isfinite(packet['y']).all((1,2))
                prior=expected[f'{context}_{name}_seed17_{scope}.json']['result']
                if len(known)!=prior['rows'] or int(known.sum())!=prior['known_rows']:
                    raise ValueError('Exact primary evaluation support required')
                result=candidate_ceiling(packet['ade'],packet['floor_ade'],packet['allowed'],known,packet['hard'],packet['scales'])
                records={}
                for record in sorted(set(packet['recordings'])):
                    at=packet['recordings']==record
                    records[str(record)]=candidate_ceiling(packet['ade'][at],packet['floor_ade'][at],packet['allowed'][at],known[at],packet['hard'][at],packet['scales'][at])
                value=dict(context=context,source=name,seed=43,scope=scope,result=result,records=records,
                    registration_sha256=parent.sha(REG),result_source='fresh_future_label_candidate_ceiling_diagnostic')
                path=PUBLIC/'groups'/f'{context}_{name}_{scope}.json'
                if verify:
                    if value!=json.loads(path.read_text()):raise ValueError('Exact marginal ceiling group differs')
                else:parent.once(path,value)
                rows.append(value);references.append(dict(path=str(path.relative_to(ROOT)),sha256=parent.sha(path)))
        beat(state='ceiling_progress',context=context,groups=len(rows),total=48)
    if len(rows)!=48:raise ValueError('All48SOURCE groups required without controller-seed repetition')
    summary={}
    for scope in ('fit','inner_held'):
        group=[r for r in rows if r['scope']==scope]
        summary[scope]={subset:{key:parent.parent.readout.interval(group,lambda r,s=subset,k=key:r[s][k],pc)
            for key in ('physical_oracle_gain_floor_percent','mixed_oracle_gain_floor_percent','neural_marginal_gain_floor_pp')}
            for subset in ('all','hard')}
        summary[scope]['record_views']=sum(len(r['records']) for r in group)
        summary[scope]['unique_source_recordings']=len({(r['source'],k) for r in group for k in r['records']})
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()) or references!=json.loads((PUBLIC/'complete.json').read_text())['groups']:
            raise ValueError('Exact marginal ceiling summary differs')
        parent.once(PUBLIC/'verification.json',dict(exact_replay=True,groups=48,summary_sha256=parent.sha(PUBLIC/'summary.json')))
    else:
        parent.once(PUBLIC/'summary.json',summary);parent.once(PUBLIC/'complete.json',dict(pid=os.getpid(),seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,groups=references,optimizer_updates=0,
            independent_roles_opened=False,oracle_is_deployable=False))
    parent.json_write(PRIVATE/('verification_runtime.json' if verify else 'runtime.json'),dict(pid=os.getpid(),seconds=time.monotonic()-started))
    beat(state='ceiling_verified' if verify else 'ceiling_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','run','verify'));a=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='register':register()
        else:execute(a.phase=='verify')
