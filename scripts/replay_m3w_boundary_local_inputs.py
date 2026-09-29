"""Reconstruct frozen packet bytes from verified local inputs, without a row cache."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
import numpy as np
from scripts import manage_m3w_boundary_diagnostic as run
from scripts import run_m3w_boundary_local_stream as local


def packets(cfg):
    parent=run.parent
    parent.api.torch.set_num_threads(4);parent.api.torch.set_num_interop_threads(1)
    _,_,data,jobs,oid,_,_,_=parent.old.load()
    causal={k:data[k] for k in parent.old.parent.CAUSAL_KEYS}
    fits={Path(r['path']).parent.name:r for r in json.loads((parent.PUBLIC/'training_freeze.json').read_text())['groups']}
    for c in parent.base.floor_api.contexts(causal,jobs,oid):
        tails={};tail_counts={}
        for site in parent.sources(c):
            _,_,_,_,y,_,_=parent.training_arrays(c,data,site)
            tails[site]={e:float(np.quantile(y[np.isfinite(y[:,k]) & (y[:,k]>0),k],.95))
                         if (np.isfinite(y[:,k]) & (y[:,k]>0)).any() else 0. for e,k in [('all',1),('easy',4)]}
            tail_counts[site]={e:int((np.isfinite(y[:,k]) & (y[:,k]>0)).sum()) for e,k in [('all',1),('easy',4)]}
        directions=[('fitting',None,s) for s in parent.sources(c)]
        directions += [('internal_transfer',pair,s) for pair,(sites,_) in enumerate(c['pairs']) for s in sites]
        for phase,pair,site in directions:
            if phase=='internal_transfer':
                at,values,masks,meta,pr=parent.view_predictions(c,causal,pair,site,fits)
                name=c['name']+f'_pair{pair}_from_'+site
                original=json.loads((parent.PRIVATE/'decisions'/(name+'.json')).read_text())
                assert meta['score_hashes']==original['score_hashes']
                with np.load(ROOT/original['arrays']['path'],allow_pickle=False) as z:
                    for k,v in masks.items():np.testing.assert_array_equal(z[k],v)
                source=meta['eval_site'];outer=meta['outer_held_sites']
            else:
                at,ids,x,env,y,pr,ident=parent.training_arrays(c,data,site)
                doc=parent.checked(fits[c['name']+'_fit_'+site]);assert doc['identity']==ident
                values={};support=None
                for arm,ref in doc['artifacts'].items():
                    assert parent.base.artifact(ROOT/ref['path'])==ref
                    state=parent.api.read_checkpoint(ROOT/ref['path'])
                    values[arm],ok=parent.api.predict(state,x,env)
                    if support is not None:np.testing.assert_array_equal(support,ok)
                    support=ok
                masks=parent.api.decisions(values,c['moving'][at],support,data['recordings'][ids],data['frames'][ids],ids)
                masks.update(ids=ids,support=support)
                name=c['name']+'_fitting_'+site;source=site;outer=[]
            ids=masks['ids'];cv,_,(floor,_),(neural,_)=parent.base.floor_api.costs(c,data,at)
            y=parent.api.targets(cv,floor,neural,c['job']['design']['easy_cut'])
            _,recording=np.unique(data['recordings'][ids],return_inverse=True)
            arrays=dict(ids=ids,targets=y,envelope=c['env'][at],moving=c['moving'][at],support=masks['support'],
                        recording=recording,frame=data['frames'][ids])
            for arm in cfg['arms']:
                arrays.update({arm+'_pred':values[arm],arm+'_selected':masks[arm],arm+'_matched':masks[arm+'_matched']})
            meta=dict(view=name,phase=phase,site=source,train_site=site,outer_held_sites=outer,
                source_roles=dict(producer_sites=c['producer_sites'],controller_sites=c['controller_sites']),
                cost_scale=pr['scale'],tail_cuts=tails[site],seed=c['job']['old_identity']['seed'],
                tail_positive_training_counts=tail_counts[site],
                causal_score_sha256={arm:parent.base.inter.array_hash(values[arm]) for arm in cfg['arms']},
                targets_used_only_for_diagnostic=True)
            arrays['meta_json']=np.array(json.dumps(meta,sort_keys=True))
            out=io.BytesIO();np.savez_compressed(out,**arrays)
            yield name,out.getvalue()


def finish(cfg,reg,records,refs,checks,times,began):
    assert len(records)==288
    summary=local.aggregate(records,cfg);summary['result_source']='fresh_run_local_verified_packet_reconstruction'
    summary['accounting_checks']=checks;summary_checks=local.verify_summary(summary)
    old=json.loads((run.parent.PUBLIC/'readout.json').read_text())
    prior={(r['view'],r['policy']):r['metric'] for r in old['rows']};agreements=0
    for row in summary['parent_readout_comparison']:
        for arm,scopes in row['risk'].items():
            for scope,events in scopes.items():
                policy=arm+('_matched' if scope=='matched' else '')
                for event,risk in events.items():
                    key='selected_positive_harm_ratio' if event=='all' else 'selected_easy_positive_harm_ratio'
                    run.compare_tree(risk,prior[row['view'],policy][key]);agreements+=1
    assert agreements==1728
    current_bytes=sum(p.stat().st_size for folder in (run.PUBLIC,run.PRIVATE) for p in folder.rglob('*') if p.is_file())
    assert current_bytes+len(json.dumps(summary).encode())+2**20<cfg['max_local_metadata_bytes']
    run.immutable(run.PUBLIC/'local_summary.json',summary)
    receipt=dict(groups=288,PID=os.getpid(),platform=platform.platform(),numpy=np.__version__,
        first_pass_compute_seconds=float(times[0]),replay_compute_seconds=float(times[1]),wall_seconds=time.monotonic()-began,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        full_local_replay_exact=True,all_packet_bytes_match_committed_manifest=True,
        native_accounting_checks=checks,summary_reduction_checks=summary_checks,
        parent_risk_agreements=agreements,result_hashes=refs,registration_sha256=run.digest(run.PUBLIC/'local_reconstruction_registration.json'),
        summary_sha256=run.digest(run.PUBLIC/'local_summary.json'),remote_verification='not_run_last_known_pending_37602475_connection_unavailable',
        local_row_cache_bytes=0,parameter_updates=0,independent_roles_read=False,deployment_changed=False)
    run.immutable(run.PUBLIC/'local_compute_receipt.json',receipt)
    lines=['# Local Frozen-Boundary Readout','',
        'fresh_run local diagnosis and exact numerical replay. All 288 reconstructed packet bytes match '
        'the previously committed CREATE packet hashes. cached_verified frozen inputs, models and actions. '
        'No training or policy changes. This is not completed CREATE analysis or independent confirmation.','',
        '| Phase | Arm | Event | Actual risk (%) | Predicted excess (pp) | Harm error (pp) | Reference error (pp) | Violations / defined |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for phase,row in summary['phases'].items():
        for arm in cfg['arms']:
            for event in ('all','easy'):
                z=row['arms'][arm]['matched'][event]
                vals=[local.value(z,k) for k in ['realized_risk_ratio','predicted_excess_over_selected_reference',
                    'harm_underestimate_over_selected_reference','reference_overestimate_over_selected_reference']]
                lines.append('| '+ ' | '.join([phase,arm,event,*vals,f"{z['violating_views']}/{z['defined_views']}"])+' |')
    lines+=['','Components share the actual selected-reference denominator. Actual risk = 2% + predicted '
        'excess + harm error + reference error. Signed components may offset. Equal-locality weighting. '
        'Unknown outcomes and zero denominators remain unknown, not safety passes.','',
        'There are 72 fitting and 216 transfer views of 12 previously opened localities; they are dependent. '
        'No fresh bootstrap or independent calibration/confirmation. Image-local detector-silver, '
        'obs8/pred12 stride12 raw frames only. No metric/seconds, true3D, foundation or human-gold claim. '
        'Stage5C and SMC remain off.','',
        f"Verified: {checks:,} native checks; {summary_checks:,} summary checks; {agreements:,} parent agreements. "
        f"Peak RSS {receipt['peak_RSS_bytes']/2**30:.3f} GiB; no row cache. Wall time {receipt['wall_seconds']:.2f}s.",'']
    (run.PUBLIC/'local_results.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in receipt.items() if k!='result_hashes'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');args=p.parse_args()
    cfg,base=local.bindings()
    assert base==json.loads((run.PUBLIC/'local_execution_registration.json').read_text())
    extra=[Path(__file__),run.PUBLIC/'local_reconstruction_amendment.md']
    reg=dict(parent_local_registration_sha256=run.digest(run.PUBLIC/'local_execution_registration.json'),
        code={str(p.relative_to(ROOT)):run.digest(p) for p in extra},same_packet_manifest=True,scientific_scope_changed=False)
    registration=run.PUBLIC/'local_reconstruction_registration.json'
    if args.register:run.immutable(registration,reg);print('Registered exact local packet reconstruction');return
    assert json.loads(registration.read_text())==reg
    run.parent.base.inter.committed(registration)
    assert not (run.PUBLIC/'local_compute_receipt.json').exists()
    manifest=json.loads((run.PUBLIC/'packet_manifest.json').read_text())['packets']
    began=time.monotonic();records=[];refs=[];checks=0;times=np.zeros(2)
    for index,(name,raw) in enumerate(packets(cfg)):
        entry=manifest[index];assert name==entry['group']
        result,n,elapsed=local.evaluate_packet(raw,entry,cfg)
        records.append(result);checks+=n;times+=elapsed
        refs.append(dict(group=name,result_sha256=hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()))
        if len(records)%12==0:
            beat=dict(pid=os.getpid(),groups=len(records),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                      state='local_hash_verified_diagnosis_and_replay',local_row_cache_bytes=0)
            (run.PRIVATE/'local_heartbeat.json').write_text(json.dumps(beat)+'\n');print(json.dumps(beat),flush=True)
    finish(cfg,reg,records,refs,checks,times,began)


if __name__=='__main__':main()
