"""Complete source-held track-distinct risk retrieval, with chunk recovery."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_channel_specialist as previous
from src.evaluation import m3w_track_analogue as method
from src.evaluation.m3w_boundary_audit import summarize as boundary

source,rel,np=previous.source,previous.rel,previous.np
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_track_analogue_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_track_analogue_v1'
CONFIG=ROOT/'configs/m3w_moving_track_analogue_v1.json'; REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/evaluation/m3w_track_analogue.py',
    ROOT/'tests/test_m3w_track_analogue.py',ROOT/'src/data_unification/m3w_moving_forecast_data.py')


def register():
    cfg=source.read(CONFIG)
    if (cfg['k']!=32 or cfg['features']!='original120' or cfg['primary']!='track32_minus_shared64'
            or cfg['minimum_joint_cost_reduction']!=.01 or cfg['minimum_passing_contexts']!=2
            or cfg['risk_budget']!=.02 or cfg['easy_net_budget']!=.02
            or any(cfg[k] for k in ('threshold_search','new_H_readout','identity_certification',
                'deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed source-excluded retrieval and original rules required')
    ref=source.read(previous.audit.PUBLIC/'analysis.json')
    rel.base.exact(source.read(previous.PUBLIC/'verification.json')['analysis_sha256'],source.sha(previous.PUBLIC/'analysis.json'))
    source.once(REG,source.stable(dict(config=cfg,files={str(f.relative_to(ROOT)):source.sha(f) for f in FILES},
        previous_registration=source.artifact(previous.REG),negative_model_result=source.artifact(previous.PUBLIC/'analysis.json'),
        fitting_cache_audit=source.artifact(previous.audit.PUBLIC/'analysis.json'),
        admission=source.artifact(ROOT/'outputs/publication_readiness_2026_09/moving_development_admission_v1/admission.json'),
        result_role='exposed_own_B_source_excluded_information_probe',new_V_or_H=False,
        original_rules_unchanged=True,identity_certification=False,independent_confirmation=False)))
    return cfg,ref


def prepare(ref,beat,guard):
    _,values=previous.audit.old.register(beat); full=values[2]; records=[]
    for context in ref['contexts']:
        d=previous.inputs(context); ids=d['ids']
        rel.base.exact(full['sites'][ids],d['sites'])
        meta={k:full[k][ids] for k in ('recordings','agents','frames')}; meta['ids']=ids
        if len(set(zip(d['sites'],meta['recordings'],meta['agents'],meta['frames'])))!=len(ids):
            raise ValueError('Repeated source-recording-agent-frame row')
        path=PRIVATE/'metadata'/(context['job']+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                for k,v in meta.items(): rel.base.exact(old[k],v)
        else: np.savez_compressed(path,**meta)
        folds=[]
        for site in sorted(set(d['sites'])):
            fit=d['sites']!=site; held=~fit
            buckets=set(zip(d['sites'][fit],meta['agents'][fit]))
            tracks=set(zip(d['sites'][fit],meta['recordings'][fit],meta['agents'][fit]))
            folds.append(dict(source=str(site),query_rows=int(held.sum()),donor_rows=int(fit.sum()),
                donor_sources=sorted(set(d['sites'][fit])),donor_recordings=len(set(meta['recordings'][fit].tolist())),
                donor_recording_tracks=len(tracks),conservative_identity_buckets=len(buckets),
                collapsed_cross_recording_ids=len(tracks)-len(buckets),enough_for_k32=len(buckets)>=32,
                same_source_donors=0,same_recording_donors=0))
        records.append(dict(job=context['job'],metadata=source.artifact(path),
            hashes={k:rel.parent.kin.digest(v) for k,v in meta.items()},folds=folds))
        beat(state='metadata_complete',job=context['job']); guard()
    source.once(PUBLIC/'metadata.json',source.stable(dict(contexts=records,result_source='fresh_run_lineage_metadata',
        metadata_not_inference_features=True,new_V_or_H=False,physical_identity_certified=False)))


def inputs(context):
    d=previous.inputs(context)
    record=next(c for c in source.read(PUBLIC/'metadata.json')['contexts'] if c['job']==context['job'])
    with np.load(source.checked(record['metadata']),allow_pickle=False) as saved:
        for k in saved.files:
            a=saved[k].copy(); rel.base.exact(rel.parent.kin.digest(a),record['hashes'][k])
            if k in d: rel.base.exact(d[k],a)
            else: d[k]=a
    return d,record


def restore_bank(saved,data):
    b=dict(saved); pos=b['donor_positions']
    return dict(b,target=data['target'][pos],weights=data['weights'][pos])


def retrieve(d,context,site,cfg,guard,beat,*,resume=False,verify=False,replay=False,kind='fits',limit=None):
    held=np.flatnonzero(d['sites']==site)
    if limit is not None: held=held[:limit]
    identity=dict(registration_sha256=source.sha(REG),metadata_sha256=source.sha(PUBLIC/'metadata.json'),
        job=context['job'],held_source=str(site),cache=context['cache'],query_ids_hash=rel.parent.kin.digest(d['ids'][held]),
        donor_ids_hash=rel.parent.kin.digest(d['ids'][d['sites']!=site]),limit=limit)
    path=PRIVATE/kind/(context['job']+'_'+str(site)+'.pt.gz'); saved=None
    if path.exists():
        if not(resume or verify or replay): raise ValueError('Explicit resume or verification required')
        saved=rel.base.read_checkpoint(path); rel.base.exact(saved['identity'],identity)
    elif verify or replay: raise ValueError('Cannot verify absent fit')
    if replay:
        if saved['complete']!=len(held): raise ValueError('Incomplete retrieval cannot be replayed')
        bank=restore_bank(saved['bank'],d)
        result=method.from_indices(bank,saved['result']['local_indices'],saved['result']['distances'])
        rel.base.exact(result,saved['result'])
        return result,bank,source.artifact(path),saved['seconds'],held
    guard(); started=time.monotonic()
    bank=method.fit_bank(d['x'],d['target'],d['weights'],d['sites'],d['agents'],d['ids'],site)
    bank_state={k:v for k,v in bank.items() if k not in ('z','target','weights')}
    if len(bank['identities'])<32: raise ValueError('Registered fold lacks32 donor identity buckets; keep NA')
    if saved is not None: rel.base.exact(saved['bank'],bank_state)
    complete=0 if saved is None or verify else saved['complete']
    elapsed=0. if saved is None or verify else saved['seconds']
    chunks=[] if complete==0 else [saved['result']]
    size=31 if verify else cfg['chunk_rows']
    def current(): return {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    for block,start in enumerate(range(complete,len(held),size),1):
        guard(); ix=held[start:start+size]
        chunks.append(method.neighbors(bank,d['x'][ix],d['sites'][ix])); complete=start+len(ix)
        if complete==len(held) or block%cfg['checkpoint_every_chunks']==0:
            result=current(); chunks=[result]
            if not verify:
                path.parent.mkdir(parents=True,exist_ok=True)
                rel.base.save_checkpoint(path,dict(identity=identity,bank=bank_state,result=result,complete=complete,
                    seconds=elapsed+time.monotonic()-started))
            beat(state='retrieval_queries',job=context['job'],source=str(site),complete=complete,total=len(held))
    result=current()
    if verify:
        rel.base.exact(result,saved['result']); seconds=saved['seconds']
    else: seconds=rel.base.read_checkpoint(path)['seconds']
    return result,bank,source.artifact(path),seconds,held


def score(d,context,site,result,bank,ref,seconds,held):
    pos=result['donor_positions']
    if (d['sites'][pos]==site).any() or set(d['recordings'][held]) & set(d['recordings'][pos].ravel()):
        raise ValueError('Query source or recording appeared among references')
    rel.base.exact(d['ids'][pos],result['donor_ids'])
    rel.base.exact(bank['target_scale'],d['target_scale'][held[0]])
    predictions=dict(shared64=d['old_prediction'][held],prior=np.broadcast_to(bank['prior'],(len(held),2)),
        track32=result['prediction'],misaligned32=result['prediction'][method.permutation(d['ids'][held],site)])
    metrics={k:method.base.metrics(p,d['target'][held],d['weights'][held],bank['target_scale']) for k,p in predictions.items()}
    provenance=dict(query_source=str(site),donor_source_counts={s:int((d['sites'][pos]==s).sum()) for s in sorted(set(d['sites'][pos].ravel()))},
        same_source_references=0,same_recording_references=0,distinct_identity_buckets_per_query=32,
        unique_global_reference_rows=int(len(np.unique(result['donor_ids']))),
        donor_bank_identity_buckets=len(bank['identities']))
    return source.stable(dict(source=str(site),checkpoint=ref,seconds=seconds,rows=len(held),
        result_source='fresh_run_source_excluded_nonparametric_risk_model',control_source='cached_verified',
        metrics=metrics,diagnostics=method.diagnostics(result,d['target'][held],d['weights'][held]),
        provenance=provenance,prediction_hashes={k:rel.parent.kin.digest(v) for k,v in predictions.items()},
        boundary=boundary(d['target'][held],predictions['shared64'],predictions['track32'],d['moments'][held],
            d['weights'][held],bank['target_scale'])))


def execute(cfg,reference,guard,beat,*,resume=False,verify=False,replay=False):
    if not source.read(PUBLIC/'pilot.json')['different_chunking_exact']: raise ValueError('Real pilot required')
    contexts=[]
    for context in reference['contexts']:
        d,meta=inputs(context); folds=[]
        for site in sorted(set(d['sites'])):
            r,b,ref,seconds,held=retrieve(d,context,site,cfg,guard,beat,resume=resume,verify=verify,replay=replay)
            record=score(d,context,site,r,b,ref,seconds,held)
            path=PUBLIC/'fits'/(context['job']+'_'+str(site)+'.json')
            if replay: rel.base.exact(source.read(path),record)
            else: source.once(path,record)
            folds.append(record)
        contexts.append(dict(job=context['job'],metadata=meta,audit=context['audit'],folds=folds,summary=method.context_summary(folds)))
    passing=sum(c['summary']['passes_information_screen'] for c in contexts)
    result=source.stable(dict(contexts=contexts,passing_contexts=passing,followup_supported=passing>=2,
        new_V_or_H=False,new_neural_training=False,original_rules_unchanged=True,identity_certification=False,
        original24group_primary='not_estimable_not_passed',independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if replay: rel.base.exact(source.read(PUBLIC/'analysis.json'),result)
    else: source.once(PUBLIC/'analysis.json',result)
    if verify: source.once(PUBLIC/'verification.json',dict(all8_full_retrievals_exact=True,
        source_exclusion_pass=True,different_chunking_exact=True,analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('register','prepare','pilot','run','verify','replay'))
    p.add_argument('--resume',action='store_true'); args=p.parse_args(); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        runtime=previous.audit.old.parent.old.previous.old.run.old.old.previous.teacher.lineage; runtime.PRIVATE=PRIVATE
        start,guard,beat=runtime.runtime({},source.read(CONFIG),args.phase)
        before={str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()} if args.phase=='replay' else None
        cfg,reference=register()
        if args.phase=='register': return
        for f in (*FILES,REG):
            b=source.subprocess.run(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if source.hashlib.sha256(b).hexdigest()!=source.sha(f): raise ValueError('Commit fixed definition before execution')
        receipt=(PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json')
        if receipt.exists(): raise ValueError('Completed phase immutable')
        if args.phase=='prepare': prepare(reference,beat,guard)
        elif args.phase=='pilot':
            context=reference['contexts'][-1]; d,_=inputs(context); site=sorted(set(d['sites']))[-1]
            a,b,r,t,h=retrieve(d,context,site,cfg,guard,beat,resume=args.resume,kind='pilot',limit=cfg['pilot_rows'])
            repeated,*_=retrieve(d,context,site,cfg,guard,beat,verify=True,kind='pilot',limit=cfg['pilot_rows'])
            rel.base.exact(a,repeated)
            source.once(PUBLIC/'pilot.json',dict(different_chunking_exact=True,query_rows=len(h),seconds=t,
                estimated_full_query_seconds=t/len(h)*sum(c['audit']['known_proposed_rows'] for c in reference['contexts']),
                checkpoint=r,donor_rows=len(b['donor_ids']),donor_buckets=len(b['identities'])))
        else:
            if args.phase=='replay':
                def forbidden(*a,**k): raise RuntimeError('Distance search and fitting forbidden during replay')
                method.neighbors=forbidden; method.fit_bank=forbidden
            execute(cfg,reference,guard,beat,resume=args.resume,verify=args.phase=='verify',replay=args.phase=='replay')
        unchanged=None
        if before is not None:
            rel.base.exact(before,{str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()}); unchanged=len(before)
        source.once(receipt,dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,unchanged_public_files=unchanged))
        beat(state='complete')


if __name__=='__main__': main()
