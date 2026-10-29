"""Post-primary mechanism control: preserve recording/action-support marginals."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_coupled_local_cost as run
from src.evaluation import m3w_multiaction_cost_readout as base

REG=run.PUBLIC/'stratified_registration_v2.json'
FILES=(Path(__file__),ROOT/'tests/test_m3w_coupled_local_shuffle.py',run.PUBLIC/'stratified_protocol.md')


def stratified_shuffle(target,known,recordings,allowed):
    result=target.copy();counts=dict(known_rows=int(known.sum()),permutable_rows=0,changed_rows=0,groups=0,singletons=0)
    code=np.sum(allowed.astype(np.int64)*(1<<np.arange(7)),1)
    for recording in sorted(set(recordings.tolist())):
        for pattern in sorted(set(code[known & (recordings==recording)].tolist())):
            ids=np.flatnonzero(known & (recordings==recording) & (code==pattern))
            salt=int.from_bytes(hashlib.sha256(f'20261029|{recording}|{pattern}'.encode()).digest()[:8],'little')
            result[ids]=target[np.random.default_rng(salt).permutation(ids)]
            np.testing.assert_allclose(result[ids].sum(0),target[ids].sum(0),rtol=1e-12,atol=1e-8)
            counts['groups']+=1;counts['singletons']+=int(len(ids)==1)
            counts['permutable_rows']+=len(ids) if len(ids)>1 else 0
    counts['changed_rows']=int(np.any(result!=target,axis=(1,2)).sum())
    np.testing.assert_array_equal(result[~known],target[~known])
    return result,counts


def admission(register=False):
    cfg,values,learned=run.admission();check=json.loads((run.PUBLIC/'verification.json').read_text())
    if not check['exact_replay'] or check['summary_sha256']!=run.sha(run.PUBLIC/'summary.json'):
        raise ValueError('Completed primary exact replay required')
    run.once(REG,dict(bindings={str(p.relative_to(ROOT)):run.sha(p) for p in FILES},
        pre_run_draft_sha256=run.sha(run.PUBLIC/'stratified_registration.json'),
        amendment='memoize_identical_source_counts_before_any_diagnostic_inference_no_numeric_rule_change',
        primary_summary_sha256=run.sha(run.PUBLIC/'summary.json'),
        primary_verification_sha256=run.sha(run.PUBLIC/'verification.json'),
        scope='posthoc_fitting_mechanism_not_new_primary',expected_source_contexts=24,
        shuffle_strata='donor_recording_and_causal_seven_action_eligibility_pattern',
        no_new_threshold_or_optimizer_updates=True,independent_roles_opened=False,deployment_changed=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT/rel)!=digest:
                raise ValueError('Registered stratified mechanism changed')
            run.parent.parent.parent.committed(ROOT/rel)
        run.parent.parent.parent.committed(REG)
        run.parent.parent.parent.committed(run.PUBLIC/'verification.json')
    return cfg,values,learned


def execute(verify=False):
    cfg,values,learned=admission();started,beat,guard=run.runtime(cfg);rows=[];freezes=[]
    for context,items in sorted(values[2]['contexts'].items()):
        packets=run.parent.packets(context,items,values,beat)
        state=run.parent.api.base.read_checkpoint(ROOT/learned[context,17,'physical_neural','known_only']['checkpoint']['path'])
        bank,spans=run.arrays(packets,state['preprocess'])
        queries,nn,refs=run.retrieve(bank,context,cfg,beat,resume=True)
        if run.parent.parent.fingerprint(nn)!=json.loads((run.PUBLIC/'prediction_freeze'/f'{context}.json').read_text())['neighbor_sha256']:
            raise ValueError('Primary neighbors changed')
        original,support=run.cost_predictions(bank,queries,nn)
        target,counts=stratified_shuffle(bank['target'],bank['known'],bank['recordings'],bank['allowed'])
        counts_by_source={s:int((bank['causal'] & (bank['sites']==s)).sum()) for s in sorted(set(bank['sites']))}
        weights=np.array([1/counts_by_source[s] for s in bank['sites']])
        new=np.zeros((len(bank['ids']),7,5));yes=np.zeros((len(bank['ids']),7),bool)
        old=np.zeros_like(new);oldyes=yes.copy();old[queries]=original['local_bounds'];oldyes[queries]=support['local_bounds']
        for start in range(0,len(queries),256):
            ix=queries[start:start+256];donors=nn[start:start+256]
            new[ix],yes[ix]=run.api.estimate(target[donors],bank['known'][donors],bank['allowed'][donors],weights[donors],
                bank['envelope'][ix],bank['scales'][ix],bounds=True)
        if run.parent.parent.fingerprint(old)!=json.loads((run.PUBLIC/'prediction_freeze'/f'{context}.json').read_text())['prediction_sha256']['local_bounds']:
            raise ValueError('Primary predictions changed')
        freezes.append(dict(context=context,target_sha256=run.parent.parent.fingerprint(target),counts=counts,
            prediction_sha256=run.parent.parent.fingerprint(new),same_neighbors=True))
        for source,packet in sorted(packets.items()):
            guard();lo,hi=spans[source]
            a,ar=run.parent.api.base.select_actions(new[lo:hi],packet['allowed'] & yes[lo:hi],arm='physical_neural')
            b,br=run.parent.api.base.select_actions(old[lo:hi],packet['allowed'] & oldyes[lo:hi],arm='physical_neural')
            for seed in cfg['comparison_seeds']:
                primary=json.loads((run.PUBLIC/'groups'/f'{context}_{source}_seed{seed}_local_bounds_fit.json').read_text())
                if run.parent.parent.fingerprint(b)!=primary['action_sha256']['physical_neural']:
                    raise ValueError('Primary action hash changed')
            pair=run.score.paired(packet,a,b,ar,br)
            risks={}
            for recording in sorted(set(packet['recordings'])):
                ix=packet['recordings']==recording
                risks[str(recording)]={n:base.compact_score(packet['y'][ix],packet['envelope'][ix],c[ix])
                    for n,c in dict(primary=b,stratified=a).items()}
            rows.append(dict(context=context,source=source,seed=0,result=dict(contrasts=pair,recordings=risks),
                action_sha256=dict(primary=run.parent.parent.fingerprint(b),stratified=run.parent.parent.fingerprint(a))))
        beat(state='stratified_context_complete',context=context,groups=len(rows),total=24)
        del packets,bank,original,target,new,old
    expected={(c,i['packet']['identity']['source']) for c,items in values[2]['contexts'].items() for i in items}
    if len(rows)!=24 or {(r['context'],r['source']) for r in rows}!=expected:
        raise ValueError('Complete24 fitting source/context groups required')
    bootstrap=dict(values[1],seeds=[0])
    summary=dict(primary_over_stratified={m+'_'+b:base.interval(rows,lambda r,mode=m,bound=b:r['contrasts'][mode][bound],bootstrap)
        for m in ('full','matched') for b in ('known_pp','lower_pp','upper_pp')},safety={})
    for name in ('primary','stratified'):
        records=[v[name] for r in rows for v in r['result']['recordings'].values()]
        summary['safety'][name]=dict(required_record_views=len(records),supported_record_views=sum(p['budget_status']['supported'] for p in records),
            undefined_easy_record_views=sum(p['easy_selected_risk_upper'] is None for p in records),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk']>.02+1e-12 for p in records))
    value=dict(rows=rows,freeze=freezes,summary=summary,result_source='fresh_posthoc_fitting_stratified_target_control',
        primary_unchanged=True,original_primary_gates_replaced=False,held_scored=False,independent_roles_opened=False,
        deployment_changed=False,new_optimizer_updates=0,stage5c_executed=False,smc_enabled=False)
    path=run.PUBLIC/'stratified_diagnosis.json'
    if verify:
        if value!=json.loads(path.read_text()):
            raise ValueError('Exact stratified mechanism replay differs')
        run.once(run.PUBLIC/'stratified_verification.json',dict(exact_replay=True,groups=24,diagnosis_sha256=run.sha(path)))
    else:
        run.once(path,value)
    run.write(run.PRIVATE/('stratified_verify_runtime.json' if verify else 'stratified_runtime.json'),
        dict(pid=run.os.getpid(),seconds=time.monotonic()-started,peak_RSS_bytes=run.parent.rss_bytes()))
    beat(state='stratified_verified' if verify else 'stratified_complete',groups=24)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','verify'));args=parser.parse_args()
    run.PRIVATE.mkdir(parents=True,exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        else:execute(args.phase=='verify')
