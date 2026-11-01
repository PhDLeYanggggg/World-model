"""Post-readout SOURCE support diagnosis; no new choices,training or thresholds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_current_bank_deferral as run
from src.evaluation.m3w_current_guard_support import profile

REG=run.PUBLIC/'guard_diagnostic_registration.json'
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_current_guard_support.py',ROOT/'tests/test_m3w_current_guard_support.py',run.PUBLIC/'guard_diagnostic_protocol.md')


def admission():
    bindings={str(p.relative_to(ROOT)):run.sha(p) for p in FILES}
    artifacts={str((run.PUBLIC/n).relative_to(ROOT)):run.sha(run.PUBLIC/n) for n in ('registration.json','complete.json','summary.json','verification.json')}
    run.once(REG,dict(bindings=bindings,artifacts=artifacts,scope='post_readout_exploratory_SOURCE_label_aware_diagnostic',
        no_training_no_new_policy_no_threshold_change=True,full72groups_required=True,original2percent_contract_unchanged=True,
        independent_roles_opened=False,deployment_changed=False))
    for rel,digest in dict(bindings,**artifacts).items():
        if json.loads(REG.read_text()).get('bindings',{}).get(rel,json.loads(REG.read_text())['artifacts'].get(rel))!=digest:
            raise ValueError('Diagnostic definition or parent readout changed')
    return run.sha(REG)


def main(register=False,resume=False):
    binding=admission()
    if register:return
    for path in (*FILES,REG):run.parent.previous.parent.parent.source.frozen.BASE.engine.committed(path)
    cfg,values,fits,fitted=run.admission();start,beat,_=run.runtime(cfg,'guard_support_diagnostic')
    output=run.PUBLIC/'guard_support_diagnostic.json';progress=run.PRIVATE/'guard_support_progress.json'
    if output.exists():raise ValueError('Completed diagnostic immutable')
    saved=dict(binding=binding,groups={})
    if progress.exists():
        if not resume:raise ValueError('Explicit diagnostic resume required')
        saved=json.loads(progress.read_text())
        if saved['binding']!=binding:raise ValueError('Diagnostic resume definition changed')
    old=run.parent.aggregate_groups(run.NAME)
    for context,seed,data,ranges,packets,_,_,risk in run.inputs(values,fits,fitted,beat):
        for source,p in sorted(packets.items()):
            key=f'{context}_{source}_seed{seed}_fit.json';at=ranges[source]
            if run.api.base.digest(p['ids'])!=old[key]['ids_sha256']:raise ValueError('Original rows changed')
            records,mask=profile(risk[at],p['y'],p['allowed'],p['scales'],p['recordings'],p['frames'])
            # Probe every guard bit with causal scores,not a future label.
            for action in range(7):
                score=run.api.np.zeros((len(p['ids']),8));score[:,action+1]=1
                actual=run.api.choices(score,p['allowed'],risk[at])['guarded'][0]
                run.api.np.testing.assert_array_equal(actual==action,mask[:,action])
            row=dict(context=context,source=source,seed=seed,ids_sha256=old[key]['ids_sha256'],recordings=records)
            if key in saved['groups'] and row!=saved['groups'][key]:raise ValueError('Resumed diagnostic differs')
            saved['groups'][key]=row;run.write(progress,saved)
            beat(state='guard_support_group_complete',groups=len(saved['groups']),total=72)
    cells=[dict(context=r['context'],source=r['source'],seed=r['seed'],recording=rec,**v)
        for r in saved['groups'].values() for rec,v in r['recordings'].items()]
    if len(saved['groups'])!=72 or len(cells)!=444:raise ValueError('Full72groups444record views required')
    blocked=[r for r in cells if r['frozen_pointwise_guard_cannot_supply_known_easy_reference']]
    singleton=[r for r in cells if r['frozen_native_joint_cannot_supply_known_easy_reference_singleton_proof']]
    run.once(output,dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,result_source='fresh_run_label_aware_SOURCE_diagnostic',
        groups=72,record_views=444,pointwise_support_blocked_views=len(blocked),native_joint_singleton_proof_views=len(singleton),
        unique_pointwise_blocked_records=len({(r['source'],r['recording']) for r in blocked}),
        blocked_views=blocked,recordings=cells,registration_sha256=binding,
        no_new_training=True,no_new_policy=True,no_threshold_change=True,diagnostic_future_labels_used=True,
        oracle_is_not_a_model_result=True,independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    beat(state='guard_support_diagnostic_complete',blocked_views=len(blocked),singleton_proofs=len(singleton))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    with (run.PRIVATE/'guard_support.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args.register,args.resume)
