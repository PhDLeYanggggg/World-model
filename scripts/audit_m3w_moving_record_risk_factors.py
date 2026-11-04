"""Fresh fixed-decision factor arithmetic on cached-verified source costs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system()=='Darwin'and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 analysis environment required')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.intake_m3w_movingdronecrowd import sha,once,atomic
from src.evaluation.m3w_fixed_risk_factor_audit import factors,aggregate,FACTORS

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_record_risk_factors_v1'
PARENT=ROOT/'outputs/publication_readiness_2026_09/moving_shared_harm_refinement_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_record_risk_factors_v1'
CONFIG=ROOT/'configs/m3w_moving_record_risk_factors_v1.json'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/evaluation/m3w_fixed_risk_factor_audit.py',
    ROOT/'tests/test_m3w_fixed_risk_factor_audit.py')


def read(path):return json.loads(path.read_text())


def artifact(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def checked(ref):
    path=ROOT/ref['path']
    if sha(path)!=ref['sha256']:raise ValueError('Changed frozen diagnostic input')
    return path


def register():
    cfg=read(CONFIG);summary=read(PARENT/'summary.json');verification=read(PARENT/'verification.json')
    if (verification['summary_sha256']!=sha(PARENT/'summary.json')
            or not verification['all54fresh_cost_outputs_and_all108choice_scalar_blocks_exact']
            or cfg['positive_harm_budget']!=.02 or cfg['factors']!=list(FACTORS)
            or any(cfg[k]for k in ('new_training','new_inference','new_decisions','threshold_search',
                'independent_confirmation','stage5c_executed','smc_enabled'))):
        raise ValueError('Keep the complete parent, exact budget and diagnostic-only role')
    refs=summary['group_receipts']+summary['cost_diagnostic_receipts']
    for ref in refs:checked(ref)
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p)for p in FILES},configuration=cfg,
        parent_summary=artifact(PARENT/'summary.json'),parent_verification=artifact(PARENT/'verification.json'),
        inputs=refs,result_source='fresh_arithmetic_cached_verified_costs_no_new_inference',
        post_readout_diagnostic=True,independent_confirmation=False,stage5c_executed=False,smc_enabled=False))
    return cfg,summary


def validate(cost,original):
    if cost['selected']!=original['selected']or cost['selected_unknown']!=original['unknown_selected']:
        raise ValueError('All selected counts must match the original fixed policy')
    for x,y in (('known_realized_all_risk','known_all_risk'),('known_realized_easy_risk','known_easy_risk'),
        ('completion_all_risk_upper','completion_all_risk_upper'),('completion_easy_risk_upper','completion_easy_risk_upper')):
        a,b=cost[x],original[y]
        if (a is None)!=(b is None)or a is not None and not np.isclose(a,b,rtol=1e-12,atol=1e-12):
            raise ValueError('Cached cost moments and original risk readout must agree')


def calculate(cfg,parent):
    groups={read(checked(ref))['job']['key']:read(checked(ref))for ref in parent['group_receipts']}
    rows=[];cells=[]
    for ref in parent['cost_diagnostic_receipts']:
        cost=read(checked(ref));job=cost['job'];g=groups[job];identity=g['job']
        scopes=[set(identity[k])for k in ('producer_sites','critic_sites','evaluation_sites')]
        if any(scopes[a]&scopes[b]for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Preserve original source exclusion')
        part=[]
        for arm in cfg['arms']:
            for policy in cfg['policies']:
                expected={record:(r['site'],value)for r in g['rows']for record,value in
                    r['policies'][arm][policy]['recordings'].items()}
                observed=cost['diagnostics'][arm][policy]['recordings']
                if set(expected)!=set(observed):raise ValueError('Every original recording must be retained')
                for record,value in observed.items():
                    site,original=expected[record];validate(value,original)
                    row=dict(job=job,seed=identity['seed'],site=site,recording=record,arm=arm,policy=policy,
                        selected=value['selected'],known=value['selected_known'],unknown=value['selected_unknown'],
                        factors=factors(value),original_supported=original['supported'],
                        original_signed_easy_net=original['known_easy_net_degradation'])
                    rows.append(row)
                    if policy==cfg['primary_diagnostic_policy']:part.append(row)
        path=PUBLIC/'primary_records'/(job+'.json');once(path,dict(rows=part,input_cost=ref,input_group=artifact(PARENT/'groups'/(job+'.json'))))
        cells.append(artifact(path))
        atomic(PRIVATE/'heartbeat.json',dict(pid=__import__('os').getpid(),job=job,complete_jobs=len(cells),total=18,
            result_source='fresh_arithmetic_cached_verified_inputs'))
    result={}
    for arm in cfg['arms']:
        result[arm]={}
        for policy in cfg['policies']:
            take=[r for r in rows if r['arm']==arm and r['policy']==policy]
            if len(take)!=348:raise ValueError('Retain all348repeated recording views per arm/policy')
            result[arm][policy]={kind:aggregate(take,kind)for kind in ('all','easy')}
            result[arm][policy]['original_full_support']=sum(r['original_supported']for r in take)
            result[arm][policy]['unchanged_original_gate']=True
    if len(rows)!=4176 or len(cells)!=18:raise ValueError('Complete18jobs x3arms x4policies required')
    once(PUBLIC/'analysis.json',dict(results=result,primary_records=cells,all4176original_count_and_risk_blocks_match=True,
        all8352risk_excess_decompositions_valid=True,all16subsets_evaluated_per_component=True,
        original24group_primary='not_estimable_not_passed',independent_confirmation=False,new_training=False,
        new_inference=False,new_decisions=False,threshold_search=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();cfg,parent=register()
    if args.phase=='register':return
    for path in (*FILES,REG):
        import subprocess
        b=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(b).hexdigest()!=sha(path):raise ValueError('Commit definition before factor counts')
    receipt=PUBLIC/(args.phase+'_execution.json')
    if receipt.exists():raise ValueError('Complete diagnostic phase is immutable')
    if args.phase=='verify'and not(PUBLIC/'run_execution.json').exists():raise ValueError('Complete run before replay')
    calculate(cfg,parent)
    once(receipt,dict(pid=__import__('os').getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_training=False,new_inference=False))
    if args.phase=='verify':once(PUBLIC/'verification.json',dict(all8352decompositions_and_all_counts_exact=True,
        analysis_sha256=sha(PUBLIC/'analysis.json'),new_decisions=False,independent_confirmation=False))
    print(json.dumps(dict(phase=args.phase,complete=True,seconds=time.monotonic()-start)))


if __name__=='__main__':main()
