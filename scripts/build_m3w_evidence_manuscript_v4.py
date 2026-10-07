"""Integrate completed temporal and TRAIN diagnostics without new inference."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from scripts import build_m3w_evidence_manuscript_v3 as previous

ROOT, BASE = previous.ROOT, previous.BASE
OUTPUT=BASE/'evidence_manuscript_v4'
SOURCES={
    'temporal_readout':('european_temporal_auxiliary_v1/readout/summary.json','587de5ca7d10721bed56c3d1600e9631efcf038eb3cad50dd750fb1eb7f08ebb'),
    'readout_complete':('european_temporal_auxiliary_v1/readout/complete.json','001a4bfe919865750252a66cfc8c54871e0d55f374e20f7abdf2372d0d937c6d'),
    'training_freeze':('european_temporal_auxiliary_v1/training_freeze.json','04fcfbb74a3c9724aa57b87a86a68b126af51d7424b2360a1301bc12c532d270'),
    'train_diagnostic':('european_temporal_auxiliary_v1/false_safe_train_diagnostic_v1/summary.json','5b540cfb0a8d804bd12a8f6e1fcfe31fd951c453abdaccd618a15741a26fd406'),
    'harm_ratios':('european_temporal_auxiliary_v1/false_safe_train_diagnostic_v1/harm_ratio_audit.json','748e03cda79b302a60fe7efddf6b7ec99dc924d32452988c2e53c0fdade0ff54'),
    'train_evidence':('evidence_manuscript_v3/train_selection_evidence.json','91c7ec3d7463c592057689e09641c0e895cc5fc88cba636ffdefd87627e71f5f'),
}
require=previous.require


def load_sources(root=ROOT):
    docs={}
    for key,(relative,digest) in SOURCES.items():
        raw=(root/BASE/relative).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==digest,'Changed source: '+relative)
        docs[key]=json.loads(raw)
    for relative,digest in docs['train_evidence']['sources'].items():
        require(hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,'TRAIN evidence source changed')
    for collection in ('readout_complete','training_freeze'):
        refs=docs[collection]['groups' if collection=='readout_complete' else 'fits']
        for ref in refs:
            p=(root/ref['path']).resolve()
            require(p.is_relative_to((root/BASE/'european_temporal_auxiliary_v1').resolve()),'Foreign receipt')
            require(hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'],'Changed underlying receipt')
    return previous.load_sources(root),docs


def contrast(value, source_pointer):
    mean,ci,sites=value['mean'],value['CI95'],value['localities']
    require(mean is not None and len(ci)==2 and len(sites)==12,'Complete locality support required')
    require(all(math.isfinite(v) for v in [mean,*ci,*sites.values()]) and ci[0]<=ci[1],'Invalid contrast')
    require(math.isclose(mean,sum(sites.values())/12,abs_tol=1e-12),'Wrong locality weighting')
    require(value['bootstrap_draws']==3000,'Bootstrap contract changed')
    return dict(estimate=mean,ci_low=ci[0],ci_high=ci[1],localities=sites,
                source='temporal_readout',json_pointer=source_pointer,
                result_source='cached_verified',evidence_role='exposed_development')


def build(old_docs,docs):
    old=previous.build(old_docs)
    readout=docs['temporal_readout'];complete=docs['readout_complete'];freeze=docs['training_freeze']
    diagnostic=docs['train_diagnostic'];ratios=docs['harm_ratios'];train_evidence=docs['train_evidence']
    require(readout['groups']==72 and readout['localities']==12 and readout['trained_cost_heads']==216,'Changed population')
    require(readout['advance_to_transfer_design'] is False and not readout['independent_confirmation'],'Do not promote failed experiment')
    require(readout['bootstrap_resamples']==3000 and readout['head_seeds_are_not_forecaster_seeds'],'Incorrect uncertainty unit')
    require(freeze['neural_fits']==216 and len(freeze['fits'])==216,'Full temporal training required')
    require(len(complete['groups'])==72 and complete['summary_sha256']==SOURCES['temporal_readout'][1]
            and complete['training_freeze_sha256']==SOURCES['training_freeze'][1],'Unlinked training/readout')
    require(complete['exact_inference_replay'] and complete['strong_control_replay_exact'],'Missing exact inference checks')
    for row in (readout,complete,diagnostic):
        require(row['deployment_changed'] is False,'Deployment claim changed')
    require(diagnostic['train_resubstitution_only'] and not diagnostic['new_validation_evaluation']
            and diagnostic['new_optimizer_updates']==0 and not diagnostic['independent_confirmation'],'Wrong TRAIN diagnostic role')
    require(diagnostic['source_localities']==12 and diagnostic['views_per_arm']==72,'Wrong TRAIN support')
    require(train_evidence['selected_positive_easy_harm_budget']==.02
            and train_evidence['budget_is_net_easy_ADE_degradation'] is False,'Risk definition changed')
    rows=[]
    for comparator in ('original','additive','poisson','cost','none','rowmean'):
        row={'comparator':comparator}
        for metric in ('all_MSE','full_paired_lower_percent','matched_paired_lower_percent'):
            key=comparator+'_'+metric
            row[metric]=contrast(readout['contrasts'][key],'/contrasts/'+key)
        selected=readout['contrasts'][comparator+'_original_selected_MSE']
        require(selected['mean'] is None and selected['status']=='undefined_support_no_dropping',
                'Do not turn undefined selected-cohort contrast into a result')
        row['original_selected_MSE']=None;rows.append(row)
    train=[]
    for arm in ('none','rowmean','temporal'):
        item=diagnostic['arms'][arm];ratio=ratios['ratios'][arm]
        require(item['views']==72 and item['defined_easy_risk_views']==72
                and item['harm_underpredicted_views']==72,'Changed selected-harm evidence')
        components=item['components']
        values={k:v['equal_locality_mean_pp'] for k,v in components.items()}
        require(math.isclose(values['predicted_slack_mass']+values['harm_underprediction_mass']+
                             values['reference_overprediction_budget_mass'],values['realized_excess_mass'],abs_tol=1e-12),
                'Selected-risk decomposition mismatch')
        for cohort in ('known_all','known_selected'):
            require(ratio[cohort]['defined_views']==72 and ratio[cohort]['undefined_views']==0,'Ratio support mismatch')
        train.append(dict(arm=arm,violations=item['easy_risk_violations'],views=72,
            median_known_risk_percent=item['median_known_easy_risk_percent'],
            median_all_harm_ratio=ratio['known_all']['median_predicted_actual_harm_ratio'],
            median_selected_harm_ratio=ratio['known_selected']['median_predicted_actual_harm_ratio'],
            unknown_selected=item['unknown_selected_occurrences'],components=values,
            source='train_diagnostic',json_pointer='/arms/'+arm,
            evidence_role='TRAIN_resubstitution',result_source='cached_verified'))
    require(sum(diagnostic['arms'][a]['scalar_sum_identity_assertions'] for a in diagnostic['arms'])==432,
            'Do not count scalar summands as independent assertions')
    return dict(assembly_source='fresh_run',numerical_results_source='cached_verified',
        source_hashes={**old['source_hashes'],**{str(BASE/p):h for p,h in SOURCES.values()}},
        prior_evidence=old,temporal_comparisons=rows,temporal_safety=readout['safety'],train_diagnostic=train,
        trained_temporal_cost_heads=216,completed_readout_views=72,localities=12,head_seeds=[17,29,43],
        bootstrap_resamples=3000,train_sum_identity_assertions=432,
        temporal_advance=False,new_training=False,new_inference=False,new_bootstrap=False,
        independent_confirmation=False,risk_calibrated=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False)


def interval(row):
    return f"{row['estimate']:+.6f} [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]"


def table_text(e):
    tables=previous.table_text(e['prior_evidence'])
    rows=['| Comparator | Signed-score MSE | Full paired lower utility | Same-count paired lower utility |',
          '|---|---:|---:|---:|']
    for r in e['temporal_comparisons']:
        rows.append('| '+r['comparator']+' | '+' | '.join(interval(r[k]) for k in
                    ('all_MSE','full_paired_lower_percent','matched_paired_lower_percent'))+' |')
    tables['TEMPORAL_TABLE']='\n'.join(rows)
    rows=['| Auxiliary arm | TRAIN risk violations /72 | Median risk (%) | Median predicted/actual harm: all known | Median predicted/actual harm: selected known | Unknown selected |',
          '|---|---:|---:|---:|---:|---:|']
    for r in e['train_diagnostic']:
        rows.append(f"| {r['arm']} | {r['violations']} | {r['median_known_risk_percent']:.6f} | "
                    f"{r['median_all_harm_ratio']:.6f} | {r['median_selected_harm_ratio']:.6f} | {r['unknown_selected']:,} |")
    tables['TRAIN_TABLE']='\n'.join(rows)
    rows=['| Policy | Selected occurrences | Unknown selected | Undefined easy risk /72 | Upper-risk violations /72 | Complete finite support /72 |',
          '|---|---:|---:|---:|---:|---:|']
    for a in ('original','additive','poisson','cost','none','rowmean','temporal'):
        r=e['temporal_safety'][a]
        rows.append(f"| {a} | {r['selected_occurrences']:,} | {r['unknown_selected_occurrences']:,} | "
                    f"{r['undefined_easy_risk']} | {r['easy_upper_violations']} | {r['finite_completion_supported']} |")
    tables['TEMPORAL_SAFETY_TABLE']='\n'.join(rows)
    return tables


def artifacts(old_docs,docs,template):
    e=build(old_docs,docs);tables=table_text(e);manuscript=template
    for key,value in tables.items():
        marker='{{'+key+'}}';require(manuscript.count(marker)==1,'Missing or duplicate table slot: '+key)
        manuscript=manuscript.replace(marker,value)
    require('{{' not in manuscript,'Unresolved template marker')
    stream=io.StringIO(newline='')
    columns=['comparator','metric','estimate','ci_low','ci_high','source','json_pointer','result_source','evidence_role']
    writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
    for r in e['temporal_comparisons']:
        for metric in ('all_MSE','full_paired_lower_percent','matched_paired_lower_percent'):
            writer.writerow({'comparator':r['comparator'],'metric':metric,
                             **{k:v for k,v in r[metric].items() if k!='localities'}})
    return {'manuscript.md':manuscript,'evidence.json':json.dumps(e,indent=2,allow_nan=False)+'\n',
            'temporal_contrasts.csv':stream.getvalue(),
            'tables.md':'# Separate Development and TRAIN Evidence\n\n'+'\n\n'.join(tables.values())+'\n'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    folder=ROOT/OUTPUT
    content=artifacts(*load_sources(),(folder/'manuscript.template.md').read_text())
    for name,value in content.items():
        path=folder/name
        if a.check:require(path.exists() and path.read_bytes()==value.encode(),'Artifact differs: '+name)
        else:path.write_bytes(value.encode())
    print(json.dumps(dict(status='checked' if a.check else 'exported',new_training=False,new_inference=False,
        source_files=len(SOURCES)+len(previous.SOURCES),verified_fit_receipts=216,verified_readout_groups=72,
        submission_ready=False)))


if __name__=='__main__':
    main()
