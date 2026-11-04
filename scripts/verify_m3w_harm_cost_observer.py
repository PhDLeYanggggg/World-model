"""Observe held costs only after unchanged causal decisions during exact replay."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_harm_refinement as run
from scripts import resume_m3w_harm_inference_local as launch
from scripts.report_m3w_moving_harm_refinement import factorization,text_once
from src.evaluation.m3w_selected_cost_audit import selected_costs

source=run.source
PUBLIC=run.PUBLIC/'held_cost_check'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),PUBLIC/'protocol.md',ROOT/'src/evaluation/m3w_selected_cost_audit.py',
    ROOT/'scripts/report_m3w_moving_harm_refinement.py')


def register():
    if not(run.PUBLIC/'summary.json').exists():raise ValueError('Complete the full held matrix first')
    source.once(REG,source.stable(dict(bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        scientific_registration=source.artifact(run.REG),complete_held_summary=source.artifact(run.PUBLIC/'summary.json'),
        fitting_diagnosis=source.artifact(run.PUBLIC/'fitting_check/analysis.json'),
        expected_cells=72,post_decision_only=True,new_training=False,threshold_search=False,
        changed_primary=False,independent_confirmation=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','verify'));args=parser.parse_args();register()
    if args.phase=='register':return
    for path in (*FILES,REG):
        old=source.subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if source.hashlib.sha256(old).hexdigest()!=source.sha(path):raise ValueError('Commit observer definition first')
    original_register,original_allocate=run.register,source.allocated;context={};cells=[]
    def admitted(beat):
        result=original_register(beat);context['data']=result[2];return result
    def observed(prediction,causal,packet,job,arm,cfg,beat,*,verify=False):
        if not verify:raise ValueError('Observer cannot create or alter an allocation')
        choices,report=original_allocate(prediction,causal,packet,job,arm,cfg,beat,verify=True)
        before={p:source.array_hash(v)for p,v in choices.items()};cost_hash=source.array_hash(prediction)
        data=context['data'];ids=packet['ids']
        truth=run.api.parent.targets(packet,data['target_eval'][ids]-data['origin'][ids,None],data['valid'][ids],job['easy_cut'])
        policies={}
        for policy,choice in choices.items():
            all_scores=selected_costs(prediction,truth['moments'],truth['observed_future'],choice,causal['groups'])
            records={}
            for record in sorted(set(packet['recordings'])):
                at=packet['recordings']==record
                records[str(record)]=selected_costs(prediction[at],truth['moments'][at],truth['observed_future'][at],
                    choice[at],[source.np.arange(int(at.sum()))])
            policies[policy]=dict(query_scores=all_scores,recordings=records)
        if before!={p:source.array_hash(v)for p,v in choices.items()}or cost_hash!=source.array_hash(prediction):
            raise ValueError('Observation must not mutate decisions or predictions')
        cost_model='frozen'if arm=='amplitude_bank'else arm
        path=PUBLIC/'cells'/(job['key']+'_'+cost_model+'.json')
        source.once(path,source.stable(dict(job=job['key'],cost_model=cost_model,policies=policies,
            choice_hashes=before,prediction_sha256=cost_hash,causal_decisions_unchanged=True,
            evaluation_labels_read_after_allocation_only=True)))
        cells.append(source.artifact(path))
        return choices,report
    run.register=admitted;source.allocated=observed
    sys.argv=[sys.argv[0],'verify'];launch.main()
    if len(cells)!=72:raise ValueError('All72post-decision diagnostic cells required')
    rows=[source.read(source.checked(ref))for ref in cells];aggregate={}
    for arm in run.ARMS:
        aggregate[arm]={}
        for policy in ('independent','frame_uniform','joint_matched','joint_free'):
            values=[r['policies'][policy]['query_scores']for r in rows if r['cost_model']==arm]
            actual=source.np.sum([r['realized_selected_known_moments']for r in values],0)
            predicted=source.np.sum([r['predicted_on_selected_known_moments']for r in values],0)
            aggregate[arm][policy]=dict(selected=sum(r['selected']for r in values),
                selected_known=sum(r['selected_known']for r in values),**factorization(actual,predicted))
    source.once(PUBLIC/'analysis.json',source.stable(dict(cells=cells,aggregate=aggregate,
        all72causal_decisions_and_cost_predictions_unchanged=True,all_labels_post_decision=True,
        result_source='fresh_run_post_decision_diagnostic_during_exact_replay_cached_verified_weights_and_choices',
        independent_confirmation=False,new_training=False,changed_primary=False)))
    lines=['# Held selected costs, post-decision only','',
        'All72cells are observed during the unchanged complete exact replay. All causal decisions and prediction hashes remain identical. Labels enter only these post-decision diagnostics.',
        '', '|Cost model, joint-free|Selected repeated rows|Known selected|H actual/predicted|EH actual/predicted|R actual/predicted|ER actual/predicted|',
        '|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'undefined'if x is None else f'{x:.6f}'
    for arm in run.ARMS:
        r=aggregate[arm]['joint_free'];m=r['moments']
        lines.append(f"|{arm}|{r['selected']}|{r['selected_known']}|"+'|'.join(fmt(m[k]['ratio'])for k in ('H','EH','R','ER'))+'|')
    lines+=['','These pooled ratios do not replace complete per-record risk checks or identify a unique cause of domain shift. The fitting and held selected populations differ. All per-recording costs and four causal policies are retained in the cell receipts. No independent calibration, model selection, metric/seconds, Stage5C or SMC claim.']
    text_once(PUBLIC/'readout.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
