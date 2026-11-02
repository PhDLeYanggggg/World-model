"""SOURCE-only frozen-estimator stage diagnosis; never a new deployed policy."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_selection_cost_forest as parent
from src.evaluation.m3w_cost_decode_decomposition import STAGES,prediction_stages,augmented_objective
from src.evaluation.m3w_selection_cost_fit_objective import score

np = parent.np
NAME = 'european_cost_decode_decomposition_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),ROOT/'src/evaluation/m3w_cost_decode_decomposition.py',
    ROOT/'tests/test_m3w_cost_decode_decomposition.py',PUBLIC/'protocol.md')


def admission(register=False):
    cfg,producers,ids = parent.admission()
    previous = json.loads((parent.PUBLIC/'verification.json').read_text())
    if not all(previous[k] for k in ('all18_SOURCE_cost_inferences_exact','all18_selection_cost_inferences_exact','all54_metric_groups_exact')):
        raise ValueError('Complete preceding inference and metric replay required')
    definition = dict(bindings={str(p.relative_to(ROOT)):parent.sha(p) for p in FILES},
        metric_sha256=parent.sha(ROOT/'src/evaluation/m3w_selection_cost_fit_objective.py'),
        parent_registration=parent.artifact(parent.REG),
        parent_training=parent.artifact(parent.PUBLIC/'training_freeze.json'),
        parent_decoded_objective=parent.artifact(parent.PUBLIC/'decoded_SOURCE_objective.json'),
        parent_verification=parent.artifact(parent.PUBLIC/'verification.json'),
        source_cases=18,stages=list(STAGES),source_whitelist_rows=len(ids),
        source_role='original_eight_producer_excluded_SOURCE_localities',
        new_training=False,new_policy=False,development_array_access=False,
        calibration_or_confirmation_access=False,threshold_search=False,
        original_risk_or_primary_changed=False,stage5c_executed=False,smc_enabled=False)
    if register: parent.once(REG,definition)
    if json.loads(REG.read_text()) != definition:
        raise ValueError('Registered diagnostic code or predecessor binding changed')
    if not register:
        for path in (*FILES,REG):
            old = subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)
            if hashlib.sha256(old).hexdigest() != parent.sha(path):
                raise ValueError('Commit the fixed diagnostic before execution')
    return cfg,producers,ids


def summarize(rows,old):
    expected = {(r['trial'],r['arm']) for r in old}
    if len(rows) != 18 or {(r['trial'],r['arm']) for r in rows} != expected:
        raise ValueError('Complete18case fixed roster required')
    output = {}
    for arm in parent.api.ARMS:
        cells = [r for r in rows if r['arm'] == arm]
        agg = {}
        for stage in STAGES:
            total = sum(r['scores'][stage]['total'] for r in cells)/9
            gain = sum(sum(v['query_visits']*v['gain_MSE'] for v in r['scores'][stage]['by_site'].values())
                /r['scores'][stage]['query_visits'] for r in cells)/9
            components = {k:sum(sum(v['query_visits']*v['components'][k]
                for v in r['scores'][stage]['by_site'].values())/r['scores'][stage]['query_visits'] for r in cells)/9
                for k in ('B','H','R','ER','EH')}
            agg[stage] = dict(total=total,gain_MSE=gain,components=components)
        output[arm] = dict(stages=agg,
            mean_raw_augmented_objective=sum(r['augmented_objective']['total'] for r in cells)/9,
            reference_sharing_delta=agg['shared_reference']['total']-agg['unconstrained']['total'],
            projection_delta=agg['projected']['total']-agg['shared_reference']['total'],
            reference_sharing_increase_cases=sum(r['scores']['shared_reference']['total'] > r['scores']['unconstrained']['total'] for r in cells),
            projection_increase_cases=sum(r['scores']['projected']['total'] > r['scores']['shared_reference']['total'] for r in cells))
    return dict(arms=output,cases=18,stage_scores=54,
        all_final_SOURCE_predictions_and_scores_exact=True,
        all_original_target_weight_and_query_bindings_exact=True,
        new_training=False,new_policy=False,development_array_access=False,
        calibration_or_confirmation_access=False,original_failed_primary_unchanged=True,
        fitting_decomposition_not_generalization=True,submission_ready=False,
        stage5c_executed=False,smc_enabled=False)


def run(verify,resume):
    cfg,producers,ids = admission()
    finish = PUBLIC/('verification.json' if verify else 'complete.json')
    if finish.exists(): raise ValueError('Completed diagnostic phase immutable')
    start = time.monotonic(); parent.torch.set_num_threads(4); parent.torch.set_num_interop_threads(1)
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Keep original10GiB disk reserve')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; preserve receipts for same-binding resume')
    def beat(**values):
        guard(); value = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            phase='verify' if verify else 'run',**values)
        parent.write(PRIVATE/'heartbeat.json',value); print(json.dumps(value),flush=True)
    beat(state='runtime_ready',architecture='arm64',cpu_threads=4,interop_threads=1,prediction_threads=1,workers=0)
    source = parent.prior.source_inputs(json.loads(parent.prior.frozen.REG.read_text()))
    models,controls = parent.fitted(),parent.parent.fitted()
    prior_result = json.loads((parent.PUBLIC/'decoded_SOURCE_objective.json').read_text())
    old = {(r['trial'],r['arm']):r for r in prior_result['rows']}
    rows,refs = [],[]
    for producer in producers:
        guard(); data,_,counts = parent.fitting_data(producer,source,ids,controls); c = data['causal']
        if len(set(c['sites'])) != 8 or set(c['sites'])&set(producer['fit_sites']):
            raise ValueError('Original producer-excluded SOURCE only')
        inputs = {k:parent.api.fingerprint(v) for k,v in dict(x=c['x'],envelope=c['envelope'],scales=c['scales'],
            allowed=c['allowed'],ids=c['ids'],truth=data['truth'].numpy(),query_counts=counts).items()}
        for arm in parent.api.ARMS:
            guard(); key = (producer['trial'],arm); saved = models[key]
            path = PUBLIC/'cases'/(producer['trial']+'_'+arm+'.json')
            if resume and not verify and path.exists():
                result = json.loads(path.read_text())
                if result['input_hashes'] != inputs or result['checkpoint'] != saved['checkpoint'] or result['registration_sha256'] != parent.sha(REG):
                    raise ValueError('Same-binding diagnostic resume required')
            else:
                state = parent.joblib.load(parent.checked(saved['checkpoint']))
                native,aug = prediction_stages(state,c)
                final_hash = parent.api.fingerprint(native['projected'])
                if final_hash != saved['SOURCE_prediction_sha256'] or final_hash != old[key]['prediction_sha256']:
                    raise ValueError('Final stage must exactly reproduce the frozen SOURCE prediction')
                scores = {s:score(native[s],data['truth'].numpy(),c['allowed'],data['known'],c['scales'],
                    data['preprocess']['rms'],c['groups'],counts,c['sites']) for s in STAGES}
                if scores['projected'] != old[key]['score']:
                    raise ValueError('Final complete decoded score differs from predecessor')
                measured = augmented_objective(aug,data['truth'].numpy(),c['allowed'],data['known'],data['preprocess']['rms'],c['groups'],counts)
                for name,digest in measured['matrix_hashes'].items():
                    if digest != state['input_hashes'][name]:
                        raise ValueError('Original augmented target or candidate/query weight mismatch')
                np.testing.assert_allclose(measured['total'],scores['unconstrained']['total'],rtol=1e-5,atol=1e-6)
                result = dict(trial=producer['trial'],arm=arm,checkpoint=saved['checkpoint'],input_hashes=inputs,
                    SOURCE_sites=sorted(set(c['sites'])),scores=scores,augmented_objective=measured,
                    prediction_hashes={s:parent.api.fingerprint(native[s]) for s in STAGES},
                    final_SOURCE_prediction_and_score_exact=True,
                    reference_aliases_in_original_all7_mean=True,
                    unconstrained_reference_is_diagnostic_not_a_legal_policy=True,
                    registration_sha256=parent.sha(REG))
                if verify:
                    if result != json.loads(path.read_text()): raise ValueError('Full SOURCE stage replay differs')
                else: parent.once(path,result)
                del state,native,aug
            rows.append(result); refs.append(parent.artifact(path))
            beat(state='SOURCE_case_complete',cases=len(rows),total=18,trial=producer['trial'],arm=arm)
        del data,c
    selected_old = [r for r in prior_result['rows'] if r['arm'] in parent.api.ARMS]
    summary = summarize(rows,selected_old)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Summary replay differs')
    else: parent.once(PUBLIC/'summary.json',summary)
    parent.once(finish,dict(result_source='fresh_SOURCE_three_stage_inference_and_scoring_no_new_training',
        pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        registration_sha256=parent.sha(REG),cases=refs,summary_sha256=parent.sha(PUBLIC/'summary.json'),
        all18_SOURCE_inferences_and54_stage_scores_exact=bool(verify),
        original_failed_development_primary_unchanged=True,private_independent_arrays_read=False))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','run','verify')); parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(); PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register': admission(True)
        else: run(args.phase == 'verify',args.resume)
