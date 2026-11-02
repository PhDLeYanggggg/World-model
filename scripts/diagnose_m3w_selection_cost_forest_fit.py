"""Supplementary SOURCE-only decoded-objective diagnosis, not a new primary."""
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_selection_cost_forest as run
from src.evaluation.m3w_selection_cost_fit_objective import score


def main():
    cfg,producers,ids = run.admission(); beat,guard,receipt = run.runtime(cfg,'decoded_SOURCE_objective')
    finish = run.PUBLIC/'decoded_SOURCE_objective.json'
    if finish.exists(): raise ValueError('Completed SOURCE diagnostic immutable')
    models = run.fitted(); controls = run.parent.fitted()
    source = run.prior.source_inputs(json.loads(run.prior.frozen.REG.read_text())); rows = []
    for row in producers:
        guard(); data,case,counts = run.fitting_data(row,source,ids,controls); c = data['causal']
        for arm in ('mean_absolute',*run.api.ARMS):
            if arm == 'mean_absolute':
                ref = controls[row['trial'],arm]['checkpoint']
                p = run.prior.api.predict(run.prior.api.read_checkpoint(run.checked(ref)),c)
            else:
                ref = models[row['trial'],arm]['checkpoint']
                p = run.api.predict(run.joblib.load(run.checked(ref)),c)
            value = score(p,data['truth'].numpy(),c['allowed'],data['known'],c['scales'],
                          data['preprocess']['rms'],c['groups'],counts,c['sites'])
            rows.append(dict(trial=row['trial'],arm=arm,checkpoint=ref,score=value,
                prediction_sha256=run.api.fingerprint(p),query_counts_sha256=run.api.fingerprint(counts)))
            beat(state='SOURCE_objective_scored',models=len(rows),total=27); del p
        del data,case
    expected = {(r['trial'],a) for r in producers for a in ('mean_absolute',*run.api.ARMS)}
    if {(r['trial'],r['arm']) for r in rows} != expected or len(rows) != 27:
        raise ValueError('Complete nine-producer three-estimator comparison required')
    result = dict(**receipt(),rows=rows,source_only=True,training_or_policy_changed=False,
        supplementary_post_registration_diagnostic_not_new_primary=True,
        fitting_resubstitution_not_generalization=True,calibration_or_confirmation_access=False,
        script_sha256=run.sha(Path(__file__)),metric_code_sha256=run.sha(ROOT/'src/evaluation/m3w_selection_cost_fit_objective.py'))
    run.once(finish,result)
    means = {a:sum(r['score']['total'] for r in rows if r['arm'] == a)/9 for a in ('mean_absolute',*run.api.ARMS)}
    print(json.dumps(dict(complete_SOURCE_models=27,mean_decoded_SOURCE_objective=means)))


if __name__ == '__main__':
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
