"""Direct full refit of the actual 96-tree interrupted producer, not efficacy."""
import fcntl
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_selection_cost_forest as run


def main():
    cfg,producers,ids = run.admission()
    start = time.monotonic()
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve original10GiB reserve')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('Original12h boundary; resume same verification')
    def beat(**values):
        guard()
        row = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            phase='actual_interruption_direct_refit',**values)
        run.write(run.PRIVATE/'actual_interruption_verification_heartbeat.json',row)
        print(json.dumps(row),flush=True)
    def receipt():
        return dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            registration_sha256=run.sha(run.REG))
    finish = run.PUBLIC/'actual_interruption_refit.json'
    if finish.exists():
        raise ValueError('Completed actual-interruption verification immutable')
    recovery = json.loads((run.PUBLIC/'turn_interrupt_recovery.json').read_text())
    snapshot_path = ROOT/recovery['private_snapshot']
    if run.sha(snapshot_path) != recovery['saved_checkpoint_sha256']:
        raise ValueError('Original 96-tree checkpoint snapshot changed')
    row = next(r for r in producers if r['trial'] == 'single1_seed29')
    arm = 'context_forest'
    completed = json.loads((run.PUBLIC/'fits'/(row['trial']+'_'+arm+'.json')).read_text())
    if completed['trees'] != 128 or completed['registration_sha256'] != run.sha(run.REG):
        raise ValueError('Actually interrupted producer must already have a complete immutable fit receipt')
    snapshot = run.joblib.load(snapshot_path)
    if snapshot['trees'] != 96 or snapshot['settings']['trees'] != 128:
        raise ValueError('Verify actual interrupted96/128state, not another pilot')
    source = run.prior.source_inputs(json.loads(run.prior.frozen.REG.read_text()))
    data,_,counts = run.fitting_data(row,source,ids,run.parent.fitted())
    matrix = run.api.training_matrix(data,counts,arm)
    direct = run.one(matrix,data,row,arm,cfg,beat,guard,family='actual_interruption_direct_refit',
        resume=run.fit_path(row,arm,'actual_interruption_direct_refit').exists())
    recovered = run.joblib.load(run.checked(completed['checkpoint']))
    run.api.exact_forest(recovered,direct)
    for k in ('input_hashes','identity','settings','preprocess','seed','arm'):
        run.prior.api.exact(snapshot[k],direct[k])
    run.prior.api.exact(snapshot['trace'],direct['trace'][:len(snapshot['trace'])])
    for old,new in zip(snapshot['model'].estimators_,direct['model'].estimators_):
        if old.random_state != new.random_state:
            raise ValueError('Interrupted-prefix random states changed')
        run.prior.api.exact(old.tree_.__getstate__(),new.tree_.__getstate__())
    run.once(finish,dict(**receipt(),result_source='fresh_direct128_refit_of_actually_interrupted96_tree_model',
        trial=row['trial'],arm=arm,retained_prefix_trees=96,full_trees=128,
        actual_recovery_vs_direct_full_forest_exact=True,interrupted96_tree_prefix_exact=True,
        original96_snapshot_sha256=run.sha(snapshot_path),recovery_receipt=run.artifact(run.PUBLIC/'turn_interrupt_recovery.json'),
        recovered=completed['checkpoint'],
        direct=run.artifact(run.fit_path(row,arm,'actual_interruption_direct_refit')),
        extra_reproduction_trees=128,development_labels_read=False,calibration_or_confirmation_access=False,
        policy_or_risk_changed=False,script_sha256=run.sha(Path(__file__))))
    beat(state='actual_interruption_recovery_exact',trees=128)


if __name__ == '__main__':
    with (run.PRIVATE/'actual_interruption_verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
