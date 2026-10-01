"""Resource-only retry: verify actual tree storage layout, preserve original fits."""
import fcntl
import json
import math
import shutil
import time
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_source_forest as run


def main():
    cfg, reg = run.registration()
    assert json.loads((run.PUBLIC/'registration.json').read_text()) == reg
    run.base.inter.committed(run.PUBLIC/'registration.json')
    amendment = run.PUBLIC/'resource_amendment.json'
    expected = dict(original_registration_sha256=run.digest(run.PUBLIC/'registration.json'),
        wrapper_sha256=run.digest(Path(__file__)),
        amendment_sha256=run.digest(run.PUBLIC/'resource_amendment.md'),
        original_pilot_sha256=run.digest(run.PUBLIC/'pilot.json'))
    assert json.loads(amendment.read_text()) == expected
    run.base.inter.committed(amendment)
    pilot = json.loads((run.PUBLIC/'pilot.json').read_text())
    assert pilot['local_memory_feasible']
    run.core.torch.set_num_threads(4);run.core.torch.set_num_interop_threads(1)
    with (run.PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = run.joblib.load(run.PRIVATE/'heads'/pilot['group']/'checkpoint.joblib')
        layouts=[]
        for t in state['model'].estimators_:
            arrays=t.tree_.__getstate__()
            layouts.append(arrays['nodes'].dtype.itemsize+arrays['values'][0].nbytes)
        assert len(set(layouts))==1
        sizes=[d['partition']['train_rows'] for d in run.policy.fit_docs().values()]
        leaf,depth=cfg['estimator']['min_samples_leaf'],cfg['estimator']['max_depth']
        nodes=[min(2**(depth+1)-1,max(1,2*(n//leaf)-1)) for n in sizes]
        # Raw arrays plus 1% codec slack, 1MiB/fit metadata, two largest extra
        # copies for replay/atomic write, and 32MiB public reporting headroom.
        bound=math.ceil((sum(nodes)+2*max(nodes))*cfg['estimator']['trees']*layouts[0]*1.01)
        bound+=(len(nodes)+2)*2**20+32*2**20
        free=shutil.disk_usage(run.PRIVATE).free
        if free-bound<=cfg['disk_reserve_bytes']:
            raise OSError('Actual-layout bound still violates10GiB reserve; use CREATE')
        receipt=dict(
            source='fresh_run_checkpoint_structure_audit_not_outcome_selection',
            pilot_report=run.base.artifact(run.PUBLIC/'pilot.json'),
            node_payload_bytes=layouts[0],model_rows_bound=sizes,
            conservative_storage_bytes=bound,free_bytes=free,reserve_bytes=cfg['disk_reserve_bytes'],
            scientific_configuration_unchanged=True,original_pilot_veto_preserved=True)
        clearance=run.PUBLIC/'resource_clearance.json'
        if clearance.exists():
            prior=json.loads(clearance.read_text())
            for k in receipt:
                if k!='free_bytes':assert prior[k]==receipt[k]
        else:
            run.immutable(clearance,receipt)
        del state
        run.beat(state='resource_checked_resume',bound_bytes=bound)
        _,_,data,jobs,oid,_,_,_=run.inner.old.load()
        run.train(cfg,data,jobs,oid,resume=True)


if __name__=='__main__':main()
