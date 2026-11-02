"""Export real scalar fitting monitors, not data or trained parameter arrays."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_conditional_moment as run


def main():
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text());cells=[]
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Frozen fit receipt changed')
        r=json.loads((ROOT/ref['path']).read_text());path=ROOT/r['checkpoint']['path']
        if run.sha(path)!=r['checkpoint']['sha256']:raise ValueError('Completed weights changed')
        state=run.api.base.read_checkpoint(path)
        if state['trace'][0]!=r['first_trace'] or state['trace'][-1]!=r['final_trace']:raise ValueError('Trace endpoints changed')
        trace=[dict(step=v['step'],anchor_monitor=v['monitor']['anchor_objective'],
            occurrence_monitor=v['monitor']['occurrence_BCE'],conditional_monitor=v['monitor']['conditional_magnitude_MSE'],
            total_monitor=v['monitor']['total'],gradient_norm=v.get('gradient_norm'),minimum_query_visits=v.get('minimum_query_visits')) for v in state['trace']]
        cells.append(dict(context=r['context'],seed=r['seed'],arm=r['arm'],budget=r['budget'],checkpoint_sha256=r['checkpoint']['sha256'],trace=trace))
    if len(cells)!=36:raise ValueError('All36complete curves required')
    run.once(run.PUBLIC/'training_traces.json',dict(result_source='fresh_scalar_export_of_cached_verified_real_checkpoints',
        training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),cells=cells,
        monitor_role='fixed_fitting_queries_not_validation',optimization_reexecuted=False,raw_data_features_weights_included=False))


if __name__=='__main__':main()
