"""Account separately for stored, gradient-enabled and optimizer-active weights."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_eqmotion as run


def main():
    cfg=json.loads(run.CONFIG.read_text())
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text());rows=[]
    for ref in freeze['fits']:
        path=ROOT/ref['path']
        if run.sha(path)!=ref['sha256']:raise ValueError('Frozen training receipt differs')
        row=json.loads(path.read_text());cp=ROOT/row['checkpoint']
        if run.sha(cp)!=row['checkpoint_sha256']:raise ValueError('Checkpoint bytes differ')
        model=run.SourceForecaster(cfg['architecture'],row['parent_trial']['baseline_index'])
        state=run.torch.load(cp,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);params=list(model.parameters())
        ids=[i for group in state['optimizer']['param_groups'] for i in group['params']]
        if ids!=list(range(len(params))) or state['step']!=4000:
            raise ValueError('The original full AdamW parameter order and budget are required')
        active=set(state['optimizer']['state'])
        rows.append(dict(trial=row['trial'],checkpoint_sha256=row['checkpoint_sha256'],
            stored_parameters=sum(p.numel() for p in params),
            requires_grad_parameters=sum(p.numel() for p in params if p.requires_grad),
            optimizer_state_parameters=sum(p.numel() for i,p in enumerate(params) if i in active),
            optimizer_step_range=[min(int(s['step']) for s in state['optimizer']['state'].values()),
                max(int(s['step']) for s in state['optimizer']['state'].values())]))
    if len(rows)!=9:raise ValueError('Complete nine-model accounting required')
    result=dict(result_source='cached_verified_checkpoint_accounting_not_new_inference_or_training',
        auditor_sha256=run.sha(Path(__file__)),training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        rows=rows,input_or_label_arrays_read=False,requires_grad_not_synonymous_with_received_gradient=True,
        unmatched_architecture_and_compute_disclosed=True,unique_failure_cause_inferred=False,
        deployment_changed=False,stage5c=False,smc=False)
    (run.PUBLIC/'parameter_accounting.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
