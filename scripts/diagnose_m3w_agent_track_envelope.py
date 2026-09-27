"""Describe the existing output bound after prediction freezing; no policy sweep."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_agent_track_refit as run
import numpy as np
import torch
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_european_source_forecast import baseline_torch
from src.world_model.m3w_baseline_relative_forecaster import past_motion_budget
from src.evaluation.m3w_motion_envelope_diagnostic import envelope_statistics


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,reg,data,jobs,identity=run.load()
    run.committed(run.PUBLIC/'prediction_freeze.json')
    frozen=json.loads((run.PUBLIC/'prediction_freeze.json').read_text()); rows=[]
    for j,ref in zip(jobs,frozen['predictions']):
        ids=j['design']['held_ids']; pred={}
        for arm in ('grouped','flat'):
            assert run.artifact(ROOT/ref[arm]['path'])==ref[arm]
            with np.load(ROOT/ref[arm]['path'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids); pred[arm]=z['prediction'].copy()
        baseline=[]; budget=[]
        for start in range(0,len(ids),1024):
            x=pack_geometry(data['geometry'][ids[start:start+1024]])
            x['baseline']=baseline_torch(x['history'],j['design']['baseline_index'])
            baseline.append(x['baseline'].numpy()); budget.append(past_motion_budget(x).numpy())
        baseline=np.concatenate(baseline); budget=np.concatenate(budget)
        masks=run.old.metrics.masks(data['baseline_ade'][ids,1],j['design']['easy_cut'],j['design']['hard_cut'])
        for site in sorted(set(data['sites'][ids])):
            for name,mask in masks.items():
                use=(data['sites'][ids]==site)&mask
                for arm in ('grouped','flat'):
                    stat=envelope_statistics(baseline[use],pred[arm][use],data['target'][ids][use],
                        data['valid'][ids][use],budget[use])
                    rows.append(dict(trial=j['key'],site=str(site),subset=name,arm=arm,metric=stat))
    doc=dict(result_source='fresh_source_development_diagnostic',diagnostic_not_primary=True,rows=rows,
        prediction_freeze=run.artifact(run.PUBLIC/'prediction_freeze.json'),
        target_labels_used_only_for_diagnostic=True,oracle_lower_bound_not_model=True,
        inference_changed=False,independent_roles_read=False,deployment_changed=False)
    run.immutable_json(run.PUBLIC/'envelope_diagnostic.json',doc)
    lines=['# Existing Motion Envelope Diagnostic','',
        'This uses labels ONLY for an oracle geometric lower bound after prediction freezing.',
        'At each labeled step, the best possible error inside the existing correction disk',
        'is max(0, baseline error minus causal radius). It is not a realizable learned model.',
        'The radius and inference behavior are not changed. Targets outside the disk establish',
        'a representational limit, not that relaxing it would improve prediction or safety.',
        'A usage ratio above0.95 is a fixed descriptive proxy, never a deployment threshold.',
        'Costs are image-local; rows cannot be pooled into a metric distance.','',
        '| Trial | Locality | Subset/arm | Supported rows | ADE | Oracle lower ADE | Lower/error ratio | Outside-step fraction | Mean usage | >95% usage | Zero-budget misses |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def f(v): return 'undefined' if v is None else f'{v:.6f}'
    for r in rows:
        m=r['metric']; values=[m.get(k) for k in ('prediction_ADE','oracle_envelope_lower_ADE',
            'lower_fraction_of_prediction_error','labels_outside_envelope_fraction','envelope_usage_mean',
            'fraction_above_95pct_budget','zero_budget_positive_error_steps')]
        lines.append(f"| {r['trial']} | {r['site']} | {r['subset']}/{r['arm']} | {m['rows']} | "+' | '.join(map(f,values))+' |')
    (run.PUBLIC/'envelope_diagnostic.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(views=len(rows),oracle_not_model=True)))


if __name__=='__main__': main()
