"""Post-hoc registered direction null; no selection, extra fitting or held-role access."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_motion_candidate import load_config, context, immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_direction_null import rotated_costs
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    public = ROOT/reg['reports']; _, control, data = context(reg)
    protocol = json.loads((public/'direction_null_registration.json').read_text())
    assert protocol['rotations'] == ['original','plus90','minus90','reverse']
    assert protocol['selection'] is False and protocol['posthoc'] is True
    for name,sha in protocol['bindings'].items(): assert file_digest(ROOT/name) == sha
    report = json.loads((public/'report.json').read_text())
    verification = json.loads((public/'verification.json').read_text())
    assert verification['identity'] == report['identity'] and verification['exact_replayed_models'] == 12
    ids = np.flatnonzero(data.source_sites != 'bookstore')+data.nmain
    loc = ids-data.nmain; sites = data.source_sites[loc]; target = data.target[loc]
    cv = np.linalg.norm(target.astype(float),axis=-1).mean(1)
    site_cv = np.array([cv[sites == s].mean() for s in reg['sites']])
    resamples = np.random.default_rng(reg['bootstrap_seed']).integers(4,size=(reg['bootstrap_resamples'],4))
    results, contrasts = [], []
    for arm, receipts in [('unconditional_control',control),('motion_loss',report)]:
        collected = {name:[] for name in protocol['rotations']}
        for receipt in receipts['oof_labels']:
            path = ROOT/receipt['path']; assert file_digest(path) == receipt['sha256']
            with np.load(path,allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'],ids)
                costs = rotated_costs(a['prediction'],target)
            for name,value in costs.items(): collected[name].append(value)
        site_oracle, site_error = {}, {}
        for name,values in collected.items():
            values = np.asarray(values); mean = values.mean(0); oracle = np.minimum(values,cv[None,:]).mean(0)
            site_oracle[name] = np.array([oracle[sites == s].mean() for s in reg['sites']])
            site_error[name] = np.array([mean[sites == s].mean() for s in reg['sites']])
            results.append(dict(arm=arm,rotation=name,
                equal_site_all_gain=float(100*(1-site_error[name].mean()/site_cv.mean())),
                equal_site_oracle_gain=float(100*(1-site_oracle[name].mean()/site_cv.mean())),
                nonzero_target_pooled_gain=float(100*(1-mean[cv>0].sum()/cv[cv>0].sum())),
                easy_pixel_harm=float((mean[cv==0]*data.native_scale[loc][cv==0]).mean())))
        for name in protocol['rotations'][1:]:
            difference = site_oracle[name]-site_oracle['original']
            samples = 100*difference[resamples].mean(1)/site_cv[resamples].mean(1)
            contrasts.append(dict(arm=arm,contrast='original_minus_'+name+'_oracle_gain',
                point_pp=float(100*difference.mean()/site_cv.mean()),
                conditional_site_ci95=np.quantile(samples,[.025,.975]).tolist()))
    evidence = dict(result_source='fresh_run_posthoc_fixed_direction_null_on_training_OOF_predictions',
        protocol_sha256=file_digest(public/'direction_null_registration.json'), results=results, contrasts=contrasts,
        interpretation='direction_null_not_causal_routing_or_new_independent_confirmation',
        future_oracle_is_deployable=False, label_selected_rotation=False, new_training_updates=0,
        outer_rows_scored=0, main_rows_scored=0, new_deployment=False)
    immutable_json(public/'direction_null.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__': main()
