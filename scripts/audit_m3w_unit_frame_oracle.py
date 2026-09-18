"""Post hoc labeled-fit oracle ceiling; never a deployment policy or selector."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_m3w_objective_alignment import paired_interval
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args(); reg=json.loads(args.registration.read_text())
    parent=json.loads((ROOT/reg['parent_registration']).read_text())
    mainreg=json.loads((ROOT/parent['main_registration']).read_text())
    inputs=ROOT/mainreg['output']/'inputs'
    receipt=json.loads((inputs/'data_manifest.json').read_text())
    for name in ('targets.npy','baselines.npy','folds.npy'):
        if file_digest(inputs/name)!=receipt['arrays'][name]:
            raise ValueError('Main cache changed')
    y=np.load(inputs/'targets.npy',mmap_mode='r'); b=np.load(inputs/'baselines.npy',mmap_mode='r')
    folds=np.load(inputs/'folds.npy',mmap_mode='r'); cv=receipt['baseline_names'].index('constant_velocity_causal_fd')
    reports=ROOT/reg['reports']; rp=reports/'report.json'; report=json.loads(rp.read_text())
    if not report['complete']:
        raise ValueError('Complete fixed matrix required')
    lookup={(t['arm'],t['seed'],t['fold']):t for t in report['trials']}
    names=['constant_velocity_causal_fd','legacy_sdd_aux']+reg['arms']; cells=[]
    for seed in reg['seeds']:
        for fold in range(3):
            ids=np.flatnonzero(folds==fold)
            errors=[np.linalg.norm(b[ids,cv].astype(float)-y[ids],axis=-1).mean(1)]
            for a in names[1:]:
                t=lookup[a,seed,fold]; path=ROOT/t['prediction_path']
                if file_digest(path)!=t['prediction_sha256']:
                    raise ValueError('Prediction artifact changed')
                with np.load(path,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['held_indices'],ids)
                    errors.append(np.linalg.norm(saved['prediction'].astype(float)-y[ids],axis=-1).mean(1))
            costs=np.stack(errors,1); choice=costs.argmin(1); oracle=costs.min(1)
            cells.append(dict(seed=seed,fold=fold,rows=len(ids),reference_ADE=float(costs[:,0].mean()),
                legacy_oracle_ADE=float(costs[:,:2].min(1).mean()),expanded_oracle_ADE=float(oracle.mean()),
                oracle_choices={a:int((choice==i).sum()) for i,a in enumerate(names)},
                strict_beneficial_switch_fraction=float((oracle<costs[:,0]).mean())))
    def matrix(key):
        return np.array([[next(c[key] for c in cells if c['seed']==s and c['fold']==f)
                          for f in range(3)] for s in reg['seeds']])
    result=dict(result_source='fresh_run_posthoc_labeled_fit_oracle_diagnostic',
        report_sha256=file_digest(rp),registration_sha256=file_digest(args.registration),
        cells=cells,expanded_vs_CV=paired_interval(matrix('expanded_oracle_ADE'),matrix('reference_ADE'),2000),
        old_vs_CV=paired_interval(matrix('legacy_oracle_ADE'),matrix('reference_ADE'),2000),
        future_labels_used_only_for_diagnostic_oracle=True,not_deployable=True,
        learned_policy_trained=False,model_selected=False,independent_confirmation=False)
    json_write(reports/'oracle_diagnostic.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
