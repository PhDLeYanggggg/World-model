"""Post-hoc training-side attribution; no refit, selection or new held readout."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_crossfit import load_config, load_data, CrossfitCorpus
from scripts.run_m3w_source_continuation import immutable_json
from src.evaluation.m3w_crossfit_cost_diagnostic import decompose_costs
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    public = ROOT / reg['reports']
    report = json.loads((public / 'report.json').read_text())
    verification = json.loads((public / 'verification.json').read_text())
    assert report['identity'] == verification['identity']
    assert verification['exact_train_and_held_replayed_models'] == 12
    data = CrossfitCorpus(load_data(Path(reg['data_registration'])))
    ids = np.flatnonzero(data.source_sites != 'bookstore') + data.nmain
    loc = ids - data.nmain
    cv = np.linalg.norm(data.target[loc].astype(float), axis=-1).mean(1)
    errors = []
    for receipt in report['oof_labels']:
        path = ROOT / receipt['path']
        assert file_digest(path) == receipt['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['ids'], ids)
            np.testing.assert_array_equal(arrays['cv_ade'], cv)
            errors.append(arrays['ade'].copy())
    result = decompose_costs(cv, np.asarray(errors), data.source_sites[loc], data.native_scale[loc])
    result.update(result_source='fresh_run_posthoc_training_only_cost_attribution',
        script_sha256=file_digest(Path(__file__)),
        function_sha256=file_digest(ROOT / 'src/evaluation/m3w_crossfit_cost_diagnostic.py'),
        report_sha256=file_digest(public / 'report.json'),
        verification_sha256=file_digest(public / 'verification.json'),
        registration_changed=False, new_training_updates=0, policy_selected=False,
        outer_rows_scored=0, main_rows_scored=0, independent_confirmation=False,
        scope='fixed_per_seed_CV_or_candidate_action_class_not_all_models_or_all_policies')
    immutable_json(public / 'cost_attribution.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
