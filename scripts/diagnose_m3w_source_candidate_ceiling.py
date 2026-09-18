"""Post-hoc future-informed candidate ceiling; not a deployable selector."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_transfer_control import load_config, build_data
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    public = ROOT/reg['reports']
    evaluation = json.loads((public/'evaluation.json').read_text())
    data, train, _, held, _, _, _ = build_data(reg)
    loc = held-data.nmain
    target = data.target[loc].astype(float)
    cv = np.linalg.norm(target, axis=-1).mean(1)
    zero = cv == 0
    hard = cv >= evaluation['training_hard_cut']
    costs, names = [cv], ['stationary_cv']
    zero_fractions, states = [], []
    for result in evaluation['results']:
        path = ROOT/result['prediction_path']
        assert file_digest(path) == result['prediction_sha256']
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'], held)
            error = np.linalg.norm(a['prediction'].astype(float)-target, axis=-1).mean(1)
        costs.append(error)
        names.append(result['trial'])
        positive_harm = np.maximum(error-cv, 0)
        zero_fraction = float(positive_harm[zero].sum()/positive_harm.sum()) if positive_harm.sum() else None
        zero_fractions.append(zero_fraction)
        states.append(dict(trial=result['trial'], zero_fraction_of_positive_harm=zero_fraction,
            positive_harm=float(positive_harm.mean()), mean_benefit=float(np.maximum(cv-error, 0).mean()),
            binary_oracle_gain_percent=float(100*(1-np.minimum(error, cv).sum()/cv.sum()))))
    costs = np.stack(costs)
    choice = costs.argmin(0)
    oracle = costs.min(0)
    report = dict(result_source='fresh_run_post_hoc_diagnostic_not_preregistered_primary',
        evaluation_sha256=file_digest(public/'evaluation.json'), rows=len(held), candidate_states=18,
        oracle_uses_future_labels=True, oracle_is_deployable=False, new_threshold_or_model_selection=False,
        fixed_path_oracle_gain_percent=float(100*(1-oracle.sum()/cv.sum())),
        fixed_path_oracle_hard_gain_percent=float(100*(1-oracle[hard].sum()/cv[hard].sum())),
        optimal_label_choice_distribution={name:int((choice==k).sum()) for k,name in enumerate(names)},
        states=states, mean_seed_averaging_not_used_for_this_oracle=True,
        proof_scope='Only choosing one complete saved path or CV per query; not blending, rescaling, new trajectories or other datasets',
        independent_confirmation=False, deployment_changed=False)
    json_write(public/'candidate_ceiling.json', report)
    text = ['# Post-Hoc Candidate-Family Ceiling', '',
        'This supplementary diagnosis was added after fixed scores were observed. It is not the preregistered primary comparison or a selected deployment policy.', '',
        f"Choosing the best of all18 saved complete predictor paths andCV using future labels yields {report['fixed_path_oracle_gain_percent']:.6f}% ADE gain on the6944 bookstore rows; training-defined hard gain is {report['fixed_path_oracle_hard_gain_percent']:.6f}%.",
        'Every causal selector restricted to this same complete-path set is bounded above on these labeled rows by this per-query minimum. This is an empirical action-class ceiling, not a bound for blending, rescaling, different models, new scenes or the full main benchmark.', '',
        f"Across individual candidates, exact-zero-target rows account for {min(zero_fractions):.2%} to {max(zero_fractions):.2%} of positive error increase. The model can also damage moving cases; zero-target harm is not asserted to be the only cause.",
        'No oracle choices are exported as inference features, no threshold is tuned, and no model is promoted. The outputs use future labels for diagnosis only. No Stage5C or SMC.', '']
    (public/'candidate_ceiling.md').write_text('\n'.join(text))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
