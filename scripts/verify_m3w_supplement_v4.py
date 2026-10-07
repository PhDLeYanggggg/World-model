"""Executable mathematical examples, not a new real-data experiment."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before importing numerical libraries')

import numpy as np
import torch

from src.world_model import m3w_easy_harm_deviance as deviance
from src.world_model import m3w_inner_separability as core
from src.world_model import m3w_temporal_auxiliary_readout as readout
from src.world_model import m3w_temporal_target_audit as temporal

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09/evidence_manuscript_v4'
SOURCES = (
    'src/world_model/m3w_inner_separability.py',
    'src/world_model/m3w_query_excess.py',
    'src/world_model/m3w_easy_component_diagnostic.py',
    'src/world_model/m3w_temporal_auxiliary.py',
    'src/world_model/m3w_temporal_target_audit.py',
    'src/world_model/m3w_temporal_auxiliary_readout.py',
    'src/world_model/m3w_unknown_outcome_bounds.py',
    'src/world_model/m3w_component_calibration.py',
    'src/world_model/m3w_easy_harm_deviance.py',
    'configs/m3w_european_temporal_auxiliary_v1.json',
    'outputs/publication_readiness_2026_09/evidence_manuscript_v4/supplement.md',
)


def examples():
    y = np.array([[1., 0, 10, 10, 0], [0, 1, 8, 8, 1], [2, 0, 5, 0, 0]])
    p = np.array([[2., 0, 10, 10, 0], [0, 3, 8, 8, 3], [1, 0, 5, 0, 0]])
    segments = np.array([0, 0, 1])
    rms = np.arange(1., 9)

    def signed(row):
        b, h, r, er, eh = row
        return [b-h, h-.02*r, eh-.02*er]

    row_terms = []
    for truth, pred in zip(y, p):
        a, b = list(truth)+signed(truth), list(pred)+signed(pred)
        square = [((j-i)/s)**2 for i, j, s in zip(a, b, rms)]
        row_terms.append(.5*(sum(square[:5])/5+sum(square[5:])/3))
    scalar = .5*(sum(row_terms[:2])/2+row_terms[2])
    actual = core.losses(torch.tensor(p), torch.tensor(y), torch.tensor(segments), 2, torch.tensor(rms))
    assert math.isclose(float(actual['total']), scalar, abs_tol=1e-14)

    unknown = np.full((3, 5), np.nan)
    env = np.array([2., 3., 4.])
    left, right = np.array([True, True, False]), np.array([True, False, True])
    pair = readout.paired_completion(unknown, left, right, env)
    extremes = [sum((int(b)-int(a))*s*e for a, b, s, e in zip(left, right, signs, env))
                for signs in itertools.product((-1, 1), repeat=3)]
    assert (pair['lower_mass'], pair['upper_mass']) == (min(extremes), max(extremes)) == (-7., 7.)
    proxy = (readout.completion_bounds(unknown, right, env)['selected_net_gain_lower_mass']
             -readout.completion_bounds(unknown, left, env)['selected_net_gain_lower_mass'])
    assert proxy == -1 and pair['shared_unknown_selected'] == 1

    # This fixed ranking illustrates a ratio property, not a learned policy.
    cost = np.array([[0., 1, 10, 10, 1], [0, 0, 1000, 1000, 0], [0, 50, 10, 10, 50]])
    scores = np.array([3., 2., 1.])
    risks = [readout.completion_bounds(cost, scores > cut, np.array([1., 0., 50.]))[
        'easy_selected_risk_upper'] for cut in (2.5, 1.5, .5)]
    assert risks[0] > risks[1] < risks[2]
    assert risks[0] > .02 and risks[1] < .02 and risks[2] > .02

    reference, candidate, target = np.zeros((1, 12, 2)), np.zeros((1, 12, 2)), np.zeros((1, 12, 2))
    reference[..., 0] = 1
    candidate[:, ::2, 0] = 3
    _, trajectory = temporal.targets(reference, candidate, target, np.ones((1, 12), bool))
    assert trajectory['harm'][0] == .5 and trajectory['gross_harm'][0] == 1.
    assert trajectory['harm'][0] == trajectory['gross_harm'][0]-trajectory['cancellation'][0]

    log_u = torch.tensor([-20.], dtype=torch.float64, requires_grad=True)
    observed = torch.ones(1, dtype=torch.float64)
    loss = deviance.easy_deviance(log_u, observed, 2*observed, observed[0])
    gradient = torch.autograd.grad(loss.sum(), log_u)[0].item()
    expected = 2*(math.exp(-20)-1)
    assert math.isclose(gradient, expected, rel_tol=1e-14)
    squared_gradient = 2*math.exp(-20)*(math.exp(-20)-1)

    return dict(
        query_balanced_loss=dict(scalar=scalar, production=float(actual['total']),
                                 row_weights=[.25, .25, .5], not_uniform_rows=True),
        paired_unknown=dict(lower=-7., upper=7., lower_proxy_difference=proxy, shared_cancelled=1),
        threshold_counterexample=dict(descending_thresholds=[2.5, 1.5, .5], easy_risks=risks),
        temporal_identity=dict(gross_harm=1., cancellation=.5, trajectory_harm=.5),
        log_gradient=dict(log_prediction=-20., target=1., deviance=gradient, quadratic=squared_gradient),
        scientific_result=False, synthetic_mathematical_examples=True,
        real_data_read=False, optimizer_updates=0, independent_roles_read=False,
        population_safety_guarantee=False)


def artifact():
    cfg = json.loads((ROOT/SOURCES[-2]).read_text())
    assert cfg['arms'] == ['none', 'rowmean', 'temporal']
    assert cfg['head_training'] == dict(width=32, steps=2000, learning_rate=.001,
        query_batch_size=16, gradient_clip=5., checkpoint_every=100, heartbeat_every=100, auxiliary_weight=.1)
    assert cfg['risk_budget'] == .02 and cfg['bootstrap_resamples'] == 3000 and cfg['bootstrap_seed'] == 20261005
    evidence = examples()
    evidence['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    evidence['result_source'] = 'fresh_run_synthetic_identity_checks_only'
    return json.dumps(evidence, indent=2, allow_nan=False)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = HOME/'supplement_examples.json'
    content = artifact()
    if args.check:
        if path.read_text() != content:
            raise ValueError('Supplement examples or source hashes changed')
    else:
        path.write_text(content)
    print(json.dumps(dict(status='checked' if args.check else 'exported', real_data_read=False,
                          optimizer_updates=0, population_safety_guarantee=False)))


if __name__ == '__main__':
    main()
