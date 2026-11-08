"""Decompose the registered 5% result without selecting a new policy."""
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_tumdot_ing_frozen_transfer as run
import numpy as np


def decompose(ref, forecast, cv, y, valid):
    from src.evaluation.m3w_opendd_location import paired

    cv_error = np.linalg.norm(cv-y, axis=-1)
    observed, lower = paired(ref, forecast, y, valid)
    mass = cv_error[valid].sum()
    complete = valid.all(1)
    return dict(observed_pp=float(100*observed.sum()/mass),
                lower_pp=float(100*lower.sum()/mass),
                debit_pp=float(100*(observed-lower).sum()/mass),
                complete_only_pp=float(100*observed[complete].sum()/cv_error[complete].sum()))


def main():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_ols_replacement as motion
    from src.world_model import m3w_opendd_velocity_candidate as model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg = run.register()
    path = run.PUBLIC / 'analysis.json'
    evidence = json.loads(path.read_text())
    replay = json.loads((run.PUBLIC / 'replay_execution.json').read_text())
    digest = run.prior.sha(path)
    if evidence['registration'] != reg or not replay['fresh_exact_replay'] or replay['analysis_sha256'] != digest:
        raise ValueError('Require the exact frozen result and replay')
    nets = run.load_networks(torch, model)
    cuts = evidence['open_dd_positive_complete_cv_cuts']
    out = {}
    for key, (name, _) in run.ROSTER.items():
        print(key, flush=True)
        data, _, _ = run.read_tracks(run.SOURCE / name)
        h, y, _ = run.windows(data)
        del data
        valid = np.isfinite(y).all(2)
        cv, ols, old, eligible = run.motion_bank(h, motion)
        x, rot, scale = model.design(h, np.full((len(h), 2, 2, 8, 2), np.nan))
        samples = torch.from_numpy(x)
        out[key] = []
        for seed, weights in nets.items():
            with torch.inference_mode():
                local = np.mean([model.decode(net(samples)).numpy().astype(float) for net in weights], axis=0)
            neural = (local*scale[:, None, None]) @ rot.transpose(0, 2, 1)
            pred, matched = run.blend(cv, neural, old, eligible, .05)
            read = next(v['arms']['blend05'] for v in evidence['recorded_files'][key]['views'] if v['seed'] == seed)
            values = dict(neural_vs_cv=decompose(cv, pred, cv, y, valid),
                          neural_vs_matched=decompose(matched, pred, cv, y, valid),
                          matched_vs_cv=decompose(cv, matched, cv, y, valid))
            if (values['neural_vs_cv']['lower_pp'] != read['lower_pp']['cv'] or
                values['neural_vs_matched']['lower_pp'] != read['lower_pp']['matched_hindcast']):
                raise ValueError('Decomposition does not reproduce the frozen result')
            r = read['risk']
            matched_risk = run.score(matched, cv, ols, y, valid, key, cuts)['risk']
            out[key].append(dict(seed=seed, pairs=values, matched_risk=matched_risk,
                neural_risk_failures=[metric for metric in ('completion_all_risk_upper',
                    'completion_easy_risk_upper', 'known_easy_net_degradation')
                    if r[metric] is None or r[metric] > .02]))
    views = [v for seq in out.values() for v in seq]
    means = {p: {k: float(np.mean([v['pairs'][p][k] for v in views])) for k in metrics}
             for p, metrics in views[0]['pairs'].items()}
    dest = run.PUBLIC / 'decomposition.json'
    run.prior.write_json(dest, dict(posthoc_diagnostic=True, primary_sha256=digest,
        unchanged_primary_policy=True, mean=means, views=out,
        matched_risk_passes=sum(v['matched_risk']['supported'] for v in views)), immutable=True)


if __name__ == '__main__':
    main()
