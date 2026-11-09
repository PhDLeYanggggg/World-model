"""Distinguish predicted abstention from single-action hindsight feasibility."""

import argparse
from collections import Counter
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_vehicle_matched as run

np, torch, parent = run.np, run.torch, run.parent


def admission(v, *, require_easy):
    utility = (v[..., :2] > 0.).all(axis=-1)
    all_ok = v[..., 2] <= .02 * v[..., 4]
    easy_ok = v[..., 3] <= .02 * v[..., 5]
    ok = utility & all_ok & easy_ok
    if require_easy:
        ok &= v[..., 5] > 0.
    return utility, all_ok, easy_ok, ok


def collect(site, d, value, actual, legal, cv, ols, paths, cut):
    gain, all_ok, easy_ok, accepted = admission(value, require_easy=False)
    _, _, _, hindsight = admission(actual, require_easy=True)
    recs, rec = np.unique(d['recording'], return_inverse=True)
    w, y, valid = d['weight'], d['future'], d['valid']
    _, ols_floor = parent.paired(ols, cv, y, valid)
    offset = np.bincount(rec, weights=w * ols_floor.sum(axis=1))
    # Exact conservative gain versus OLS must be positive after applying the edit.
    hindsight &= offset[rec, None] + actual[..., 1] * 12. * w[:, None] > 0.
    choice = cv.copy()
    candidates = legal & hindsight
    for i in np.unique(rec[np.any(candidates, axis=1)]):
        ix = np.flatnonzero(rec == i)
        benefit = np.where(candidates[ix], actual[ix, :, 0]*w[ix, None], -np.inf)
        a, b = np.unravel_index(benefit.argmax(), benefit.shape)
        choice[ix[a]] = paths[ix[a], b]
    from src.evaluation.m3w_selection_forecast_readout import policy
    check = policy(choice, cv, cv, y, valid, d['recording'], easy_cut=cut[0], hard_cut=cut[1])
    n = int(np.unique(rec[candidates.any(axis=1)]).size)
    verified = sum(v['supported'] for v in check['recordings'].values())
    if verified > n:
        raise ValueError('Hindsight verification found an unexpected recording')
    stats = Counter(actions=int(legal.sum()), gain=int((legal & gain).sum()),
        all_risk=int((legal & all_ok).sum()), easy_risk=int((legal & easy_ok).sum()),
        gain_and_all_risk=int((legal & gain & all_ok).sum()),
        gain_and_easy_risk=int((legal & gain & easy_ok).sum()), predicted_single_admissible=int((legal & accepted).sum()),
        hindsight_admissible=int(candidates.sum()), hindsight_recordings=n, hindsight_verified_recordings=verified,
        recordings=len(recs))
    return dict(counts=dict(stats), site=site,
        predicted_mass_means=value[legal].mean(axis=0).tolist(), actual_mass_means=actual[legal].mean(axis=0).tolist(),
        hindsight_cv_gain_pp=parent.metric(choice, cv, ols, d, cut)['cv_gain_known_lower_pp'][1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data_key, pf = run.register()
    fp, rp = run.OUT / 'freeze.json', run.OUT / 'readout.json'
    f, result = run.read(fp), run.read(rp)
    if f['registration'] != key or result['freeze'] != parent.prior.sha(fp) or result['songdo_read']:
        raise ValueError('Require prior frozen and scored development run')
    sites = parent.load(data_key)
    rows = {}
    for site, d in sites.items():
        cut = run.calibration_cut(sites, [s for s in sites if s != site])
        for seed in run.SEEDS:
            tag = f'{site}_{seed}'
            source = parent.LOCAL / f'{tag}_completion.pt'
            value_file = run.LOCAL / f'{tag}_value.pt'
            if (parent.prior.sha(source) != pf['checkpoints'][f'{tag}_completion']['sha256']
                    or parent.prior.sha(value_file) != f['cells'][tag]['value']['sha256']):
                raise ValueError('Frozen model changed')
            net = run.motion.motion.network(17).eval()
            net.load_state_dict(torch.load(source, weights_only=True)['weights'])
            val = run.bridge.BridgeValue().eval()
            val.load_state_dict(torch.load(value_file, weights_only=True)['weights'])
            cv, ols, _, paths, x, unit, legal = run.menu(d, run.forecast(net, d))
            with torch.inference_mode():
                predicted = val(torch.from_numpy(x)).numpy() * unit[:, None, None]
            actual = run.bridge.targets(cv, ols, paths, d['future'], d['valid'], cut[0], unit) * unit[:, None, None]
            rows[tag] = collect(site, d, predicted, actual, legal, cv, ols, paths, cut)
            print(tag, rows[tag]['counts'], flush=True)
    stats = sum((Counter(r['counts']) for r in rows.values()), Counter())
    result = dict(scope='posthoc_fixed_menu_hindsight_only_not_deployable', prior_readout=parent.prior.sha(rp),
        counts=dict(stats), rows=rows, no_threshold_change=True, songdo_read=False, original_2pct_unchanged=True)
    run.write(run.OUT / 'diagnosis.json', result)
    print(dict(stats), flush=True)


if __name__ == '__main__':
    main()
