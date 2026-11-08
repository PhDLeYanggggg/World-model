"""Produce a small readout and a post-hoc known-easy oracle diagnostic."""
from hashlib import file_digest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_cv_abstention as run


def known_easy_oracle(d, cut):
    import numpy as np
    from src.evaluation.m3w_opendd_cv_abstention import edit, metrics

    cv = np.linalg.norm(d['cv'] - d['y'], axis=-1)
    bank = np.linalg.norm(d['bank'] - d['y'][:, None], axis=-1)
    valid = d['valid'][:, None]
    lower = np.where(valid, cv[:, None] - bank, -d['dist']).sum(2)
    row = cv.mean(1)
    easy = d['valid'].all(1) & (row > 0) & (row <= cut)
    # Hindsight only: no forecast may increase any known easy target's ADE.
    legal = d['legal'] & ~(easy[:, None] & (lower < 0.))
    pred, switches = edit({**d, 'legal': legal}, lower, threshold=0.)
    got = metrics(d, pred, switches, cut)
    if got['positive_easy_pct'] is None or got['positive_easy_pct'] > 1e-10:
        raise ValueError('Oracle increased a known easy error or lacks easy support')
    return got


def main():
    import numpy as np
    import torch
    from src.world_model import m3w_ols_replacement as motion

    # The registered data preparer validates cohort SHA and bindings before loading.
    run.np, run.torch, run.motion = np, torch, motion
    source = run.PUBLIC / 'analysis.json'
    result = json.loads(source.read_text())
    if run.prior.sha(run.PUBLIC/'registration.json') != result['registration'] or len(result['fits']) != 168:
        raise ValueError('Require completed registered analysis')
    with source.open('rb') as f:
        digest = file_digest(f, 'sha256').hexdigest()
    reg = run.prior.sha(run.prior.PUBLIC/'registration.json')
    sites = list(result['sites'])
    cohort = {s: run.prepare(s, reg) for s in sites}
    out, oracle = {}, {}
    for site in sites:
        cut = run.easy_cut(cohort, [s for s in sites if s != site])
        oracle[site] = known_easy_oracle(cohort[site], cut)
        out[site] = [dict(seed=r['seed'], arms=r['arms'],
                          thresholds={a: c['selected'] for a,c in r['calibration'].items()})
                     for r in result['sites'][site]]
    gains = np.asarray([r['lower_pp'] for r in oracle.values()])
    draws = np.random.default_rng(817).integers(len(gains), size=(3000,len(gains)))
    compact = dict(analysis_sha256=digest, registration=result['registration'], fits=len(result['fits']),
        thresholds=result['thresholds'], summary=result['summary'], sites=out, runtime=result['runtime'],
        data_source=result['data_source'], scope=result['scope'], independent_confirmation=False,
        known_easy_oracle=dict(sites=oracle, lower_pp=float(gains.mean()),
            ci95_pp=np.quantile(gains[draws].mean(1), [.025,.975]).tolist(),
            switches_per_seed=sum(r['switches'] for r in oracle.values()),
            hindsight_only=True, official_risk_pass=False))
    run.prior.write_json(run.PUBLIC/'metrics.json', compact, immutable=True)
    print(json.dumps(compact['known_easy_oracle'], sort_keys=True))


if __name__ == '__main__':
    main()
