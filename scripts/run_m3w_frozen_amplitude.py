"""Evaluate preregistered, count-preserving amplitude controls on exposed data."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_external_score_rehearsal as parent
from src.evaluation import m3w_frozen_amplitude as model
from src.evaluation.m3w_grouped_recording_readout import grouped_readout

h, run, np = parent.h, parent.run, parent.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_frozen_amplitude_v1'
REG = PUBLIC/'registration.json'
ALPHAS = {'eighth': .125, 'quarter': .25, 'half': .5}
ARMS = ('direct', *ALPHAS, 'past_scale_cap')
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_frozen_amplitude.py')


def register():
    a = parent.PUBLIC/'analysis.json'
    replay = parent.PUBLIC/'replay_execution.json'
    if run.read(replay)['analysis_sha256'] != run.sha(a) or run.read(a)['external_labels_opened']:
        raise ValueError('Require exact development replay with external labels sealed')
    parents = (a, replay, Path(parent.__file__), ROOT/'src/evaluation/m3w_grouped_recording_readout.py')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], alphas=ALPHAS, past_scale_cap=1.,
        primary='quarter_vs_direct', budget=.02, no_count_changes=True,
        independent_outcomes_opened=False)))


def check_bound(observed, upper):
    for key, bound in upper.items():
        v = observed[key]
        if (bound is None) != (v is None) or (
                v is not None and v > bound + 1e-8 * max(1., abs(bound))):
            raise ValueError('Convex risk bound or support preservation failed')


def analyze():
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = h.previous.prior.blocks(d, lab['frames'])
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cells, controls, ranks = h.cost.banks()
    cuts, p = run.cuts_by_source(), run.parent
    rows, cause, guaranteed = [], {a: Counter() for a in ARMS}, Counter()
    for cell in cells:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        for rec, (ix, _) in records.items():
            site = roster[rec]
            c, b = m.floor[ix], m.bank[ix]
            original = run.model.selected_paths(m.direct[ix], b, c)
            scale = dict(ALPHAS, past_scale_cap=model.trust_alpha(c, original, m.scale[ix]))
            paths = {p.CV: c, 'direct': original}
            paths.update({a: model.shrink(c, original, k) for a, k in scale.items()})
            pairs = {f'{a}_vs_cv': (p.CV, a, p.CV) for a in ARMS}
            pairs.update({f'{a}_vs_direct': ('direct', a, p.CV) for a in scale})
            out = p.readout.score_recording(paths, pairs, y[ix], valid[ix], d['queries'][ix], rec,
                easy_cut=cuts[site]['easy'], hard_cut=cuts[site]['hard'])
            base = out['metrics']['direct']['risk']
            for a in ARMS:
                risk = out['metrics'][a]['risk']
                if risk['selected'] != base['selected']:
                    raise ValueError('Intervention count changed')
                cause[a][model.total_risk_cause(risk)] += 1
                if a == 'direct':
                    continue
                bound = model.bounds(c, original, y[ix], valid[ix], cuts[site]['easy'], scale[a])
                check_bound(risk, bound)
                guaranteed[a] += int(all(v is not None and v <= .02 for v in bound.values()))
                if a in ALPHAS:
                    x, z = (out['comparisons'][f'{name}_vs_cv'] for name in ('direct', a))
                    lower = z['known_delta']-z['unknown_radius']
                    if lower + 1e-8 < ALPHAS[a]*(x['known_delta']-x['unknown_radius']):
                        raise ValueError('Uniform shrink violated its paired lower bound')
            rows.append(dict(out, source=site, seed=m.seed))
        print(dict(seed=m.seed, recordings=len(records)), flush=True)
        del m
    seeds = list(run.kin.prior.SEEDS)
    result = grouped_readout(rows, roster, seeds)
    orig = run.read(parent.PUBLIC/'analysis.json')['original_source_specific']
    if (result['risk']['direct'] != orig['risk']['direct']
            or result['effects']['direct_vs_cv'] != orig['effects']['direct_vs_cv']):
        raise ValueError('Require original score before comparing amplitude')
    groups = sorted(set(roster.values()))
    group_means = np.array([np.mean([
        int(r['metrics']['quarter']['risk']['supported'])-int(r['metrics']['direct']['risk']['supported'])
        for r in rows if r['source'] == g]) for g in groups])
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    delta = dict(point=float(group_means.mean()),
        interval95=np.quantile(group_means[draw].mean(1), [.025, .975]).tolist())
    return dict(grouped=result, total_risk_cause={a: dict(x) for a, x in cause.items()},
        bound_sufficient_views=dict(guaranteed), quarter_minus_direct_group_pass_delta=delta,
        alphas=ALPHAS, original_choices_fixed=True, original_rules_unchanged=True,
        all_observed_bound_checks_passed=True, independent_confirmation=False,
        external_labels_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = parser.parse_args().phase
    register()
    if phase == 'register':
        return
    for p in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))],
                                   cwd=ROOT) != p.read_bytes():
            raise ValueError('Commit amplitude protocol before score')
    output = PUBLIC/'analysis.json'
    if phase == 'replay' and not output.is_file():
        raise ValueError('Score before replay')
    start = time.monotonic()
    run.once(output, run.stable(analyze()))
    receipt = dict(phase=phase, seconds=time.monotonic()-start, analysis_sha256=run.sha(output))
    p = PUBLIC/f'{phase}_execution.json'
    if p.is_file():
        if run.read(p)['analysis_sha256'] != receipt['analysis_sha256']:
            raise ValueError('Previous receipt differs from replay')
    else:
        run.once(p, receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
