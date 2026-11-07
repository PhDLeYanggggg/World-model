"""Rehearse the grouped external scorer on already-exposed development only."""
import argparse
from pathlib import Path
import platform
import sys
import time

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_box_visibility as parent
from src.evaluation.m3w_grouped_recording_readout import grouped_readout

h, run, np = parent.h, parent.run, parent.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/dronecrowd_scoring_rehearsal_v1'
EXTERNAL = PUBLIC.parent/'dronecrowd_blind_preflight_v1'


def evaluate():
    study = parent.PUBLIC/'analysis.json'
    if run.read(parent.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(study):
        raise ValueError('Require the unchanged exposed direct policy')
    external = run.read(EXTERNAL/'analysis.json')
    if external['scientific_roles_assigned'] or external['future_label_method_called']:
        raise ValueError('Rehearsal cannot imply external admission')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = h.previous.prior.blocks(d, lab['frames'])
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    if len(roster) != 34 or len(set(roster.values())) != 14:
        raise ValueError('Use only the original exposed development recordings')
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cells, controls, ranks = h.cost.banks()
    cuts = run.cuts_by_source()
    cut_source = run.old.PARENTS[4]
    candidate_cuts = run.read(cut_source)['cuts']
    if set(candidate_cuts) != {'0', '1'} or any(v not in list(candidate_cuts.values()) for v in cuts.values()):
        raise ValueError('The original two development reference pairs are required')
    p = run.parent
    if run.read(p.REG)['fitting_selected_accuracy_comparator'] != p.OLS:
        raise ValueError('Keep the original fitting-selected comparator')
    all_rows, alt = [], {k: [] for k in sorted(candidate_cuts)}
    for cell in cells:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        for rec, (ix, _) in records.items():
            s = roster[rec]
            cv, bank = m.floor[ix], m.bank[ix]
            direct = run.model.selected_paths(m.direct[ix], bank, cv)
            paths = {p.CV: cv, 'direct': direct}
            paths.update({name: bank[:, j] for j, name in enumerate(run.NAMES)})
            pairs = {f'{k}_vs_cv': (p.CV, k, p.CV) for k in paths if k != p.CV}
            pairs['direct_vs_fitting_selected_ols4'] = (p.OLS, 'direct', p.CV)
            args = y[ix], valid[ix], d['queries'][ix], rec
            cut = cuts[s]
            native = p.readout.score_recording(paths, pairs, *args, easy_cut=cut['easy'], hard_cut=cut['hard'])
            identity = dict(seed=m.seed, source=s)
            all_rows.append(dict(native, **identity))
            for key, alternate in candidate_cuts.items():
                check = p.readout.score_recording({p.CV: cv, 'direct': direct},
                    {'direct_vs_cv': (p.CV, 'direct', p.CV)}, *args,
                    easy_cut=alternate['easy'], hard_cut=alternate['hard'])
                if check['comparisons']['direct_vs_cv'] != native['comparisons']['direct_vs_cv']:
                    raise ValueError('A different reference split changed all-roster utility')
                alt[key].append(dict(check, **identity))
        print(dict(seed=m.seed, recordings=len(records)), flush=True)
        del m
    seeds = list(run.kin.prior.SEEDS)
    primary = grouped_readout(all_rows, roster, seeds)
    original = run.read(study)['risk']['direct']
    check = primary['risk']['direct']
    for a, b in (('passes', 'supported_views'), ('switches', 'switches'),
                 ('undefined_easy', 'undefined_easy_views'), ('easy_net_over_2', 'signed_easy_over_2_views')):
        if original[a] != check[b]:
            raise ValueError('Development score does not replay the original direct policy')
    by_cut = {}
    for key, rows in alt.items():
        result = grouped_readout(rows, roster, seeds)
        if result['effects']['direct_vs_cv'] != primary['effects']['direct_vs_cv']:
            raise ValueError('Reference sensitivity should leave all-roster comparisons unchanged')
        by_cut[key] = dict(reference_cuts=candidate_cuts[key], risk=result['risk']['direct'])
    bindings = (cut_source, p.REG, study, parent.PUBLIC/'replay_execution.json',
                EXTERNAL/'analysis.json', EXTERNAL/'kinematic_coverage.json')
    return dict(scope='already_exposed_development_scoring_rehearsal',
        original_source_specific=primary, counterfactual_fixed_reference_cuts=by_cut,
        source_sha256={str(path.relative_to(ROOT)):run.sha(path) for path in bindings},
        original_risk_reference=p.CV, fitting_selected_accuracy_comparator=p.OLS,
        original_budget=.02, external_reference_cut_frozen=False, external_controller_frozen=False,
        external_scientific_roles_assigned=False, external_labels_opened=False,
        independent_confirmation=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'replay'))
    phase = parser.parse_args().phase
    output = PUBLIC/'analysis.json'
    if phase == 'replay' and not output.is_file():
        raise ValueError('Score before replay')
    start = time.monotonic()
    run.once(output, run.stable(evaluate()))
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
