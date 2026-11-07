"""Locate the support-joint failure using only calibration donors."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_support_joint as study

run, np = study.run, study.np


def main():
    a = study.PUBLIC/'analysis.json'
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact preceding readout')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = study.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
    parent, controls, ranks = study.cost.banks()
    old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
           for c in run.read(study.prune.PUBLIC/'freeze.json')['cells']}
    saved = {c['seed']: c['folds'] for c in run.read(study.PUBLIC/'freeze.json')['cells']}
    summary = {k: Counter() for k in study.model.ARMS}
    cuts = run.cuts_by_source()
    for cell in parent:
        m = study.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        for f in saved[m.seed]:
            ident = f['identity']
            site = ident['held']
            co, rank = m.co[site], m.rank(site)
            with np.load(run.checked(old[m.seed][site]['cache']), allow_pickle=False) as z:
                moment = z['coefficients']
            donors = []
            for s in ident['cal']:
                for ix, g in m.groups[s]:
                    args = study.prepare(m, ix, g, co, rank, moment)
                    dc = args[0]
                    _, z, full = study.anchor.model.labels(m.floor[ix], m.bank[ix], y[ix], valid[ix],
                                                           cuts[s]['easy'])
                    j = np.flatnonzero(dc >= 0)
                    orig = z[j, dc[j]].sum(0)
                    donors.append((args, z, full, 1./d['crowd'][ix], valid[ix].any(1),
                                   study.anchor.model.risk(orig, full), orig))
            for arm, expected in zip(study.model.ARMS, f['thresholds']):
                if study.calibrate(donors, arm) != expected:
                    raise ValueError('Require exact donor threshold replay')
                any_insertion, any_gain, risk_ok, gain_ok = False, False, False, False
                record_flags = np.zeros((len(donors), 4), bool)
                for cut in study.model.CUTS:
                    new = lost = extra = 0
                    utility = 0.
                    for t, (args, z, full, w, known, before, orig) in enumerate(donors):
                        dc = args[0]
                        choice = study.model.select(*args, arm, cut)
                        i = np.flatnonzero((choice >= 0) & (dc < 0))
                        j = np.flatnonzero((choice < 0) & (dc >= 0))
                        added, removed = z[i, choice[i]], z[j, dc[j]]
                        after = study.anchor.model.risk(orig + added.sum(0) - removed.sum(0), full)
                        g = after['supported'] and not before['supported']
                        new += int(g)
                        lost += int(before['supported'] and not after['supported'])
                        extra += int(after['easy_net_over'] and not before['easy_net_over'])
                        utility += float(w[i] @ added[:, -1] +
                                         w[j] @ np.where(known[j], -removed[:, -1], -removed[:, 2]))
                        if orig[1] == 0.:
                            record_flags[t] |= (True, bool(len(i)), bool((added[:, 1] > 0.).any()), g)
                        any_insertion |= bool(len(i))
                    any_gain |= new > 0
                    risk_ok |= new > 0 and not lost and not extra
                    gain_ok |= new > 0 and not lost and not extra and utility >= 0.
                summary[arm].update(folds=1, folds_proposing_insertion=int(any_insertion),
                    folds_with_a_new_pass=int(any_gain), folds_also_preserving_old_passes_and_easy=int(risk_ok),
                    folds_also_preserving_gain=int(gain_ok),
                    **dict(zip(('missing_support_occurrences', 'with_insertion', 'with_observed_easy',
                                'with_full_pass'), record_flags.sum(0).tolist())))
        del m
    run.once(study.PUBLIC/'calibration_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(a), repeated_donor_occurrences=True, held_outcomes_read=False,
        arms={k: dict(v) for k, v in summary.items()}, frozen_decisions_changed=False)))


if __name__ == '__main__':
    main()
