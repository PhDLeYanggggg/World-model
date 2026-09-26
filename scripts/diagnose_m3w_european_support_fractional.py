"""Post-freeze fit/transport diagnosis, not another selection endpoint."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_support_fractional as run


def fit_transport_summary(rows):
    result = {}
    for pair in sorted({r['pair'] for r in rows}):
        records = []
        for row in rows:
            if row['pair'] != pair:
                continue
            for fold in row['folds']:
                values = {}
                for scope in ('training', 'held'):
                    arm_values = [fold['training'][a]['training'] if scope == 'training'
                                  else fold['metrics'][a]['envelope_positive']
                                  for a in ('mean', 'fractional')]
                    old, new = [v.get('harm_MSE') for v in arm_values]
                    values[scope] = (100 * (old - new) / old
                                     if old is not None and new is not None and old > 0 else None)
                records.append(dict(group=row['group'], held_locality=fold['held'], **values))
        def stats(key):
            v = [r[key] for r in records if r[key] is not None]
            return dict(n=len(v), missing=len(records)-len(v), improved=sum(x > 0 for x in v),
                        median=float(np.median(v)) if v else None,
                        range=[min(v), max(v)] if v else None)
        joint = [r for r in records if r['training'] is not None and r['held'] is not None]
        result[pair] = dict(views=len(records), training_gain=stats('training'), held_gain=stats('held'),
                           fitting_improved_held_not=sum(r['training'] > 0 and r['held'] <= 0 for r in joint),
                           fitting_not_improved_held_not=sum(r['training'] <= 0 and r['held'] <= 0 for r in joint),
                           fitting_and_held_improved=sum(r['training'] > 0 and r['held'] > 0 for r in joint),
                           records=records)
    return result


def main():
    cfg, identity = run.registration()
    checks = json.loads((run.PUBLIC / 'completion_checks.json').read_text())
    rows = []
    for ref in checks['groups']:
        assert run.artifact(ROOT / ref['path']) == ref
        rows.append(json.loads((ROOT / ref['path']).read_text()))
    assert len(rows) == 36
    summary = fit_transport_summary(rows)
    run.immutable_json(run.PUBLIC / 'fit_transport_diagnosis.json', dict(
        source_binding=run.artifact(Path(__file__)), summary=summary,
        purpose='exploratory_post_freeze_diagnosis_no_gate_or_selection_change'))
    lines = ['# Fitting Versus Held-Locality Transport', '',
             'Post-freeze descriptive diagnosis; not a new endpoint or selection rule.',
             'Each row is a dependent seed/source/fold view with its own training-only easy cut.',
             'Training uses three equally weighted localities. Held evaluation uses the fourth.',
             'Held MSE is conditional on positive causal forecast disagreement. Structural-zero',
             'rows contribute zero harm and zero harm prediction, so including them changes',
             'absolute MSE but not its within-view relative gain. No independent replication claim.', '',
             '| Pair | Views | Training improved | Held improved | Train improves, held does not | Neither improves | Both improve |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for pair, s in summary.items():
        lines.append(f"| {pair} | {s['views']} | {s['training_gain']['improved']} | {s['held_gain']['improved']} | "
                     f"{s['fitting_improved_held_not']} | {s['fitting_not_improved_held_not']} | {s['fitting_and_held_improved']} |")
    lines += ['', 'Counts diagnose this fixed objective, not causal proof of a unique failure mechanism.',
              'No future-input change, forecast fitting, C policy, deployment, Stage5C or SMC.']
    (run.PUBLIC / 'fit_transport_diagnosis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({p: {k: v for k, v in s.items() if k != 'records'} for p, s in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
