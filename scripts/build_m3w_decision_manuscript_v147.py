"""Bind the OLS-anchored capacity experiment into the working paper."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v146 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

parent, BASE = previous.parent, previous.BASE
PUBLIC = previous.PUBLIC.parent / 'evidence_manuscript_v147'
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    src = PUBLIC.parent / 'moving_ols_anchor_menu_v2'
    files = [src / name for name in ('registration.json', 'readout.json', 'location_readout.json')]
    reg, result, location = [json.loads(f.read_text()) for f in files]
    h = BASE.cal.digest
    if (result['registration']['sha256'] != h(files[0]) or not reg['exact_query_counts']
            or not reg['outcome_aware_diagnostic_only'] or not result['original_2pct']
            or result['a_labels_opened'] or result['independent_confirmation']
            or location['parent']['sha256'] != h(files[1])
            or location['original_rule_changed'] or not location['risk_relaxed_diagnostic_only']):
        raise ValueError('Require matched development-only capacity evidence')
    r = result['summary']
    a, b = [r['statuses'][k] for k in ('old', 'expanded')]
    point, (lo, hi) = r['effect_pp'], r['ci95_pp']
    values = dict(Edits=f'{r["edits"]:,}', Views=r['views'],
        Old=a.get('verified', 0), Expanded=b.get('verified', 0),
        OldBound=a.get('nonpositive_bound', 0), NewBound=b.get('nonpositive_bound', 0),
        OldUnknown=a.get('unresolved', 0), NewUnknown=b.get('unresolved', 0),
        Empty=a.get('no_edits', 0),
        OldRisk=a.get('risk_infeasible', 0), NewRisk=b.get('risk_infeasible', 0),
        OldSupport=a.get('undefined_support', 0), NewSupport=b.get('undefined_support', 0),
        Effect=f'{point:+.2f} [{lo:+.2f}, {hi:+.2f}]', Examined=location['examined'],
        Reallocate=location['counts'].get('positive_upper_only_after_reallocating', 0),
        StillNegative=location['counts'].get('still_nonpositive_after_reallocating', 0),
        QueryPositive=location['counts'].get('positive_upper_at_exact_query_counts', 0))
    if part == 'main':
        doc = parent.build(out, style, archive, 'main')
        tex, defs = out / 'manuscript.tex', out / 'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        anchor = 'None of the calibration or development comparisons substitutes for the 30 sealed confirmation groups.'
        update = (r'At fixed query counts, both path banks yield verified hindsight plans in '
                  r'\OLSMenuExpanded{} of \OLSMenuViews{} views. Even with risk removed and edits reallocated '
                  r'within recordings, OLS4 remains out of reach in \OLSMenuStillNegative{} of '
                  r'\OLSMenuExamined{} excluded views (supplement). Independent confirmation remains sealed.')
        content = BASE.replace_once(tex.read_text(), anchor, update)
    else:
        doc = previous.build(out, style, archive)
        tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        fragment = PUBLIC / 'ols_anchor_capacity.tex'
        files.append(fragment)
        anchor = r'\section*{LLM Usage Statement}'
        content = BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor)
    tex.write_text(content)
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\OLSMenu{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): h(p) for p in files})
    doc[key].update({p.name: h(p) for p in (tex, defs)})
    (out / receipt).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT / ('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == 'main':
        parent.PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
