"""Bind the Moving pooled-budget upper bound and exact risk witnesses."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v155 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent / 'evidence_manuscript_v156'
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent
    src = root / 'moving_record_peak_allocation_v1'
    files = [src / k for k in ('registration.json', 'readout.json', 'break_even.json', 'repair.json')]
    g, r, b, a = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (r['registration'] != h(files[0]) or b['parent']['sha256'] != h(files[1])
            or a['binding']['parent']['sha256'] != h(files[1])
            or any(z['a_labels_opened'] or z['confirmation_access'] or not z['original_2pct'] for z in (g, r, b, a))
            or any(not z['hindsight_only'] or z['independent_confirmation'] for z in (r, b, a))
            or not a['no_exclusion_from_kernel'] or not b['ignores_risk_and_count'] or r['deployment_changed']):
        raise ValueError('Require the checked development-only capacity experiment')
    s = r['summary']
    v = dict(Excluded=s['arbitrary']['record']['ols4_excluded'], Views=s['views'], Records=s['recordings'],
             Sources=s['sources'], Anchor=s['anchor_dual'], PositiveBudget=b['summary']['with_missing_debit']['finite_nonzero_budget_views'],
             KnownMultiple=f'{b["summary"]["known_only"]["median_multiple"]:.2f}',
             DebitedMultiple=f'{b["summary"]["with_missing_debit"]["median_multiple"]:.2f}',
             KnownGap=f'{b["summary"]["median_gap_pp"]["gap_known_pp"]:.2f}',
             MissingGap=f'{b["summary"]["median_gap_pp"]["gap_missing_pp"]:.2f}',
             QueryMedian=f'{s["arbitrary"]["query"]["median_pp"][1]:+.2f}',
             RecordMedian=f'{s["arbitrary"]["record"]["median_pp"][1]:+.2f}', Rechecked=a['examined_views'])
    for k, name in (('base_query','BaseQuery'), ('base_record','BaseRecord'),
                    ('expanded_query','ExpandedQuery'), ('expanded_record','ExpandedRecord')):
        t = s['status'][k]
        v[name + 'Verified'] = t['verified'] + a['added_verified'][k]
        v[name + 'Unresolved'] = t['unresolved'] - a['added_verified'][k]
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out / 'manuscript.tex', out / 'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        text = BASE.replace_once(tex.read_text(),
            r'even an arbitrary one-query replacement cannot beat OLS4 in \CrossDomainExcluded{}/\CrossDomainViews{} views at the frozen displacement budget.',
            r'even arbitrary paths pooled across all queries cannot beat OLS4 in \RecordPeakExcluded{}/\RecordPeakViews{} views at the frozen displacement budget.')
    else:
        tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC / 'record_peak_allocation.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(), anchor, frag.read_text() + '\n' + anchor)
    tex.write_text(text)
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\RecordPeak{k}}}{{{val}}}' for k, val in v.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(root)): h(p) for p in files})
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
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
