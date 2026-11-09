"""Bind the matched OpenDD velocity-frame experiment to the manuscript."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v148 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent / 'evidence_manuscript_v149'
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    src = PUBLIC.parent / 'opendd_matched_spline_v1'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json',
                              'slot_registration.json', 'slot_readout.json')]
    reg, f, r, slot_reg, slots = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1])
            or not f['before_held_scoring'] or reg['model_selection']
            or r['independent_confirmation'] or r['confirmation_access'] or not r['original_2pct']
            or slots['registration']['sha256'] != h(files[3])
            or slot_reg['parents'][1]['sha256'] != h(files[2]) or slots['confirmation_access']
            or slots['independent_confirmation'] or not slots['original_2pct']):
        raise ValueError('Require the matched development forecast experiment')
    summary = r['summary']
    roster = summary['roster']
    values = dict(Fits=f['fits'], Updates=f'{f["updates"]:,}', Rows=f'{roster["rows"]:,}',
        Complete=f'{roster["complete"]:,}', Sites=roster['sites'], Recordings=roster['recordings'],
        RiskPasses=summary['risk']['comoving_spline']['passes'],
        RiskViews=summary['risk']['comoving_spline']['views'],
        WorstEasy=f'{summary["risk"]["comoving_spline"]["worst_signed_easy_percent"]:+.2f}')

    def ci(v):
        a, (lo, hi) = v['point'], v['interval95']
        return f'{a:+.3f} [{lo:+.3f}, {hi:+.3f}]'

    for k, name in zip(('cv', 'ols4', 'comoving_ego', 'ols_spline', 'cv_spline', 'comoving_spline'),
                       ('CV', 'OLS', 'Eq', 'OLSSpline', 'CVSpline', 'Moving'), strict=True):
        a = summary['raw'][k]
        values[name+'ADE'] = f'{a["complete_ade"]:.3f}'
        values[name+'FDE'] = f'{a["endpoint_fde"]:.3f}'
        values[name+'Gain'] = ci(a['lower']['cv'])
    for k, name in (('cv_anchor_vs_ols', 'CVAnchor'), ('comoving_vs_cv', 'MovingAgainstCV'),
                    ('comoving_vs_centered', 'MovingCentered'), ('comoving_vs_eqmotion', 'MovingEq')):
        values[name] = ci(summary['effects'][k]['lower'])
    by = {site: [x for x in r['rows'] if x['site'] == site] for site in sorted(reg['sites'])}
    smallest = min(by, key=lambda s: by[s][0]['raw']['complete'])
    means = np.array([np.mean([x['paired']['comoving_vs_cv']['lower'] for x in rows])
                      for s, rows in by.items() if s != smallest])
    draw = np.random.default_rng(817).integers(len(means), size=(3000, len(means)))
    values.update(SmallComplete=by[smallest][0]['raw']['complete'], LeaveSmallest=ci(dict(
        point=means.mean(), interval95=np.quantile(means[draw].mean(1), [.025, .975]))))
    values.update(Slots=f'{slots["policies"]["anchor"]["edits"]:,}',
        OldDual=slots['policies']['anchor']['dual_positive'],
        SplineDual=slots['policies']['comoving_spline']['dual_positive'],
        EqCapacity=slots['capacity']['comoving_ego']['qualified'],
        SplineCapacity=slots['capacity']['comoving_spline']['qualified'],
        SlotEffect=ci(slots['effects']['comoving_spline']),
        Undefined=slots['policies']['comoving_spline']['undefined'])
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out/'manuscript.tex', out/'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        marker = r'\section{Limitations and Independent Evidence}'
        first, last = tex.read_text().split(marker, 1)
        anchor = 'Independent confirmation remains sealed.'
        text = (r'On OpenDD, the matched comoving spline gains \OpenSplineMovingGain{} pp versus CV '
                r'but does not improve the fixed-slot policy (supplement). '
                r'Independent confirmation remains sealed.')
        content = first+marker+BASE.replace_once(last, anchor, text)
        content = BASE.replace_once(content, 'Replaying aggregates\ndoes not reproduce upstream training or add independent evidence.\n'
            'The submission package is not yet complete.', 'Aggregate replay does not reproduce training or add independent evidence.')
    else:
        tex, defs = out/'supplement.tex', out/'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC/'matched_opendd_spline.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        content = BASE.replace_once(tex.read_text(), anchor, frag.read_text()+'\n'+anchor)
    tex.write_text(content)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(
        rf'\newcommand{{\OpenSpline{k}}}{{{v}}}' for k, v in values.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): h(p) for p in files})
    doc[key].update({p.name: h(p) for p in (tex, defs)})
    (out/receipt).write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == 'main':
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
