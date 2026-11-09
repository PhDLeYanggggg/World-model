"""Bind causal spline forecasts and matched capacity into the working manuscript."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v147 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.parent.PACK
PUBLIC = previous.PUBLIC.parent / 'evidence_manuscript_v148'
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    src = PUBLIC.parent / 'moving_causal_velocity_spline_v1'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json', 'slices.json',
                              'capacity_registration.json', 'capacity.json', 'tail_registration.json',
                              'tail_freeze.json', 'tail_readout.json')]
    _, f, r, s, _, cap, tail_reg, tail_freeze, tail = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1])
            or s['parent']['sha256'] != h(files[2]) or cap['registration']['sha256'] != h(files[4])
            or not f['before_held_scoring'] or not cap['outcome_aware_diagnostic_only']
            or any(p['a_labels_opened'] or p['independent_confirmation'] for p in (r, s, cap, tail))
            or tail_reg['parent']['sha256'] != h(files[2]) or tail_freeze['registration'] != h(files[6])
            or tail['freeze'] != h(files[7]) or not tail_freeze['before_held_scoring']
            or not r['original_2pct'] or not cap['original_2pct']):
        raise ValueError('Require frozen development-only spline evidence')
    summary, capacity = r['summary'], cap['summary']
    a, b = [capacity[k].get('verified', 0) for k in ('before', 'after')]
    choices = Counter(c['selected'] for c in f['cells'])
    values = dict(Updates=f'{f["optimizer_steps"]:,}', FitCount=sum(len(c['fits']) for c in f['cells']),
        Views=summary['views'], OldCapacity=a, NewCapacity=b, CapacityEdits=f'{sum(v["edits"] for v in cap["views"]):,}',
        LearnedEdits=f'{capacity["learned_path_edits"]:,}',
        SelectedChoices='/'.join(str(choices[k]) for k in ('ego', 'peers', 'completion')))

    def ci(v):
        p, (lo, hi) = v['point'], v['interval95']
        return f'{p:+.2f} [{lo:+.2f}, {hi:+.2f}]'

    for k, name in zip(('ego', 'peers', 'completion', 'selected', 'ols8', 'eqmotion', 'pct'),
                       ('Ego', 'Peers', 'Completion', 'Selected', 'LongOLS', 'EqMotion', 'PCT'), strict=True):
        v = summary['effects'][f'{k}_vs_ols4']
        values[name+'Known'] = ci(v['known_pp'])
        values[name+'Lower'] = ci(v['completion_lower_pp'])
    values['PeerAblation'] = ci(summary['effects']['peers_vs_ego']['known_pp'])
    values['PairSelected'] = ci(s['paired_geometry']['selected_vs_ols4']['known_pair_gain_pixel'])
    values['PartialDebit'] = f'{s["slices"]["completion"]["partial_debit"]["point"]:.2f}'
    values['MissingDebit'] = f'{s["slices"]["completion"]["wholly_missing_debit"]["point"]:.2f}'
    values['CapacityDelta'] = ci(dict(point=capacity['effect_pp'], interval95=capacity['interval95_pp']))
    values['TailSelected'] = ci(tail['summary']['effects']['selected_tail_vs_ols4']['completion_lower_pp'])
    values['TailMatched'] = ci(tail['summary']['effects']['taper_vs_uniform']['completion_lower_pp'])
    values['TailPasses'] = tail['summary']['risk']['selected_tail']['supported_views']
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out / 'manuscript.tex', out / 'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        anchor = 'Independent confirmation remains sealed.'
        change = (r'Training a causal velocity spline yields no conservative whole-roster gain over OLS4 '
                  r'for the calibration-selected arm (supplement). Independent confirmation remains sealed.')
        # The sentence is also used in the abstract; the scoped replacement is in limitations.
        old_text = tex.read_text()
        marker = r'\section{Limitations and Independent Evidence}'
        first, last = old_text.split(marker, 1)
        content = first + marker + BASE.replace_once(last, anchor, change)
    else:
        tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC / 'spline_forecast.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        content = BASE.replace_once(tex.read_text(), anchor, frag.read_text() + '\n' + anchor)
    tex.write_text(content)
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\VelSpline{k}}}{{{v}}}' for k, v in values.items()) + '\n')
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
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
