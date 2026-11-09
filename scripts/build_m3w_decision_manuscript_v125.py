"""Bind the source-excluded learned choice experiment to the working paper."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v124 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v125'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    s = PUBLIC.parent / 'moving_cv_joint_selector_v1'
    d = PUBLIC.parent / 'moving_cv_joint_oof_v1'
    files = [s / f'{k}.json' for k in ('registration', 'freeze', 'readout')]
    files.extend(d / f'{k}.json' for k in ('registration', 'freeze', 'verification'))
    reg, freeze, result, _, donor, verified = [json.loads(f.read_text()) for f in files]
    digest = BASE.cal.digest
    if (result['registration'] != digest(files[0]) or result['freeze'] != digest(files[1])
            or verified['parent']['sha256'] != digest(files[4]) or reg['donor']['sha256'] != digest(files[4])
            or donor['a_labels_accessed'] or not freeze['before_a_labels'] or result['independent_confirmation']):
        raise ValueError('Require bound source-excluded forecasts and full comparison')

    def fmt(metric):
        if metric is None or metric['point'] is None:
            return 'n/a'
        lo, hi = metric['interval95']
        return f'{metric["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'

    e, p = result['agent']['effects'], result['pair']
    k = 'joint_vs_additive'
    values = dict(Edits=f'{sum(result["matched_counts"].values()):,}', Donor=donor['fits'],
        DonorSteps=f'{donor["updates"]:,}', HeadSteps=f'{freeze["training_updates"]:,}',
        Agent=fmt(e[k]['completion_lower_pp']), Pair=fmt(p[k]['lower_pixel']), ObservedPair=fmt(p[k]['observed_pixel']),
        AgentPoint=f'{e[k]["completion_lower_pp"]["point"]:+.3f}',
        PairPoint=f'{p[k]["lower_pixel"]["point"]:+.3f}' if p[k]['lower_pixel'] else 'n/a',
        MatchedAgent=fmt(e[k+'_matched']['completion_lower_pp']), MatchedPair=fmt(p[k+'_matched']['lower_pixel']),
        IndependentAgent=fmt(e['joint_vs_independent']['completion_lower_pp']),
        UniformAgent=fmt(e['joint_vs_uniform']['completion_lower_pp']),
        JointRisk=result['agent']['risk']['joint']['supported_views'],
        AddRisk=result['agent']['risk']['additive_pair']['supported_views'],
        IndependentRisk=result['agent']['risk']['independent']['supported_views'])
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out / 'manuscript.tex', out / 'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        anchor = r'views whose analytic controls are ruled out, not a learned selection gain.'
        insert = (r'views whose analytic controls are ruled out. At identical counts, causal pair coupling '
                  r'changes lower gains vs.\ additive by \CVSelectorAgentPoint{} CV-normalized points '
                  r'for agent error and \CVSelectorPairPoint{} pixels for pair displacement (supplement).')
        tex.write_text(BASE.replace_once(tex.read_text(), anchor, insert))
    else:
        tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC / 'cv_joint_selector.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(), anchor, frag.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\CVSelector{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(f.relative_to(PUBLIC.parent)): digest(f) for f in files})
    doc[key].update({f.name: digest(f) for f in (tex, defs)})
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
        PACK.supplement_package.package(out, a.source_zip, doc)
