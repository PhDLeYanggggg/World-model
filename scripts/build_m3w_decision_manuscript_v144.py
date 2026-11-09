"""Bind the causal prefix-state ablation to the working supplement."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v143 as parent
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v144'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        return doc
    src = PUBLIC.parent / 'moving_cv_prefix_state_v1'
    fragment = PUBLIC / 'prefix_state.tex'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json',
        'prediction_audit.json', 'arithmetic_readout.json')]
    reg, f, r, audit, control = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1])
            or not f['before_held_scoring'] or not reg['causal_prefix'] or not control['matched_exact_counts']
            or any(z['parent']['sha256'] != h(files[2]) for z in (audit, control))
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct']
                   for z in (r, audit, control))):
        raise ValueError('Require frozen causal-prefix development evidence')
    risk = r['summary']['risk']
    keys = ('masked_matched', 'state_matched', 'loop_matched')
    if len({risk[k]['switches'] for k in keys}) != 1:
        raise ValueError('Query-matched prefix counts changed')
    values = dict(Updates=f'{f["optimizer_steps"]:,}', Edits=f'{risk["state_matched"]["switches"]:,}',
        Passes='/'.join(str(risk[k]['supported_views']) for k in keys), Qual=r['qualified'],
        Worst='/'.join(f'{risk[k]["worst_signed_easy_percent"]:.2f}' for k in keys))

    def ci(v, precision=3):
        p, lo, hi = (f'{x:+.{precision}f}' for x in (v['point'], *v['interval95']))
        return f'{p} [{lo}, {hi}]'

    for name, k in (('State', 'state_at_count'), ('Loop', 'loop_at_count'), ('Independent', 'state_vs_independent')):
        values[name] = ci(r['summary']['effects'][k]['completion_lower_pp'])
    values['Arithmetic'] = ci(control['summary']['effects']['arithmetic_at_count']['completion_lower_pp'])
    for name, k in (('Gain', 'gain_mae'), ('Easy', 'signed_easy_mae')):
        values[name] = ci(audit['prediction'][k]['masked_minus_state'], 4)
    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\PrefixState{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): h(p) for p in (*files, fragment)})
    doc['files'].update({p.name: h(p) for p in (tex, defs)})
    (out / 'supplement_bindings.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__=='__main__':
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
