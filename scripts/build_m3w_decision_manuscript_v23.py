"""Bind the tail-center stress test and coherent witnesses to the anonymous draft."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v22 as prior

PUBLIC = prior.base.HOME/'evidence_manuscript_v23'
DEFAULT = prior.base.DATA/'evidence_manuscript_v23'


def build(out, style, archive, part):
    doc = prior.build(out, style, archive, part)
    audit = prior.main.v13.verified_analysis('native_completion_occlusion_v1', doc['source_bindings'])
    test = prior.main.v13.verified_analysis('native_motion_centered_completion_v1', doc['source_bindings'])
    a, b = (audit['pseudo_gap_diagnostics'][f'terminal_12_{x}'] for x in ('constant','ols4'))
    if (not audit['synthetic_deletion_only'] or a['gaps'] != b['gaps'] or not test['opposite_signs']['1']
            or test['missing_labels_imputed'] or test['official_rule_changed'] or test['independent_confirmation']):
        raise ValueError('Expected only the replay-verified exposed tail-center sensitivity')
    fields = dict(TailGaps=f'{a["gaps"]:,}', TailConstant=f'{100*a["gap_1x"]["point"]:.2f}',
                  TailOLS=f'{100*b["gap_1x"]["point"]:.2f}',
                  TailDelta=f'{100*audit["primary"]["point"]:+.2f}',
                  TailDeltaCI=', '.join(f'{100*x:+.2f}' for x in audit['primary']['interval95']),
                  TailMoved=f'{test["counts"]["shifted_eligible_steps"]:,}',
                  TailSteps=f'{test["counts"]["terminal_steps"]:,}',
                  **{key: f'{test["cases"][case]["point"]:+.4f}' for key, case in (
                      ('TailLow','low_1x'), ('TailHigh','high_1x'), ('TailReference','recentered_reference'))})
    if part == 'main':
        names = ('structured_results.tex', 'structured_values.tex', 'source_bindings.json', 'build_bindings')
    else:
        names = ('supplement.tex', 'supplement_values.tex', 'supplement_bindings.json', 'files')
    tex, macros, manifest = (out/x for x in names[:3])
    frag = PUBLIC/f'{part}_tail_center.tex'
    text = tex.read_text()
    if part == 'supplement':
        anchor = r'\section{Why The Matched Recording Policies Fail}'
        if text.count(anchor) != 1:
            raise ValueError('Expected one insertion point')
        tex.write_text(text.replace(anchor, frag.read_text()+'\n\n'+anchor))
    else:
        tex.write_text(text+'\n\n'+frag.read_text())
    macros.write_text(macros.read_text()+'\n'.join(f'\\newcommand{{\\GapProbe{k}}}{{{v}}}' for k,v in fields.items())+'\n')
    doc['source_bindings'][f'{PUBLIC.name}/{frag.name}'] = prior.base.sha(frag)
    for p in (tex, macros):
        doc[names[-1]][p.name] = prior.base.sha(p)
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main','supplement'))
    p.add_argument('--archive', type=Path, default=prior.supp.base.REPLAY)
    p.add_argument('--style', type=Path, default=prior.STYLE)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--source-zip', type=Path)
    a = p.parse_args()
    out = a.output_dir or DEFAULT/('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == 'main':
            prior.main.v13.package(out, a.source_zip, doc['build_bindings'])
        else:
            prior.supp.prior.package(out, a.source_zip, doc)
