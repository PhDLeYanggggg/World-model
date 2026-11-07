"""Add the shared-target sensitivity to the anonymous evidence supplement."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v20 as prior

base = prior.base
PUBLIC = base.HOME/'evidence_manuscript_v21'
DEFAULT = base.DATA/'evidence_manuscript_v21/supplement'


def build(out, archive, style):
    doc = prior.build(out, archive, style)
    a = prior.main.verified_analysis('native_shared_pair_completion_v1', doc['source_bindings'])
    if (not a['original_pair_bounds_reproduced'] or not a['original_risk_unchanged']
            or (a['recordings'], a['sources'], a['pair_queries']) != (34, 14, 4839)):
        raise ValueError('Expected the registered shared-target sensitivity')
    j = a['comparisons']['joint_vs_pair_additive']
    fields = {f'Overlap{k.title()}': f'$[{j[k+"_lower"]["point"]:+.4f}, {j[k+"_upper"]["point"]:+.4f}]$'
              for k in ('old', 'shared')}
    fields['OverlapNew'] = fields.pop('OverlapShared')
    fields['OverlapGain'] = f'${j["bound_tightening"]["point"]:+.4f}$'
    sec, values = (out/name for name in prior.PAYLOAD)
    block = PUBLIC/'shared_pair_note.tex'
    s = sec.read_text()
    before = r'\section{Why The Matched Recording Policies Fail}'
    refs = r'\end{thebibliography}'
    if s.count(before) != 1 or s.count(refs) != 1:
        raise ValueError('Expected the v20 insertion points')
    cite = (r"\bibitem{peyre} G. Peyr\'e and M. Cuturi. Computational optimal transport."+'\n'
            r'\emph{Foundations and Trends in Machine Learning}, 11(5--6):355--607, 2019.')
    sec.write_text(s.replace(before, block.read_text()+'\n\n'+before).replace(refs, cite+'\n\n'+refs))
    values.write_text(values.read_text()+'\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in fields.items())+'\n')
    doc['shared_target_fragment_sha256'] = base.sha(block)
    doc['files'] = {name: base.sha(out/name) for name in prior.PAYLOAD}
    (out/'supplement_bindings.json').write_text(json.dumps(doc, sort_keys=True, indent=2)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=base.REPLAY)
    p.add_argument('--style', type=Path, default=base.STYLE)
    p.add_argument('--output-dir', type=Path, default=DEFAULT)
    p.add_argument('--source-zip', type=Path)
    a = p.parse_args()
    result = build(a.output_dir, a.archive, a.style)
    if a.source_zip:
        prior.package(a.output_dir, a.source_zip, result)
