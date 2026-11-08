"""Attach verified transfer and realized-displacement evidence to the supplement."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v29 as prior

PUBLIC = prior.PUBLIC.parent / 'evidence_manuscript_v30'
DEFAULT = prior.DEFAULT.parent / PUBLIC.name
base = prior.base
REFERENCES = r"""
\bibitem{selectivenet} Y. Geifman and R. El-Yaniv. SelectiveNet:
A deep neural network with an integrated reject option.
In \emph{ICML}, pages 2151--2159, 2019.

\bibitem{spibb} R. Laroche, P. Trichelair, and R. Tachet des Combes.
Safe policy improvement with baseline bootstrapping.
In \emph{ICML}, pages 3652--3661, 2019.
"""


def evidence():
    result, digests = {}, {}
    for name, audit in (('tumdot_ing_frozen_transfer_v1', 'decomposition.json'),
                        ('opendd_hindcast_residual_v1', 'amplitude_audit.json')):
        root = PUBLIC.parent / name
        files = [root / k for k in ('registration.json', 'analysis.json', 'execution.json',
                                    'replay_execution.json', audit)]
        for p in files:
            digests[f'{name}/{p.name}'] = base.cal.digest(p)
        reg, read, start, replay, detail = [json.loads(p.read_text()) for p in files]
        digest = digests[f'{name}/analysis.json']
        if (read['registration'] != digests[f'{name}/registration.json'] or read['formal_confirmation']
            or read['deployment'] or reg['dronecrowd_confirmation_access']
            or start['fresh_exact_replay'] or not replay['fresh_exact_replay']
            or start['pid'] == replay['pid'] or any(r['analysis_sha256'] != digest for r in (start, replay))
            or detail.get('primary_sha256', detail.get('original_analysis_sha256')) != digest):
            raise ValueError(f'Cannot cite unverified development evidence: {name}')
        result[name] = read, detail
    tum, decomp = result['tumdot_ing_frozen_transfer_v1']
    od, amp = result['opendd_hindcast_residual_v1']
    tv = tum['mean']['blend05']
    macros = dict(TumCV=tv['lower_pp']['cv'], TumMotion=tv['lower_pp']['matched_hindcast'],
                  TumComplete=decomp['mean']['neural_vs_matched']['complete_only_pp'],
                  TumPass=tv['risk_passes'])
    for tag, row in (('Inc', od['summary']['incumbent']), ('Guard', od['summary']['easy_guard']),
                     ('Cap', amp['studies']['incumbent_cap']['summary']['easy_guard']),
                     ('Equal', amp['studies']['equal_displacement']['summary']['easy_guard'])):
        macros.update({tag+'CV':row['comparisons']['cv']['lower_pp']['point'], tag+'Pass':row['passed'],
                       tag+'Inc':row['comparisons']['incumbent']['lower_pp']['point']})
    macros['EqualOld'] = amp['studies']['equal_displacement']['summary']['easy_guard']['comparisons']['hindcast']['lower_pp']['point']
    return macros, digests


def build(out, style, archive):
    macros, bindings = evidence()
    doc = prior.build(out, style, archive)
    tex, defs = out/'supplement.tex', out/'supplement_values.tex'
    frag = PUBLIC/'supplement_forecast_transport.tex'
    content = base.replace_once(tex.read_text(), r'\section*{LLM Usage Statement}',
        frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}')
    tex.write_text(base.replace_once(content, r'\end{thebibliography}',
                                     REFERENCES+'\n'+r'\end{thebibliography}'))
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\Transport{k}}}{{{v:+.3f}}}'
        if isinstance(v, float) else f'\\newcommand{{\\Transport{k}}}{{{v}}}'
        for k, v in macros.items())+'\n')
    doc['source_bindings'].update(bindings)
    doc['source_bindings'][f'{PUBLIC.name}/{frag.name}'] = base.cal.digest(frag)
    for p in (tex, defs):
        doc['files'][p.name] = base.cal.digest(p)
    (out/'supplement_bindings.json').write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DEFAULT/'supplement')
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=base.v22.STYLE)
    p.add_argument('--archive', type=Path, default=base.v22.supp.base.REPLAY)
    a = p.parse_args()
    result = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        base.v22.supp.prior.package(a.output_dir, a.source_zip, result)
