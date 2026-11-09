"""Add the matched camera perturbation and action-capacity results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v161 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v162'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent
    source = root/'camera_noise_control_v1'
    files = [source/f'{n}.json' for n in ('registration', 'freeze', 'readout', 'development_capacity')]
    reg, freeze, readout, cap = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (readout['registration'] != sha(files[0]) or freeze['registration'] != sha(files[0])
            or readout['freeze']['sha256'] != sha(files[1]) or cap['frozen_predictions']['sha256'] != sha(files[1])
            or cap['camera_registration'] != sha(files[0]) or freeze['selection']
            or not freeze['no_camera_outcomes_read'] or not cap['labels_hindsight_only']
            or any(d['external_test_reopened'] or d['independent_confirmation'] or not d['original_2pct']
                   for d in (reg, readout, cap))):
        raise ValueError('Require matched development-only evidence')
    full = [r['per_menu'] for r in cap['seeds'].values()]
    gain = lambda r: r['full']['witness']['lower_vs_ols4']
    increment = [gain(r['with_splines'])-gain(r['analytic']) for r in full]
    camera = readout['cameras']
    values = dict(Updates=f"{freeze['fit_steps']:,}",
                  Capped=sum(r['with_splines']['clipped']['solver']['status']=='nonpositive_bound' for r in full),
                  Analytic=sum(r['analytic']['full']['solver']['status']=='verified' for r in full),
                  Spline=sum(r['with_splines']['full']['solver']['status']=='verified' for r in full),
                  Extra=f'{min(increment):.2f}--{max(increment):.2f}')
    for site, label in (('wepdtof', 'Wep'), ('gmu', 'Gmu'), ('cmu', 'Cmu')):
        r = camera[site]
        for field in ('online','stable','complete'):
            values[label+field.title()] = r[field]
        for other in ('clean', 'ols4', 'median4'):
            for i, mask in enumerate(('Known', 'Lower')):
                z = [s['noise_gain_known_lower_pp'][other][i] for s in r['seeds'].values()]
                values[label+other.title().replace('4','')+mask] = f'{min(z):+.2f} to {max(z):+.2f}'
    easy = [100*s['original_risk']['noise']['completion_easy_risk_upper'] for s in camera['wepdtof']['seeds'].values()]
    values['Easy'] = f'{min(easy):.2f}--{max(easy):.2f}'

    doc = previous.build(out, style, archive, part)
    if part == 'main':
        key, receipt = 'build_bindings', 'source_bindings.json'
        tex, defs = out/'manuscript.tex', out/'evidence_values.tex'
        text = BASE.replace_once(tex.read_text(),
            'decision recovering it under a recording risk constraint.',
            'decision recovering it under a recording risk constraint. '
            r'Raw-video work studies tracking errors~\cite{yu}, robust denoising~\cite{natra} '
            r'and direct forecasting from detections~\cite{realtraj}. '
            'Here a more accurate forecast is a candidate action, not evidence that replacement is safe.')
        text = BASE.replace_once(text,
            r'(\WepCapObserved{} pp best upper bound; supplement).',
            r'(\WepCapObserved{} pp best upper bound). Allowing the full candidate paths at the same '
            r'count caps yields hindsight witnesses in \CamNoiseAnalytic{}/3 pilot seed views, even '
            'without a neural candidate (supplement).')
        text = BASE.replace_once(text, r'\begin{thebibliography}{5}', r'\begin{thebibliography}{99}')
        bib = r'''
\bibitem{yu}
R. Yu and Z. Zhou. Towards robust human trajectory prediction in raw videos.
In \emph{IROS}, 2021.

\bibitem{natra}
R. Li, C. Li, R. Lv, Y. Li, Y. Gao, X. Zhang, and J. Zhou.
NATRA: Noise-agnostic framework for trajectory prediction with noisy observations.
In \emph{ICCV}, 2025.

\bibitem{realtraj}
R. Fujii, H. Saito, and R. Hachiuma.
RealTraj: Towards real-world pedestrian trajectory forecasting.
In \emph{ICRA}, 2026.

'''
        text = BASE.replace_once(text, r'\end{thebibliography}', bib+r'\end{thebibliography}')
    else:
        key, receipt = 'files', 'supplement_bindings.json'
        tex, defs = out/'supplement.tex', out/'supplement_values.tex'
        frag = PUBLIC/'camera_noise_control.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(), anchor, frag.read_text()+'\n'+anchor)
    tex.write_text(text)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\CamNoise{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(root)): sha(p) for p in files})
    doc[key].update({p.name:sha(p) for p in (tex,defs)})
    (out/receipt).write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part=='main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part=='main':
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
