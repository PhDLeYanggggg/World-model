"""Build the v173 manuscript with verified citations and scoped comparisons."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_focused_candidate as focused


scope = focused.scope
main, supplement = focused.main, focused.supplement
PARENT = focused.MENU_DESTINATION
OUT = PARENT.parent/'evidence_manuscript_v174_verified_references'
RECEIPT = main.PUBLIC/'evidence_manuscript_v174_verified_references/receipt.json'
ADDITIONS = ('graphabstention', 'crc', 'geotrax')
BIB_END = r'\end{thebibliography}'
EDITS = (
    ('as a training teacher, not as a deployable policy.',
     r'''as a training teacher, not as a deployable policy.
Graph abstention evaluates whether to retain a whole multivariate
forecast~\cite{graphabstention}; we always return a forecast for each
agent and evaluate which subset benefits from replacing CV. Conformal
risk control bounds expected loss under assumptions on the calibration
units and loss family~\cite{crc}. Our three recording checks are
empirical. Adding an edit can increase or decrease the harm-to-reference
ratio, and the selected easy reference can be absent.'''),
    (r'Songdo Traffic~\cite{songdo} provides 40', r'Songdo Traffic~\cite{songdo,geotrax} provides 40'),
    (r'motion prediction. \emph{IEEE Robotics and Automation Letters}, 2020.',
     r'motion prediction. \emph{IEEE Robotics and Automation Letters}, 5(2):1696--1703, 2020.'),
    (r'Regression with multi-expert deferral. In \emph{ICML}, 2024.',
     r'Regression with multi-expert deferral. In \emph{ICML}, pages 34738--34759, 2024.'),
    (r'A benchmark and methods. \emph{arXiv:2601.12500}, 2026.',
     r'A benchmark and methods. \emph{arXiv:2601.12500v3}, 2026.'),
    (r'interaction reasoning. In \emph{CVPR}, 2023.', r'interaction reasoning. In \emph{CVPR}, pages 1410--1420, 2023.'),
    (r'''L. Wen et al. Detection, tracking, and counting meets drones in
crowds: A benchmark. In \emph{CVPR}, 2021.''',
     r'''L. Wen, D. Du, P. Zhu, Q. Hu, Q. Wang, L. Bo, and S. Lyu.
Detection, tracking, and counting meets drones in crowds: A benchmark.
In \emph{CVPR}, pages 7812--7821, 2021.'''),
    (r'In \emph{AAAI}, 2019.', r'In \emph{AAAI}, pages 1658--1665, 2019.'),
    (r'In \emph{ICCV}, 2023.', r'In \emph{ICCV}, pages 20315--20326, 2023.'),
    (r'In \emph{IROS}, 2021.', r'In \emph{IROS}, pages 8059--8066, 2021.'),
    (r'In \emph{ICCV}, 2025.', r'In \emph{ICCV}, pages 27872--27884, 2025.'),
    (r'In \emph{WACV}, 2022.', r'In \emph{WACV}, pages 503--512, 2022.'),
    (r'''R. Fonod, H. Cho, H. Yeo, and N. Geroliminis. Songdo Traffic v2.
\emph{Zenodo:17924857}, 2025.''',
     r'''R. Fonod, H. Cho, H. Yeo, and N. Geroliminis.
Songdo Traffic: High accuracy georeferenced vehicle trajectories from a
large-scale study in a smart city. \emph{Zenodo:17924857}, version 2, 2025.'''),
)
NEW_REFERENCES = r'''\bibitem{graphabstention}
J. Xie and B. Alsinglawi. Structure-aware graph abstention for reliable
selective forecasting. \emph{arXiv:2610.08322}, 2026.

\bibitem{crc}
A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster.
Conformal risk control. In \emph{ICLR}, 2024.

\bibitem{geotrax}
R. Fonod, H. Cho, H. Yeo, and N. Geroliminis. Advanced computer vision
for extracting georeferenced vehicle trajectories from drone imagery.
\emph{Transportation Research Part C: Emerging Technologies},
178:105205, 2025.

'''


def revise(source):
    result = source
    for a, b in (*EDITS, (BIB_END, NEW_REFERENCES+BIB_END)):
        if result.count(a) != 1:
            raise ValueError('Expected one occurrence of audited citation anchor: '+a)
        result = result.replace(a, b, 1)
    restored = result
    for a, b in reversed((*EDITS, (BIB_END, NEW_REFERENCES+BIB_END))):
        restored = restored.replace(b, a, 1)
    if restored != source:
        raise ValueError('Unexpected change outside verified citation edits')
    return result


def build(destination, compiler, receipt):
    parent = supplement.read_receipt('evidence_manuscript_v173_songdo_menu')
    inputs = {k: supplement.checked(PARENT/k, h) for k, h in parent['artifacts'].items()}
    with ZipFile(BytesIO(inputs['main_source.zip'])) as z:
        files = {k: z.read(k) for k in z.namelist()}
        if len(files) != len(z.namelist()) or any(Path(k).name != k for k in files):
            raise ValueError('Expected a flat, unique source archive')
    if files['manuscript.tex'] != inputs['manuscript.tex']:
        raise ValueError('Source archive does not contain the verified manuscript')
    prior_keys = scope.citation.used_citations(files)
    files['manuscript.tex'] = revise(files['manuscript.tex'].decode('ascii')).encode('ascii')
    keys = scope.citation.used_citations(files)
    if keys != sorted([*prior_keys, *ADDITIONS]):
        raise ValueError('The bibliography and citations changed unexpectedly')
    style = supplement.checked(scope.citation.main.DRAFT/'cvpr.sty', scope.citation.STYLE_SHA)
    with TemporaryDirectory(prefix='verified-references-') as temp:
        root = Path(temp)
        for k, raw in {**files, 'cvpr.sty': style}.items():
            (root/k).write_bytes(raw)
        p = subprocess.run([str(compiler), '-X', 'compile', '--untrusted', '--only-cached', '--keep-logs',
                            'manuscript.tex'], cwd=root, env={**os.environ, 'SOURCE_DATE_EPOCH': '0'},
                           capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise RuntimeError(p.stdout+'\n'+p.stderr)
        if re.search(r'(?i)undefined (citation|reference)|Citation .* undefined|Overfull \\hbox',
                     (root/'manuscript.log').read_text()):
            raise ValueError('Unresolved citation/reference or overfull horizontal box')
        pdf = (root/'manuscript.pdf').read_bytes()
        pages = focused.check_render(pdf, inputs['manuscript.pdf'], (
            '197/480', '0.000257', '0.00000472', '2601.12500v3', '2610.08322', '105205'))
        scope.citation.archive.encode(root/'main_source.zip', files)
        output = {'manuscript.tex': files['manuscript.tex'], 'manuscript.pdf': pdf,
                  'main_source.zip': (root/'main_source.zip').read_bytes()}
    result = dict(pages=pages, bibliography_entries=len(keys), added_citations=list(ADDITIONS),
                  only_changed_archive_member='manuscript.tex', prior_manuscript_sha256=parent['artifacts']['manuscript.pdf'],
                  artifacts={k: main.digest(v) for k, v in output.items()},
                  title_abstract_disclosure_and_experiments_unchanged=True, no_outcomes_opened=True,
                  no_new_experiment=True, uploaded=False, submission_ready=False)
    deliver = {**{destination/k: v for k, v in output.items()}, receipt: main.as_json(result)}
    if any(k.exists() and k.read_bytes() != v for k, v in deliver.items()):
        raise FileExistsError('Refusing to replace a different candidate artifact')
    for k, v in deliver.items():
        k.parent.mkdir(parents=True, exist_ok=True)
        if not k.exists():
            with k.open('xb') as f:
                f.write(v)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--receipt', type=Path, default=RECEIPT)
    a = p.parse_args()
    build(a.output, a.compiler.resolve(), a.receipt)
