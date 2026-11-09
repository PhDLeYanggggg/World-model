"""Build the unchanged v166 paper with direct WEPDTOF attribution."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import package_m3w_main_claims as main
from scripts import package_m3w_reviewer_evidence as archive

BASE = main.DRAFT.parent
DEFAULT = BASE.parent/'evidence_manuscript_v167_citation'
PDF_SHA = '3541d2c7cc29b04ea8db3c74151a6850530d43ac4a071f3e8dc7add15210e6ad'
ZIP_SHA = '6a43eb501acf5c12ece358339e2c6f9f9c2c904a4117af9a7acabb0cbeb92532'
STYLE_SHA = '1cd7f434b13d41c475df4a1901a5d82d9ca8b80b8cf6205dc4602a5c164286b4'
KIT_SHA = 'da1ec5db9b73531ea82a87723e4aedb631083331fe655d1972279b48a7bdb2fe'
OLD = 'WEPDTOF provides a frozen source-disjoint evaluation'
NEW = r'WEPDTOF~\cite{wepdtof} provides a frozen source-disjoint evaluation'
BIB_START = r'\begin{thebibliography}{99}'
BIB_END = r'\end{thebibliography}'
BIB_SMALL = '{' + r'\small' + '\n' + BIB_START
REFERENCE = r"""\bibitem{wepdtof}
O. Tezcan, Z. Duan, M. Cokbas, P. Ishwar, and J. Konrad.
WEPDTOF: A dataset and benchmark algorithms for in-the-wild people
detection and tracking from overhead fisheye cameras. In \emph{WACV}, 2022.

"""


def amend(text):
    if any(text.count(s) != 1 for s in (OLD, BIB_START, BIB_END)) or r'\bibitem{wepdtof}' in text:
        raise ValueError('Expected the original WEPDTOF sentence and bibliography')
    return (text.replace(OLD, NEW).replace(BIB_START, BIB_SMALL)
            .replace(BIB_END, REFERENCE+BIB_END+'}'))


def used_citations(files):
    tex = '\n'.join(v.decode() for k, v in files.items() if k.endswith('.tex'))
    keys = re.findall(r'\\bibitem\{([^{}]+)\}', tex)
    used = {c.strip() for s in re.findall(r'\\cite\{([^{}]+)\}', tex) for c in s.split(',')}
    if len(keys) != len(set(keys)) or set(keys) != used:
        raise ValueError('Bibliography and in-text citations do not match')
    return sorted(used)


def verified_inputs():
    zip_path, pdf_path, style = BASE/'main_source.zip', main.DRAFT/'manuscript.pdf', main.DRAFT/'cvpr.sty'
    for p, expect in ((zip_path, ZIP_SHA), (pdf_path, PDF_SHA), (style, STYLE_SHA)):
        if main.digest(p.read_bytes()) != expect:
            raise ValueError(f'Original artifact changed: {p.name}')
    with ZipFile(zip_path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
            raise ValueError('Unexpected archive member')
        files = {n: z.read(n) for n in names}
    for n, raw in files.items():
        if n != 'README.txt' and (main.DRAFT/n).read_bytes() != raw:
            raise ValueError(f'Working source differs from the original archive: {n}')
    return files, pdf_path, style


def check_pdf(path, original):
    a, b = PdfReader(original), PdfReader(path)
    if len(a.pages) != 8 or len(b.pages) != 8 or b.metadata.author not in (None, '', 'Anonymous'):
        raise ValueError(f'Expected eight pages and anonymous metadata; got {len(a.pages)} and {len(b.pages)} pages')
    page_text = lambda p: ' '.join(p.extract_text().split())
    unchanged = []
    for i, (x, y) in enumerate(zip(a.pages, b.pages, strict=True), 1):
        u, v = page_text(x), page_text(y)
        if u == v:
            unchanged.append(i)
        for ref in y.get('/Annots', []):
            action = ref.get_object().get('/A')
            if action and action.get('/S') in ('/URI', '/Launch', '/GoToR', '/JavaScript'):
                raise ValueError('Review PDF has an outbound or active action')
    if unchanged[:6] != list(range(1, 7)):
        raise ValueError('Citation changed the early pages')
    text = ' '.join(map(page_text, b.pages))
    if ('Tezcan' not in text or 'Anonymous CVPR submission' not in text
            or 'LLM Usage Statement' not in text or '[?]' in text):
        raise ValueError('Citation or anonymous review text was not rendered')
    return unchanged


def build(destination, compiler, receipt=None):
    files, original, style = verified_inputs()
    keys = used_citations(files)
    current = files['manuscript.tex'].decode()
    revised = amend(current)
    restored = revised.replace(NEW, OLD).replace(REFERENCE, '').replace(
        BIB_SMALL, BIB_START).replace(BIB_END+'}', BIB_END)
    if restored != current:
        raise AssertionError('Non-citation manuscript change')
    files['manuscript.tex'] = revised.encode()
    if used_citations(files) != sorted([*keys, 'wepdtof']):
        raise ValueError('New attribution not linked')
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='citation-build-', dir=destination) as scratch:
        root = Path(scratch)
        for n, raw in {**files, 'cvpr.sty': style.read_bytes()}.items():
            (root/n).write_bytes(raw)
        build_cmd = [str(compiler), '-X', 'compile', '--untrusted', '--only-cached', '--keep-logs', 'manuscript.tex']
        run = subprocess.run(build_cmd, cwd=root, capture_output=True, text=True,
                             timeout=120, env={**os.environ, 'SOURCE_DATE_EPOCH': '0'})
        if run.returncode:
            raise RuntimeError(run.stdout+'\n'+run.stderr)
        log = (root/'manuscript.log').read_text()
        if re.search(r'(?i)undefined (citation|reference)|Citation .* undefined|Overfull \\hbox', log):
            raise ValueError('Unresolved citation/reference or overfull horizontal box')
        pdf = root/'manuscript.pdf'
        same_pages = check_pdf(pdf, original)
        version = subprocess.run([str(compiler), '--version'], check=True, capture_output=True,
                                 text=True, timeout=15).stdout.strip()
        archive.encode(root/'main_source.zip', files)
        result = dict(pages=8, bibliography_entries=len(keys)+1, text_unchanged_pages=same_pages,
                      only_changed_archive_member='manuscript.tex', style_sha256=STYLE_SHA,
                      official_template_zip_sha256=KIT_SHA, original_main_pdf_sha256=PDF_SHA,
                      original_source_zip_sha256=ZIP_SHA, llm_statement_unchanged=True, compiler=version,
                      bibliography_uses_template_small_size=True,
                      artifact_sha256={p: main.digest((root/p).read_bytes()) for p in
                                       ('manuscript.pdf', 'main_source.zip', 'manuscript.tex')},
                      no_external_outcomes_opened=True, submitted=False, submission_ready=False)
        outputs = ('manuscript.pdf', 'main_source.zip', 'manuscript.tex', 'manuscript.log')
        for n in outputs:
            if (destination/n).exists() and (destination/n).read_bytes() != (root/n).read_bytes():
                raise FileExistsError(f'Refusing to replace a different candidate: {n}')
        for n in outputs:
            (root/n).replace(destination/n)
    if receipt:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_bytes(main.as_json(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--receipt', type=Path)
    args = p.parse_args()
    build(args.output, args.compiler.resolve(), args.receipt)
