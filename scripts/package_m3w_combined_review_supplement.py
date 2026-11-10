"""Combine the unchanged PDFs and both independent offline replays."""

import argparse
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import package_m3w_reviewer_evidence as archive

main = archive.main
TEMPLATE = archive.ROOT/'reproducibility/combined_review_supplement'
OUT = archive.PUBLIC/'combined_review_supplement_v1'
LOCAL = archive.LOCAL
DRAFT = main.PRIVATE/'evidence_manuscript_v167_citation'
TECHNICAL = main.PRIVATE/'evidence_manuscript_v165/supplement/supplement.pdf'
TECHNICAL_SHA = 'fb80b7d4bf72bd2eb170248a5bb8d18424d8e9b3bc73f1d4e8a78a959fe76f63'
MANTRA = ('https://openaccess.thecvf.com/content_CVPR_2020/html/'
          'Marchetti_MANTRA_Memory_Augmented_Networks_for_Multiple_Trajectory_Prediction_CVPR_2020_paper.html')


def check_pdf(raw, pages, citations=frozenset()):
    pdf = PdfReader(BytesIO(raw))
    if pdf.is_encrypted or len(pdf.pages) != pages or (pdf.metadata or {}).get('/Author') not in (None, '', 'Anonymous'):
        raise ValueError('Unexpected page count, encryption, or author metadata')
    root = pdf.root_object
    names = root.get('/Names', {})
    if any(k in names for k in ('/EmbeddedFiles', '/JavaScript')) or any(
            k in root for k in ('/AA', '/AcroForm', '/AF', '/Collection')):
        raise ValueError('Unexpected interactive or attached PDF content')
    if '/OpenAction' in root:
        action = root['/OpenAction']
        local_action = isinstance(action, DictionaryObject) and action.get('/S') == '/GoTo'
        local_dest = isinstance(action, ArrayObject) and len(action) >= 2 and str(action[1]) in (
            '/XYZ', '/Fit', '/FitH', '/FitV', '/FitR', '/FitB', '/FitBH', '/FitBV')
        if not (local_action or local_dest):
            raise ValueError('Unexpected PDF opening action')
    text = []
    for p in pdf.pages:
        text.append(p.extract_text())
        if '/AA' in p:
            raise ValueError('Unexpected PDF page action')
        for ref in p.get('/Annots', []):
            link = ref.get_object()
            action = link.get('/A', {})
            kind = action.get('/S')
            if ('/AA' in link or link.get('/Subtype') != '/Link'
                    or (kind not in (None, '/GoTo') and not (
                        kind == '/URI' and action.get('/URI') in citations))):
                raise ValueError('Unexpected PDF link or action')
    whole = '\n'.join(text)
    if archive.PRIVATE_MARKERS.search((whole+'\n'+str(pdf.metadata)).encode()):
        raise ValueError('Unexpected identity marker or visible external path in PDF')
    return whole


def checked(path, sha):
    raw = path.read_bytes()
    if main.digest(raw) != sha:
        raise ValueError(f'Artifact changed: {path.name}')
    return raw


def check_replay(raw):
    with ZipFile(BytesIO(raw)) as z:
        files = z.infolist()
        if len({f.filename for f in files}) != len(files) or not any(f.filename == 'verify.py' for f in files):
            raise ValueError('Invalid replay roster')
        for f in files:
            p = PurePosixPath(f.filename)
            if (p.is_absolute() or '..' in p.parts or str(p) != f.filename or f.is_dir()
                    or not f.filename.isascii() or '\\' in f.filename or stat.S_ISLNK(f.external_attr >> 16)
                    or p.suffix not in ('.md', '.txt', '.json', '.py')):
                raise ValueError('Unexpected replay archive member')
            data = z.read(f)
            if archive.PRIVATE_MARKERS.search(data):
                raise ValueError('Identity marker or external URL in replay')


def read_receipt(group):
    return json.loads((archive.PUBLIC/group/'receipt.json').read_text())


def members():
    paper = read_receipt('evidence_manuscript_v167_citation')
    compact = read_receipt('compact_review_supplement_v1')
    stats = read_receipt('reviewer_evidence_v1')
    moving = read_receipt('moving_peak_certificate_v1')
    draft = paper['artifact_sha256']['manuscript.pdf']
    if (not paper['llm_statement_unchanged'] or not paper['no_external_outcomes_opened']
            or compact['manuscript_sha256'] != paper['original_main_pdf_sha256']
            or compact['replay_archive_sha256'] != stats['archive_sha256']
            or not moving['offline_stdlib'] or moving['independent_confirmation']
            or not moving['original_2pct_unchanged']):
        raise ValueError('Reviewer files do not bind the unchanged evidence')
    text = check_pdf(checked(DRAFT/'manuscript.pdf', draft), 8)
    if not all(x in text for x in ('Anonymous CVPR submission', 'LLM Usage Statement', 'Tezcan')):
        raise ValueError('Expected the anonymous main PDF with the dataset citation and disclosure')
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items['reading_guide.pdf'] = checked(LOCAL/'compact_review_supplement_v1.pdf', compact['sha256'])
    items['technical_appendix.pdf'] = checked(TECHNICAL, TECHNICAL_SHA)
    check_pdf(items['reading_guide.pdf'], 5)
    check_pdf(items['technical_appendix.pdf'], 50, {MANTRA})
    for name, digest in (('reviewer_evidence.zip', stats['archive_sha256']),
                         ('moving_peak_certificate.zip', moving['archive_sha256'])):
        items[name] = checked(LOCAL/name, digest)
        check_replay(items[name])
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    return items, draft, stats['headline']


def build(destination, receipt):
    items, draft, headline = members()
    with TemporaryDirectory(prefix='combined-review-') as scratch:
        root = Path(scratch)
        a, b = root/'a.zip', root/'b.zip'
        archive.encode(a, items)
        archive.encode(b, items)
        raw = a.read_bytes()
        if raw != b.read_bytes():
            raise ValueError('Non-deterministic supplementary archive')
        extracted = root/'clean'
        with ZipFile(a) as z:
            z.extractall(extracted)
        run = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=extracted,
                             capture_output=True, text=True, check=True, timeout=150)
    replay = json.loads(run.stdout)
    if replay['headline'] != headline:
        raise ValueError('The combined replay changed a manuscript result')
    result = dict(sha256=main.digest(raw), bytes=len(raw), members=len(items),
                  main_pdf_sha256=draft, main_pdf_pages=8, supplement_pdf_pages=[5, 50],
                  headline=replay['headline'], deterministic=True, clean_stdlib_replay=True,
                  pdfs_unchanged=True, new_outcomes_opened=False, uploaded=False, submission_ready=False)
    for path, data in ((destination, raw), (receipt, main.as_json(result))):
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f'Refusing to replace {path.name}')
    for path, data in ((destination, raw), (receipt, main.as_json(result))):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as f:
                f.write(data)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=LOCAL/'review_supplement_v1.zip')
    p.add_argument('--receipt', type=Path, default=OUT/'verification.json')
    a = p.parse_args()
    build(a.output, a.receipt)
