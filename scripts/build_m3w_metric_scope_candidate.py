"""Build a separate review PDF that distinguishes the two utility scores."""

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

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_citation_candidate as citation
from scripts import package_m3w_combined_review_supplement as supplement

main = citation.main
ORIGINAL = citation.DEFAULT
DESTINATION = ORIGINAL.parent/'evidence_manuscript_v168_metric_scope'
RECEIPT = main.PUBLIC/'evidence_manuscript_v168_metric_scope/receipt.json'
SCOPE = {
    '''The observation is eight annotated image positions and the request is
twelve future annotation steps. Neighbor construction, motion features,
and every candidate forecast depend on the current and past positions.
Distances are measured in native image pixels.''': '''Each query uses eight observed positions and twelve future steps on
its dataset grid. Neighbors, features, and forecasts use only past
observations. Moving uses annotation pixels, WEPDTOF detection pixels,
and OpenDD release coordinates.''',
    '''Accuracy uses equal query weights and a common observed CV denominator.
For a comparison between two policies, subtract maximum disagreement
between those policies on wholly unobserved rows from the observed gain.
The resulting lower contrast keeps available-label errors unchanged
on partial rows. It is not a full-grid ADE bound for partial tracks or
a confidence interval. Source-level paired bootstrap intervals average
the three seeds before resampling 14 source groups 3,000 times.''': '''Accuracy contrasts weight queries equally and normalize by observed CV
error. Row scores debit maximum forecast disagreement on wholly
unobserved rows and keep available-step means on partial rows. The
Moving arbitrary-path audit and stepwise OpenDD and WEPDTOF contrasts
sum observed gains and debit disagreement at each missing step.
Only stepwise scoring bounds the full-grid numerator. The primary
Moving bootstrap averages seeds within each of 14 sources before
3,000 draws.''',
}


def revise(text):
    changed = text
    for old, new in SCOPE.items():
        if changed.count(old) != 1 or new in changed:
            raise ValueError('Expected the unchanged source paragraph')
        changed = changed.replace(old, new)
    restored = changed
    for old, new in reversed(tuple(SCOPE.items())):
        restored = restored.replace(new, old)
    if restored != text:
        raise ValueError('Unexpected modification outside the scope paragraphs')
    return changed


def check_render(raw, original):
    def without_line_numbers(body):
        # The review template appends three-digit line counters to extracted text.
        body = re.sub(r'(?m)[ \t]*[0-9]{3}$', '', body)
        return ' '.join(re.sub(r'(?<=[a-z])[-\u00ad][ \t]*\n[ \t]*(?=[a-z])', '', body).split())

    text = without_line_numbers(supplement.check_pdf(raw, 8))
    missing = [x for x in ('Anonymous CVPR submission', 'OpenDD release coordinates',
                           'each missing step', 'LLM Usage Statement') if x not in text]
    if missing:
        raise ValueError(f'The revised scope or disclosure did not render: {missing}')
    before = PdfReader(original)
    after = PdfReader(BytesIO(raw))
    a, b = (without_line_numbers('\n'.join(p.extract_text() for p in r.pages)).split('LLM Usage Statement', 1)[1]
            for r in (before, after))
    if a != b:
        raise ValueError('The rendered disclosure or bibliography changed')


def build(destination, compiler, receipt):
    prior = supplement.read_receipt('evidence_manuscript_v167_citation')
    expected = prior['artifact_sha256']
    inputs = {name: supplement.checked(ORIGINAL/name, sha) for name, sha in expected.items()}
    with ZipFile(ORIGINAL/'main_source.zip') as z:
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n for n in names):
            raise ValueError('Unexpected original source-archive member')
        members = {n: z.read(n) for n in names}
    if members['manuscript.tex'] != inputs['manuscript.tex']:
        raise ValueError('Source archive differs from the verified manuscript')
    text = members['manuscript.tex'].decode()
    members['manuscript.tex'] = revise(text).encode()
    if len(citation.used_citations(members)) != prior['bibliography_entries']:
        raise ValueError('The revised manuscript changed citations')
    style = supplement.checked(citation.main.DRAFT/'cvpr.sty', prior['style_sha256'])
    with TemporaryDirectory(prefix='metric-scope-') as tmp:
        p = Path(tmp)
        for n, raw in {**members, 'cvpr.sty': style}.items():
            (p/n).write_bytes(raw)
        run = subprocess.run([str(compiler), '-X', 'compile', '--untrusted', '--only-cached',
                              '--keep-logs', 'manuscript.tex'], cwd=p, capture_output=True, text=True,
                             timeout=120, env={**os.environ, 'SOURCE_DATE_EPOCH': '0'})
        if run.returncode:
            raise RuntimeError(run.stdout+'\n'+run.stderr)
        log = (p/'manuscript.log').read_text()
        if re.search(r'(?i)undefined (citation|reference)|Citation .* undefined|Overfull \\hbox', log):
            raise ValueError('Undefined reference or overfull horizontal box')
        citation.archive.encode(p/'main_source.zip', members)
        output = {k: (p/k).read_bytes() for k in ('manuscript.tex', 'manuscript.pdf', 'main_source.zip')}
    check_render(output['manuscript.pdf'], ORIGINAL/'manuscript.pdf')
    for k, raw in output.items():
        if (destination/k).exists() and (destination/k).read_bytes() != raw:
            raise FileExistsError(f'Refusing to replace another candidate: {k}')
    destination.mkdir(parents=True, exist_ok=True)
    for k, raw in output.items():
        if not (destination/k).exists():
            with (destination/k).open('xb') as f:
                f.write(raw)
    result = dict(pages=8, source_paragraphs_changed=len(SCOPE), only_changed_archive_member='manuscript.tex',
                  original_pdf_sha256=expected['manuscript.pdf'], original_source_zip_sha256=expected['main_source.zip'],
                  artifacts={k: main.digest(v) for k, v in output.items()},
                  abstract_tables_references_and_llm_statement_source_unchanged=True,
                  original_risk_rule_unchanged=True, new_outcomes_read=False, uploaded=False, submission_ready=False)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    raw = main.as_json(result)
    if receipt.exists() and receipt.read_bytes() != raw:
        raise FileExistsError('Existing candidate receipt differs')
    if not receipt.exists():
        receipt.write_bytes(raw)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output', type=Path, default=DESTINATION)
    p.add_argument('--receipt', type=Path, default=RECEIPT)
    a = p.parse_args()
    build(a.output, a.compiler.resolve(), a.receipt)
