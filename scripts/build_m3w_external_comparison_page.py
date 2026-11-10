"""Render a standalone comparison page from the checked external receipts."""

import json
from io import BytesIO
from pathlib import Path
import sys

from pypdf import PdfReader
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_external_budget_comparison as audit
from scripts import build_m3w_reader_supplement as reader
from scripts import build_m3w_serif_supplement as style

LOCAL = reader.ADMIN/'external_budget_comparison_v1'/'external_comparison.pdf'
RECEIPT = audit.OUT.with_name('render_receipt.json')
COHORTS = ('wep', 'songdo', 'athens')
NAMES = dict(wep='WEPDTOF', songdo='Songdo Traffic', athens='Athens pNEUMA')


def interval(values, digits=5):
    return '[' + ', '.join(f'{x:+.{digits}f}' for x in values) + ']'


def render(result):
    raw = BytesIO()
    pdf = canvas.Canvas(raw, pagesize=(612, 792), invariant=1)
    pdf.setAuthor('Anonymous')
    pdf.setTitle('External comparisons at matched intervention budgets')
    y = style.para(pdf, 'External comparisons at matched intervention budgets',
                   742, size=16, lead=20, bold=True, space=12)
    y = style.para(pdf, 'Eight observed and twelve forecast steps per cohort. '
                  'The original comparisons match edit counts; WEPDTOF and Songdo '
                  'also match peak departure from CV. The sensitivity additionally '
                  'matches departure at each step while retaining edit recipients '
                  'and directions. No cohort is pooled with another.', y)
    rows = result['cohorts']
    y = style.para(pdf, '<b>Joint versus independent selection</b>', y)
    y = style.table(pdf, [['Cohort', 'Source units used', 'Original envelope (pp)', 'Equal steps (pp)'],
        *[[NAMES[k], f"{rows[k]['units_resampled']}/{rows[k]['total_groups']}",
           interval(rows[k]['original']['envelope_pp']), interval(rows[k]['equal_steps']['envelope_pp'])]
          for k in COHORTS]], [112, 90, 149, 149], y)
    y = style.para(pdf, 'Envelopes vary the unobserved future positions in the scored sample. Positive '
                  'values favor joint selection; gains are normalized by the cohort\'s observed CV error. '
                  'The Songdo envelope favors independent selection in this sample, but the '
                  'source-bootstrap interval of the equal-step lower endpoint spans zero.', y)
    y = style.table(pdf, [['Cohort', 'Change in observed gain (pp)', '95% paired interval (pp)'],
        *[[NAMES[k], f"{rows[k]['equal_steps_minus_original_observed_pp']:+.6f}",
           interval(rows[k]['paired_source_ci95_pp'], 6)] for k in COHORTS]], [130, 167, 203], y)
    y = style.para(pdf, 'The change is equal steps minus original, evaluated only on observed positions. '
                  'The 3,000 paired draws resample original sources, source components or acquisition '
                  'dates, after averaging seeds. The Athens shift is small and is descriptive of four '
                  'dates in one city; it does not resolve the full-grid joint contrast.', y)
    y = style.para(pdf, '<b>Fixed proposals and actual release</b>', y)
    release = [f"{rows['wep']['joint_released_edits']} edits",
               f"{rows['songdo']['joint_released_views']}/{rows['songdo']['joint_fixed_dual'][1]} flight-seed views",
               'Not evaluated']
    y = style.table(pdf, [['Cohort', 'Fixed joint dual passes', 'Actual joint release'],
        *[[NAMES[k], '/'.join(map(str, rows[k]['joint_fixed_dual'])), v]
          for k, v in zip(COHORTS, release, strict=True)]], [130, 145, 225], y)
    y = style.para(pdf, 'The original dual check requires the 2% risk rules and positive conservative '
                  'utility against both CV and OLS4. In Songdo, the independent rule released '
                  f"{rows['songdo']['independent_released_views']} flight-seed views; "
                  f"{rows['songdo']['independent_released_dual']} passed both utility and risk. "
                  'Athens is a zero-shot proposal comparison with no Athens release calibration. '
                  'No edit does not constitute a risk pass.', y)
    y = style.para(pdf, 'Data source: pNEUMA &ndash; open-traffic.epfl.ch. '
                  'Barmpounakis and Geroliminis (2020), doi:10.1016/j.trc.2019.11.023; '
                  'release doi:10.5281/zenodo.10491409.', y, size=9.5, lead=12)
    style.finish(pdf, 1, y)
    pdf.save()
    content = raw.getvalue()
    pages = PdfReader(BytesIO(content)).pages
    extracted = ' '.join(pages[0].extract_text().split())
    if (len(pages) != 1 or pages[0].get('/Annots') or reader.PRIVATE.search(extracted)
            or any(x not in extracted for x in ('53/480', '0/45', '1/18', 'Not evaluated'))):
        raise ValueError('Unexpected comparison page layout or content')
    return content


def main():
    raw = {k: (audit.PUBLIC/p).read_bytes() for k, p in audit.SOURCES.items()}
    result = audit.replay(raw)
    if result != json.loads(audit.OUT.read_bytes()):
        raise ValueError('Public comparison did not reproduce')
    pdf = render(result)
    report = dict(readout_sha256=reader.digest(audit.OUT.read_bytes()),
                  pdf_sha256=reader.digest(pdf), pages=1, anonymity_text_check=True,
                  not_merged_into_author_draft=True, uploaded=False)
    for p, content in ((LOCAL, pdf), (RECEIPT, reader.as_json(report))):
        if p.exists() and p.read_bytes() != content:
            raise FileExistsError(p)
    for p, content in ((LOCAL, pdf), (RECEIPT, reader.as_json(report))):
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open('xb') as f:
                f.write(content)
    print(LOCAL)


if __name__ == '__main__':
    main()
