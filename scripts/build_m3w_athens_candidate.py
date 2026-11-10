"""Integrate the frozen Athens result without modifying the author's editing copy."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_pneuma_date_robustness as dates
from scripts import audit_m3w_external_budget_comparison as comparison
from scripts import build_m3w_external_comparison_page as comparison_page
from scripts import build_m3w_reader_supplement as reader
from scripts import build_m3w_serif_supplement as style
from scripts import build_m3w_vehicle_direction_candidate as builder
from reproducibility.m3w_athens_replay.verify import ATHENS, MEMBERS, replay

PARENT = builder.OUT.with_name('evidence_manuscript_v218_wepdtof_penalty')
OUT = PARENT.with_name('evidence_manuscript_v219_athens_confirmation')
PUBLIC = reader.PUBLIC/OUT.name
HANDOFF = reader.ADMIN/'local_handoff_v219_athens'
PREVIOUS_HANDOFF = reader.ADMIN/'local_handoff_v218_color_figures'
TEMPLATE = reader.TEMPLATE.with_name('m3w_athens_replay')


def inputs():
    raw = {k: (dates.SOURCE/f'{k}.json').read_bytes() for k in ATHENS}
    for k in ('registration', 'freeze'):
        path = (dates.SOURCE/f'{k}.json').relative_to(dates.ROOT)
        if subprocess.check_output(['git', 'show', f'{dates.REGISTRATION_COMMIT}:{path}'],
                                   cwd=dates.ROOT) != raw[k]:
            raise ValueError('Published Athens freeze changed')
    checked_dates = dates.replay(raw)
    if checked_dates != json.loads(dates.OUT.read_bytes()):
        raise ValueError('Athens date-level report did not reproduce')
    comp = comparison.replay({k: (comparison.PUBLIC/p).read_bytes() for k, p in comparison.SOURCES.items()})
    raw['comparison'] = comparison.OUT.read_bytes()
    if comp != json.loads(raw['comparison']):
        raise ValueError('Cross-cohort comparison did not reproduce')
    checked = replay(raw)
    matched = (reader.PUBLIC/'pneuma_prefix_pilot_v1/matched_receipt.json').read_bytes()
    if reader.digest(matched) != json.loads(raw['registration'])['data_binding']['matched_receipt.json']:
        raise ValueError('Matched query roster differs from registration')
    return dict(raw=raw, checked=checked, primary=json.loads(raw['confirmation']), comparison=comp,
                query_counts=json.loads(matched)), {k: reader.digest(v) for k, v in raw.items()}


def revise(text, d):
    data_anchor = ('acquisition day and city. We do not pool results across these\n'
                   'coordinate systems or treat the calibration cohorts as tests.')
    conclusion = ('effect must be reported separately. On the two frozen external\n'
                  'evaluations, neither joint rule establishes a reliable advantage\n'
                  'that survives the original recording checks.')
    anchors = (data_anchor, conclusion, '\\section{Conclusion and Limitations}', '\\end{thebibliography}')
    if any(text.count(a) != 1 for a in anchors):
        raise ValueError('Expected unchanged v218 manuscript anchors')
    r = d['checked']['athens']
    cv, independent, steps = (r[k]['mean'] for k in ('joint_vs_cv', 'joint_vs_independent', 'equal_steps'))
    ci = d['primary']['matched']['by_arm']['joint']['vs_reference']['cv']['lower_pp']['ci95']
    n = d['query_counts']['queries_per_seed']
    if n != 643 or d['query_counts']['query_seed_views'] != 1929:
        raise ValueError('Unexpected Athens query roster')
    paragraph = rf'''\paragraph{{Athens: a gain over CV need not favor joint selection.}}
On {n} pNEUMA queries, the joint proposal's missing-position envelope
against CV is [{cv[0]:+.5f}, {cv[2]:+.5f}] pp; the 95\% date interval
for its lower endpoint is [{ci[0]:+.5f}, {ci[1]:+.5f}] pp.
Joint and same-family replacement both have positive lower bounds
on all four dates. Against independent selection, however, joint
gives [{independent[0]:+.5f}, {independent[2]:+.5f}] pp. Matching departure from CV at every
forecast step still leaves [{steps[0]:+.5f}, {steps[2]:+.5f}] pp. Thus the
gain over CV does not isolate an advantage from joint selection.
Only 1/18 fixed member--seed proposals passes the original risk
and both utility checks; no Athens release rule was calibrated
or tested. Intervals describe four dates from one city, not
three times as many independent samples from the repeated seeds.

'''
    text = text.replace('The external evidence has three roles.', 'The external cohorts have distinct roles.', 1)
    text = text.replace(data_anchor, 'acquisition day and city. Athens pNEUMA~\\cite{pneuma,pneuma_data}\n'
        'contributes six members from four acquisition dates, all reserved\n'
        'for a frozen zero-shot comparison without Athens calibration.\n'
        'We do not pool these coordinate systems or treat calibration cohorts as tests.', 1)
    text = text.replace('\\section{Conclusion and Limitations}', paragraph+'\\section{Conclusion and Limitations}', 1)
    text = text.replace(conclusion, 'effect must be reported separately. Athens adds a positive\n'
        'CV comparison without establishing joint-selection gain. Across\n'
        'the three external comparisons, the extra value of joint selection\n'
        'remains unproven; the two calibrated joint rules do not release edits.', 1)
    text = text.replace("Songdo's independent partition does not establish city-to-city\ngeneralization.",
        "Songdo's partition and Athens' four-date comparison do not\nestablish city-to-city generalization.", 1)
    citations = r'''\bibitem{pneuma}
E. Barmpounakis and N. Geroliminis.
On the new era of urban traffic monitoring with massive drone data:
The pNEUMA large-scale field experiment.
\emph{Transportation Research Part C: Emerging Technologies},
111:50--71, 2020. doi:10.1016/j.trc.2019.11.023.

\bibitem{pneuma_data}
E. Barmpounakis and N. Geroliminis. pNEUMA dataset.
Zenodo, version 1, 2024. doi:10.5281/zenodo.10491409.

'''
    text = text.replace('\\end{thebibliography}', citations+'\\end{thebibliography}', 1)
    return text, (f'{cv[0]:+.5f}', f'{independent[0]:+.5f}', f'{steps[0]:+.5f}', '1/18', 'pNEUMA')


def athens_page(d):
    raw = BytesIO()
    pdf = canvas.Canvas(raw, pagesize=(612, 792), invariant=1)
    pdf.setAuthor('Anonymous')
    y = style.para(pdf, 'S10. Athens: four-date zero-shot evaluation', 742,
                   size=16.3, lead=20, bold=True, space=12)
    y = style.para(pdf, 'All six members from four acquisition dates were reserved for confirmation '
        'before real future positions were opened. The donor rule and 8/12 native-step roster were '
        'frozen. Athens supplies no calibration examples or release threshold. The 643 unique queries '
        'are repeated across three seeds, giving 1,929 query-seed views, not 1,929 independent samples.', y)
    primary = d['primary']['matched']
    contrasts = {'CV': primary['by_arm']['joint']['vs_reference']['cv'],
        'Independent': primary['joint_minus']['independent'],
        'Additive actor head': primary['joint_minus']['additive'],
        'Same family': primary['joint_minus']['same_family']}
    rows = []
    for name, m in contrasts.items():
        rows.append([name, f"{m['observed_pp']['point']:+.5f}",
            comparison_page.interval([m['lower_pp']['point'], m['upper_pp']['point']]),
            comparison_page.interval(m['lower_pp']['ci95'])])
    y = style.table(pdf, [['Joint versus', 'Observed (pp)', 'Sample envelope (pp)', 'Lower endpoint 95% CI'], *rows],
                    [107, 83, 148, 162], y)
    y = style.para(pdf, 'The envelope varies missing future positions within the scored sample; it is '
        'not a confidence interval. The interval resamples four acquisition dates 3,000 times after '
        'seed and member averaging. Joint and same-family proposals each have a positive lower '
        'bound against CV on all four dates. Joint versus independent or additive selection remains '
        'unresolved on every date. The joint gain is therefore not unique to joint selection.', y)
    rows = []
    r = d['checked']['athens']
    for day in r['joint_vs_cv']['by_date']:
        rows.append([f'{day[:4]}-{day[4:6]}-{day[6:]}',
            f"{r['joint_vs_cv']['by_date'][day][0]:+.5f}",
            f"{r['same_family_vs_cv']['by_date'][day][0]:+.5f}",
            comparison_page.interval([r['equal_steps']['by_date'][day][i] for i in (0, 2)])])
    y = style.table(pdf, [['Date', 'Joint vs CV lower', 'Same family vs CV lower', 'Joint vs independent, equal steps'],
                         *rows], [83, 110, 123, 184], y)
    y = style.para(pdf, '<b>Utility is not release.</b> The joint proposal passes the original risk '
        'and dual-reference utility checks in 1/18 member-seed views. This is a fixed proposal '
        'diagnostic, not a deployment evaluation. No Athens threshold is fitted, and the 2% rule '
        'is unchanged. Native release-coordinate errors are not physical-safety measurements.', y)
    y = style.para(pdf, '<b>Data acknowledgment.</b> Data source: pNEUMA &ndash; open-traffic.epfl.ch. '
        'E. Barmpounakis and N. Geroliminis (2020), Transportation Research Part C 111:50-71, '
        'doi:10.1016/j.trc.2019.11.023. Dataset version 1 (2024), doi:10.5281/zenodo.10491409.',
        y, size=9.5, lead=12)
    style.finish(pdf, 10, y)
    pdf.save()
    return raw.getvalue()


def package():
    d, hashes = inputs()
    parent = json.loads((reader.PUBLIC/'serif_supplement_v218_color_figures/receipt.json').read_bytes())
    main = json.loads((PUBLIC/'receipt.json').read_bytes())
    if main['source_readout_sha256'] != hashes:
        raise ValueError('Manuscript and supplement use different evidence')
    old_zip = reader.checked(PREVIOUS_HANDOFF/'replay.zip', parent['files']['replay.zip'])
    previous_path = PREVIOUS_HANDOFF/'supplement.pdf'
    if previous_path.exists():
        previous = reader.checked(previous_path, parent['files']['supplement.pdf'])
    else:
        with ZipFile(BytesIO(old_zip)) as z:
            previous = z.read('supplement.pdf')
        if reader.digest(previous) != parent['files']['supplement.pdf']:
            raise ValueError('Archived supplementary PDF differs from the verified handoff')
    paper = reader.checked(OUT/'manuscript.pdf', main['artifacts']['manuscript.pdf'])
    writer = PdfWriter()
    for source in (previous, athens_page(d), comparison_page.render(d['comparison'], page_number=11, section='S11')):
        for page in PdfReader(BytesIO(source)).pages:
            if page.get('/Annots'):
                raise ValueError('Unexpected supplementary annotation')
            writer.add_page(page)
    writer.add_metadata({'/Title': 'Supplementary material', '/Author': 'Anonymous'})
    stream = BytesIO()
    writer.write(stream)
    appendix = stream.getvalue()
    pdf = PdfReader(BytesIO(appendix))
    text = ' '.join(p.extract_text() for p in pdf.pages)
    if (len(pdf.pages) != 11 or reader.PRIVATE.search(text)
            or 'S10.' not in text or 'S11.' not in text or '+0.02383' not in text):
        raise ValueError('New appendix failed its text/page checks')
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({f'{k}.json': v for k, v in d['raw'].items()})
    parts.update({'supplement.pdf': appendix, 'prior_evidence.zip': old_zip})
    if set(parts) != MEMBERS or any(reader.PRIVATE.search(v.decode()) for k, v in parts.items() if k.endswith('.json')):
        raise ValueError('Unexpected anonymous archive content')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    archive = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='athens-replay-') as tmp:
        path = Path(tmp)/'replay.zip'
        path.write_bytes(archive)
        reader.unpack(path, Path(tmp)/'extracted')
        verification = reader.run_replay(Path(tmp)/'extracted')
        if {k: v for k, v in verification.items() if k != 'inherited'} != d['checked']:
            raise ValueError('Packaged replay differs from manuscript inputs')
    files = {'paper.pdf': paper, 'supplement.pdf': appendix, 'replay.zip': archive}
    manifest = {k: reader.digest(v) for k, v in files.items()}
    receipt = dict(files=manifest, pages={'paper': main['pages'], 'supplement': 11},
        previous_handoff_untouched=True, word_untouched=True, disclosure_unchanged=True,
        original_plot_colors_preserved=True, source_readout_sha256=hashes,
        replay=verification['athens'], archive_bytes=len(archive),
        new_training=False, new_target_access=False, original_2pct=True,
        uploaded=False, submission_ready=False)
    targets = {**{HANDOFF/k: v for k, v in files.items()}, HANDOFF/'manifest.json': reader.as_json(manifest),
               PUBLIC/'review_receipt.json': reader.as_json(receipt)}
    if any(k.exists() and k.read_bytes() != v for k, v in targets.items()):
        raise FileExistsError('Refusing to replace a different candidate package')
    for p, content in targets.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open('xb') as f:
                f.write(content)
    print(json.dumps({'files': manifest, 'pages': receipt['pages'], 'archive_bytes': len(archive)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True, type=Path)
    args = parser.parse_args()
    builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=PARENT,
                  read_inputs=inputs, revise_source=revise, added_citations=('pneuma', 'pneuma_data'))
    package()
