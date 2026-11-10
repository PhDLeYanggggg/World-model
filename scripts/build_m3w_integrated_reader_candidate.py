"""Clarify the manuscript contribution and consolidate its anonymous appendix."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_pair_label_stress_candidate as prior
from scripts import build_m3w_reader_supplement as reader
from scripts.build_m3w_full_grid_candidate import builder
from reproducibility.m3w_integrated_reader.verify import MEMBERS, compare, evidence

OUT = prior.OUT.with_name('evidence_manuscript_v216_integrated_reader')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
TEMPLATE = reader.TEMPLATE.with_name('m3w_integrated_reader')
HANDOFF = reader.ADMIN/'local_handoff_v216'

OLD = """We study replacement as a decision about predictions, not an
intervention on the agents. Given a bank of forecasts and a
motion fallback, we ask three questions. Does the budget leave
room for improvement at all? Can a rule that sees only the past
find the useful replacements? Does the rule still pass the same
risk checks on other recordings? A single average error does not
distinguish these failure modes.

Our evaluation compares independent and joint choices with
matched edit counts and, where specified, matched displacement.
We contrast realized decisions with label-aware feasible plans and
with a continuous displacement bound that does not depend on
the forecast bank. A closed-form test checks whether
missing positions can overturn a recording-level risk pass. We then test frozen rules on separate sources.
The results do not establish that joint selection is better. They
show where to look for improvement: on OpenDD, many feasible
plans go unused by a past-only rule; on MovingDroneCrowd++, the
displacement constraint and missing future labels can instead
leave little room under the conservative score."""

NEW = r"""We study the replacement of predictions, not interventions on
the agents. An average error reduction cannot tell us whether
a useful replacement existed, whether a past-only rule found it,
or whether the same rule worked on new recordings. We separate
these questions into three tests:
\emph{feasibility}, \emph{selection}, and \emph{frozen transfer}.

For feasibility, a displacement bound applies even to paths
outside the bank, and an exact prefix search asks what a rule
with access to the future could achieve inside the bank.
For selection, we match edit counts and, where specified,
displacement, so a joint rule cannot win by spending more.
For transfer, we score frozen proposals and actual releases
separately. A closed-form sensitivity test distinguishes risk
passes that survive missing future positions from those that
depend on the observed labels alone.

These tests point to different bottlenecks. The movement budget
can rule out useful replacements on MovingDroneCrowd++, while
many legal OpenDD plans go unused by the frozen scores. The
external comparisons do not establish a joint-selection advantage."""


def inputs():
    parent = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    paths = {
        'prior_replay': reader.ADMIN/'review_supplement_v49.zip',
        'old_reader': reader.ADMIN/'review_supplement_v44.pdf',
        'mask_figure': prior.OUT/'equal_coverage.pdf',
    }
    digests = {k: parent[f] for k, f in (('prior_replay', 'archive_sha256'),
        ('old_reader', 'reader_pdf_sha256'), ('mask_figure', 'figure_sha256'))}
    raw = {k: reader.checked(p, digests[k]) for k, p in paths.items()}
    with TemporaryDirectory(prefix='v216-parent-') as tmp:
        p = Path(tmp)/'parent.zip'
        p.write_bytes(raw['prior_replay'])
        reader.unpack(p, Path(tmp)/'unpacked')
        results = reader.run_replay(Path(tmp)/'unpacked')
    data = {k: evidence(results, field) for k, field in (
        ('uniform', 'opendd_fixed_spend_uniform'), ('count', 'opendd_full_grid_count_motion'),
        ('listwise', 'opendd_listwise_full_grid'), ('relative', 'opendd_backbone_relative'),
        ('masking', 'opendd_pair_label_stress'))}
    compare(data, results)
    configs = {k: reader.PUBLIC/f'opendd_vehicle_{k}_v1/registration.json' for k in ('eqmotion', 'stgcnn')}
    cfg = {k: json.loads(p.read_bytes()) for k, p in configs.items()}
    if any((c['batch'], c['checkpoint_every'], c['threads'], c['workers'], c['seeds']) !=
           (128, 100, 4, 0, [17, 29, 43]) for c in cfg.values()):
        raise ValueError('Training protocol differs from the reader recipe')
    if cfg['eqmotion']['updates'] != 400 or cfg['stgcnn']['steps'] != 400:
        raise ValueError('Expected both equal-update backbone controls')
    digests.update({f'{k}_registration': reader.digest(p.read_bytes()) for k, p in configs.items()})
    realized = (reader.PUBLIC/'opendd_realized_stream_capacity_v1/readout.json').read_bytes()
    original = json.loads(realized)
    fields = ('views', 'edits', 'broad_pass', 'exact_pass', 'online_pass')
    if (not original['original_2pct'] or original['confirmation_access']
            or tuple(original['summary'][k] for k in fields) != (606, 780, 330, 242, 57)):
        raise ValueError('The original broader-budget study differs from the manuscript')
    digests['realized_stream'] = reader.digest(realized)
    return dict(values=data, raw=raw, parent=parent), digests


def revise(text, _):
    if text.count(OLD) != 1:
        raise ValueError('Expected v215 introduction')
    return text.replace(OLD, NEW, 1), ('feasibility', 'frozen transfer', '3,711 complete queries')


def extra_pages(data):
    out = BytesIO()
    pdf = canvas.Canvas(out, pagesize=(612, 792), invariant=1)
    pdf.setAuthor('Anonymous')
    layout = reader.layout
    header, para, table, finish = layout.header, layout.para, layout.table, reader.finish

    y = header(pdf, 'REPRODUCTION', 'From a history query to a scored decision')
    y = para(pdf, '<b>1. Freeze the roster.</b> Group agents at the same last-observed step. Keep the '
             'eight past positions, the twelve requested steps, and the validity mask separate. '
             'Past-only prediction and proposal construction do not use the mask. In the OpenDD '
             'backbone study, eligible vehicle queries are spaced by twelve release ticks on the '
             'forecast lattice. Use the complete history-eligible roster; only the evaluation '
             'checks where target positions exist.', y)
    y = para(pdf, '<b>2. Fit on donor locations.</b> For each OpenDD location and each seed '
             '(17, 29, 43), train on the other six locations and use the fixed last checkpoint. '
             'The matched OpenDD backbone study has 500 recordings, 25,596 queries and 141,783 '
             'agent histories. This is distinct from the fixed-spend study below, with 202 '
             'recordings repeated across three seeds. Neither study is a new independent test.', y)
    y = table(pdf, [['Forecast', 'Deterministic architecture', 'Neighbor input'],
                    ['Completion-trained', '17 features; 64/48-unit MLP', 'None'],
                    ['EqMotion', 'Two layers, 32 channels; one fixed head', 'None / nearest three'],
                    ['Social-STGCNN', 'One graph layer, five temporal layers; kernel 3', 'None / nearest three']],
              [124, 224, 152], y)
    y = para(pdf, 'Backbones use a comoving history frame and a CV-relative peak bound of two '
             'observed RMS displacements. The completion objective adds forecast-to-CV distance '
             'at each missing step to the error at observed steps, then divides by twelve. '
             'The graph controls use 400 AdamW updates at learning rate 0.001, batch 128, '
             'uniform donor-site then within-site sampling. Native CPU runs use four compute '
             'threads and no data workers; checkpoints every 100 updates retain optimizer and '
             'sampler state. These are matched deterministic adaptations, not the original '
             'multimodal benchmark configurations.', y)
    y = para(pdf, '<b>3. Match proposals, then score release.</b> Use the same bank and query '
             'budget for independent, same-family, additive and joint proposals. CV is the '
             'fallback and OLS4 the second reference. Select thresholds only on the allowed '
             'calibration sources. Record complete-label support, observed accuracy, the '
             'direct missing-label envelope and the original three risk ratios separately. '
             'An unsupported ratio remains undefined; no-edit CV is not a risk pass.', y)
    y = para(pdf, '<b>4. Aggregate by source.</b> Average seeds within the recording or member '
             'before aggregating to the source level. The paired location intervals use 3,000 '
             'resamples of seven OpenDD locations. Do not pool coordinate units across datasets.', y)
    finish(pdf, 5, y)

    count, uniform, listwise = (data[k] for k in ('count', 'uniform', 'listwise'))
    y = header(pdf, 'OPEN DEVELOPMENT', 'Allocation at the same spend')
    y = para(pdf, 'The 606 views repeat 202 recordings over three seeds. Hold the total of '
             '780 edits fixed, using each recording-seed view\'s original count and the '
             'cap on summed peak departures from CV. At each eligible query choose a '
             'legal prefix of zero, one or two edits. A chronological budget earns '
             'one credit per eligible query, stores up to ten, and charges five per edit.', y)
    y = para(pdf, 'The abstract reports the broader quota-only comparison: 330 hindsight '
             'passes versus 57 online. Requiring the exact realized count in each view '
             'leaves 242 hindsight passes before the movement cap. The table below '
             'adds both the movement cap and the complete-grid risk test.', y)
    y = table(pdf, [['Allocation', 'Full-grid dual passes / 606'],
                    ['Frozen scores', count['blind_full_grid']],
                    ['Uniform legal prefix (expected)', f"{uniform['expected_passes_low']:.2f}"],
                    ['Hindsight: keep or one exchange', count['one_exchange_full_grid']],
                    ['Hindsight: any legal allocation', count['full_grid']]],
              [315, 185], y)
    y = para(pdf, 'The hindsight search requires the original risk rule, both conservative '
             'reference gains, the full-grid excess limits and complete easy support. '
             'It finds 212 passes under the original rule and 210 with the additional '
             'complete-grid test. Uniform allocation counts suffix completions by '
             'remaining credits and edits, then enumerates or samples ranks and rejects '
             'motion-cap violations. This comparator sees whole-recording eligibility, '
             'not outcomes. It is not an online release policy.', y)
    y = table(pdf, [['Ranker on the same 101 intervention views', 'Full-grid passes / 606'],
                    ['Pointwise, original target', f"{listwise['point_original']:.0f}"],
                    ['Listwise, original target', f"{listwise['list_original']:.0f}"],
                    ['Listwise, full-grid target', f"{listwise['list_full_grid']:.0f}"],
                    ['Uniform within the same intervention mask', f"{listwise['uniform']:.2f}"]],
              [315, 185], y)
    y = para(pdf, 'The last four rows use the same intervention mask and legal alternatives; '
             'the uniform row here is different from uniform allocation across the whole '
             'recording above. The trained rankers share 400 updates and 32 sampled donor '
             'slates per step. The stricter training target shows no reliable advantage over '
             'the original-target listwise ranker or the same-mask uniform expectation.', y)
    finish(pdf, 6, y)

    rel = data['relative']
    y = header(pdf, 'PAIR TRAJECTORIES', 'Neighbor input and label timing')
    y = para(pdf, f"All within-query pairs are defined before checking label availability. "
             f"{rel['support']['observed_pair_steps']:,} of {rel['support']['pair_steps']:,} "
             'pair-steps have both targets. Relative error uses the distance between '
             'predicted and labeled relative-position vectors. It is not a collision metric.', y)
    names = {'cv': 'CV', 'ols4': 'OLS4', 'completion': 'Completion-trained', 'comoving_ego': 'EqMotion, ego',
             'comoving_peers': 'EqMotion, neighbors', 'social_ego': 'Social-STGCNN, ego',
             'social_peers': 'Social-STGCNN, neighbors'}
    rows = [[name, *(f"{rel['absolute'][k][f]['point']:.3f}" for f in
             ('relative_ade', 'relative_fde', 'query_relative_ade_p95'))] for k, name in names.items()]
    y = table(pdf, [['Forecast', 'Relative ADE', 'Relative FDE', 'Relative ADE p95'], *rows],
              [191, 103, 103, 103], y)
    rows = []
    for k, name in (('eq_neighbors_minus_ego', 'EqMotion: neighbors vs ego'),
                    ('social_neighbors_minus_ego', 'Social-STGCNN: neighbors vs ego'),
                    ('completion_minus_cv', 'Completion-trained vs CV')):
        v = rel['paired'][k]
        point, (lo, hi) = v['observed']['point'], v['observed']['ci95']
        rows.append([name, f'{point:+.3f} [{lo:+.3f}, {hi:+.3f}]',
                     f"[{v['lower']['point']:+.3f}, {v['upper']['point']:+.3f}]"])
    y = table(pdf, [['Paired gain', 'Observed [95% interval]', 'Full-grid envelope'], *rows],
              [193, 162, 145], y)
    y = para(pdf, 'The next page hides equally many early or late steps of the same 3,711 '
             'complete queries. The diamond shows the original complete-label gain. The '
             'controlled dropout explains why a single coverage percentage cannot establish '
             'the direction of the effect; it does not estimate natural missingness.', y)
    finish(pdf, 7, y)
    pdf.save()
    return out.getvalue()


def combine(data, assets):
    writer = PdfWriter()
    for raw in (assets['old_reader'], extra_pages(data), assets['mask_figure']):
        for page in PdfReader(BytesIO(raw)).pages:
            if page.get('/Annots'):
                raise ValueError('Unexpected interactive annotation in the appendix')
            writer.add_page(page)
    writer.add_metadata({'/Title': 'Supplementary material', '/Author': 'Anonymous'})
    out = BytesIO()
    writer.write(out)
    raw = out.getvalue()
    pages = PdfReader(BytesIO(raw)).pages
    text = ' '.join(' '.join(p.extract_text().split()) for p in pages)
    markers = ('From a history query to a scored decision', '49.29', 'Hindsight: any legal allocation',
               'Pair trajectories', '5.214', 'Equal coverage, different missing steps')
    if len(pages) != 8 or reader.PRIVATE.search(text) or any(m.lower() not in text.lower() for m in markers):
        raise ValueError('Unexpected anonymous appendix text or length')
    return raw


def package(bundle, hashes):
    data, raw, parent = (bundle[k] for k in ('values', 'raw', 'parent'))
    main = json.loads((PUBLIC/'receipt.json').read_bytes())
    pdf = main['artifacts']['manuscript.pdf']
    if main['source_readout_sha256'] != hashes or main['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('The matched reader uses a different main PDF')
    manuscript = reader.checked(OUT/'manuscript.pdf', pdf)
    appendix = combine(data, raw)
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': appendix, 'reader_values.json': reader.as_json(data),
                  'prior_evidence.zip': raw['prior_replay']})
    if set(parts) != MEMBERS or reader.PRIVATE.search(parts['reader_values.json'].decode()):
        raise ValueError('Unexpected identifying reviewer archive contents')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='v216-replay-') as tmp:
        p = Path(tmp)/'review.zip'
        p.write_bytes(payload)
        reader.unpack(p, Path(tmp)/'replay')
        row = reader.run_replay(Path(tmp)/'replay')['integrated_reader']
    files = {'paper.pdf': manuscript, 'supplement.pdf': appendix, 'replay.zip': payload}
    manifest = {k: reader.digest(v) for k, v in files.items()}
    receipt = dict(companion_pdf_sha256=pdf, reader_pdf_sha256=manifest['supplement.pdf'], reader_pages=8,
                   archive_sha256=manifest['replay.zip'], archive_bytes=len(payload),
                   prior_archive_sha256=parent['archive_sha256'], evidence_replayed=row, source_hashes=hashes,
                   unchanged_external_results=True, original_2pct=True, uploaded=False, submission_ready=False)
    targets = {**{HANDOFF/k: v for k, v in files.items()}, HANDOFF/'manifest.json': reader.as_json(manifest),
               PUBLIC/'review_receipt.json': reader.as_json(receipt)}
    if any(k.exists() and k.read_bytes() != v for k, v in targets.items()):
        raise FileExistsError('Refusing to replace a different reader artifact')
    for k, v in targets.items():
        k.parent.mkdir(parents=True, exist_ok=True)
        if not k.exists():
            with k.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    args = p.parse_args()
    bundle, hashes = inputs()
    builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                  read_inputs=lambda: (bundle, hashes), revise_source=revise)
    package(bundle, hashes)
