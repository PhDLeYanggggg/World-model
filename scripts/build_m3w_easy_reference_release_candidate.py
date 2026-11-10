"""Build the easy-reference readout and a matching anonymous replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.easy_reference_release_supplement import verify
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from scripts import build_m3w_nested_swap_release_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

builder, archive, bundle, main = prior.builder, prior.archive, prior.bundle, prior.main
OUT = prior.OUT.parent/'evidence_manuscript_v199_easy_reference_release'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
STUDY = main.PUBLIC/'opendd_easy_reference_release_v1'
PACK = main.PUBLIC/'easy_reference_release_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v30.zip'
TEMPLATE = archive.ROOT/'reproducibility/easy_reference_release_supplement'


def inputs():
    keys = ('protocol.md', 'registration.json', 'freeze.json', 'calibration_freeze.json', 'readout.json')
    raw = {k: (STUDY/k).read_bytes() for k in keys}
    h = {k: main.digest(v) for k, v in raw.items()}
    f, c, r = (json.loads(raw[k]) for k in keys[2:])
    if (r['calibration'] != h['calibration_freeze.json'] or c['freeze'] != h['freeze.json']
            or not f['key'] == c['key'] == r['key'] == h['registration.json'] or not r['original_2pct']
            or r['external_outcomes_read'] or not r['development_only'] or r['deployment_changed']
            or len(c['rules']) != 126):
        raise ValueError('Inconsistent easy-reference study evidence')
    return (r, c), h


def revise(source, data):
    start = r'\paragraph{Calibrate on a different location.}'
    end = r'\section{What Transfers to Other Recordings?}'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('The nested release paragraph changed')
    r = data[0]['summary']
    s = r['sign_times_reference']
    phrase = f"{s['strict']}/{s['released']}"
    lines = rf'''\paragraph{{Calibration on a separate location.}}
Fitting, calibration and scoring use disjoint locations. Sign releases
{r['sign']['released']} views and qualifies {r['sign']['strict']}; magnitude releases none.
Ranking the same sign proposals by sign times predicted easy-reference
availability qualifies {phrase}, but worsens conservative gain.
The unmatched additive control passes risk and both references in 19/65.

'''
    return source[:source.index(start)]+lines+source[source.index(end):], (phrase, '19/65')


def package():
    (data, cal), hashes = inputs()
    parent, draft = bundle.read_receipt(prior.PACK.name), bundle.read_receipt(OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if draft['source_readout_sha256'] != hashes or draft['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Paper and review evidence do not bind the same inputs')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    s = data['summary']
    sites = sorted(s['sign']['effects']['lower_cv_pp']['by_site'])
    seeds = sorted({x['seed'] for x in cal['rules'].values()})
    directions = []
    for k, row in data['rows'].items():
        roles = [sites.index(row['held']), sites.index(row['calibration']), seeds.index(row['seed'])]
        arms = {}
        for a in verify.ARMS:
            out = {t: row['arms'][a][t] for t in (*verify.COUNTS, 'lower_cv_pp')}
            out['calibration_count'] = cal['rules'][k]['arms'][a]['calibration_recordings']
            arms[a] = out
        directions.append(dict(roles=roles, arms=arms))
    panel = dict(directions=directions, draws=np.random.default_rng(211).integers(7, size=(3000, 7)).tolist())
    counts = (*verify.COUNTS, 'calibration_active_directions')
    expected = dict(counts={a: {k: x[k] for k in counts} for a, x in s.items()},
                    cv={a: dict(point_pp=x['effects']['lower_cv_pp']['point'],
                                ci95_pp=x['effects']['lower_cv_pp']['ci95']) for a, x in s.items()},
                    paired=data['paired'])
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol.md'])
    parts['directions.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(p) for p in parts.values()):
        raise ValueError('Identifying review material')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected review archive contents')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='easy-reference-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['easy_reference_release'], expected):
            raise ValueError('Anonymous easy-reference evidence does not replay')
    row = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), prior_bundle_sha256=parent['archive_sha256'],
               companion_pdf_sha256=pdf, source_readout_sha256=hashes['readout.json'], standard_library_replay=True,
               includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for p, content in {DEST: raw, PACK/'receipt.json': main.as_json(row)}.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() != content:
            raise FileExistsError('Refusing to replace a frozen review artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(content)
    print(row)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    arg = p.parse_args()
    builder.build(OUT, arg.compiler.resolve(), RECEIPT, parent_dir=prior.OUT, read_inputs=inputs, revise_source=revise)
    package()
