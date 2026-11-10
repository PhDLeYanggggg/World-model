"""Build the anonymous recipient-eligibility draft and scalar replay."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.recipient_eligibility_supplement import verify
from scripts import build_m3w_songdo_fixed_paths_candidate as previous
from scripts import build_m3w_songdo_fixed_paths_supplement as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

builder = previous.builder
bundle, archive, main = prior.bundle, prior.archive, prior.main
OUT = previous.OUT.parent/'evidence_manuscript_v194_recipient_eligibility'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
STUDY = main.PUBLIC/'opendd_recipient_eligibility_v1'
PACK = main.PUBLIC/'recipient_eligibility_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v25.zip'
TEMPLATE = archive.ROOT/'reproducibility/recipient_eligibility_supplement'


def inputs():
    raw = {k: (STUDY/k).read_bytes() for k in ('registration.json', 'freeze.json', 'readout.json', 'protocol.md')}
    hashes = {k: main.digest(v) for k, v in raw.items()}
    f, d = (json.loads(raw[k]) for k in ('freeze.json', 'readout.json'))
    if (d['freeze'] != hashes['freeze.json'] or f['key'] != d['key'] or d['key'] != hashes['registration.json']
            or d['external_outcomes_read'] or d['independent_confirmation'] or d['deployment_changed']
            or not d['original_2pct'] or not f['predictions_frozen']):
        raise ValueError('Expected the scored frozen development comparison')
    return d, hashes


def revise(source, data):
    marker = r'\section{What Transfers to Other Recordings?}'
    if source.count(marker) != 1:
        raise ValueError('Missing external section')
    s = data['summary']
    n = s['counts']
    if n['gain_swaps'] != n['archived_swaps'] or n['eligibility_swaps'] != n['gain_swaps']:
        raise ValueError('Expected exact matched edits')
    effect = s['paired']['eligibility_minus_gain']['lower_pp']
    lower, upper = effect['ci95']
    t = rf'''\paragraph{{Changing the recipient at the same budget.}}
On OpenDD, a swap moves one of two edits to a different vehicle in
its heading frame without changing the displacement at any step.
Only {n['opportunity_on_mask']} of the {n['opportunity_all']:,} views with a qualifying
hindsight improvement are in the frozen {n['gain_swaps']}-swap mask.
Within that exact mask, ranking by predicted gain finds
{n['gain_strict']} qualifying improvements; prioritizing a separate
prediction of risk and reference eligibility finds {n['eligibility_strict']}.
The latter loses \({abs(effect['point']):.6f}\) pp of conservative trajectory
gain (95\% location interval [{lower:+.6f}, {upper:+.6f}]).
The guard retains only {n['guarded_swaps']} swaps. Missing opportunities before
ranking, and ranking the remaining choices, are distinct failures.

'''
    return source.replace(marker, t+marker, 1), (f"{n['opportunity_all']:,}", f"{n['gain_swaps']}-swap", f"{abs(effect['point']):.6f}")


def package():
    d, hashes = inputs()
    prior_pack = bundle.read_receipt(prior.OUT.name)
    paper = bundle.read_receipt(OUT.name)
    pdf = paper['artifacts']['manuscript.pdf']
    if paper['source_readout_sha256'] != hashes or paper['prior_manuscript_sha256'] != prior_pack['companion_pdf_sha256']:
        raise ValueError('Draft and replay use different source evidence')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    s = d['summary']
    sites = sorted(next(iter(s['paired'].values()))['lower_pp']['by_site'])
    counts = [dict(sum((Counter(r['counts']) for r in d['by_direction'] if r['identity']['held'] == site),
                       Counter())) for site in sites]
    panel = dict(counts_by_site=counts, draws=np.random.default_rng(196).integers(len(sites), size=(3000, len(sites))).tolist(),
                 location_means={c: {k: [v['by_site'][x] for x in sites] for k, v in m.items()}
                                 for c, m in s['paired'].items()})
    expected = dict(counts=s['counts'], paired={c: {k: {x: v[x] for x in ('point', 'ci95')} for k, v in m.items()}
                                               for c, m in s['paired'].items()})
    # Include explicit zeros for the rare guarded cases at each site.
    panel['counts_by_site'] = [{k: r.get(k, 0) for k in expected['counts']} for r in counts]
    expected['counts']['guarded_strict'] = expected['counts'].get('guarded_strict', 0)
    for r in panel['counts_by_site']:
        r.setdefault('guarded_strict', 0)
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol.md'])
    parts['eligibility.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(p) for p in parts.values()):
        raise ValueError('Identifying material in the new panel')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, prior_pack['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected archive roster')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='recipient-eligibility-review-') as tmp:
        a = Path(tmp)
        for name in ('first.zip', 'second.zip'):
            archive.encode(a/name, parts)
        data = (a/'first.zip').read_bytes()
        if data != (a/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review package')
        unpack(a/'first.zip', a/'unpacked')
        if not check(run_replay(a/'unpacked')['recipient_eligibility'], expected):
            raise ValueError('Grouped eligibility readout does not replay')
    row = dict(archive_sha256=main.digest(data), archive_bytes=len(data), prior_bundle_sha256=prior_pack['archive_sha256'],
               companion_pdf_sha256=pdf, source_readout_sha256=hashes['readout.json'], standard_library_replay=True,
               includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for path, content in {DEST: data, PACK/'receipt.json': main.as_json(row)}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError('Refusing to replace a different archive or receipt')
        if not path.exists():
            with path.open('xb') as f:
                f.write(content)
    print(row)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    arg = p.parse_args()
    builder.build(OUT, arg.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
    package()
