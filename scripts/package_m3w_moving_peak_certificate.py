"""Export an anonymous, solver-free certificate for the frozen Moving bound."""

import argparse
from math import isclose
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.moving_peak_certificate import verify
from scripts import audit_m3w_record_peak_allocation as audit
from scripts import package_m3w_reviewer_evidence as archive

np, old = audit.np, audit.old
OUT = audit.OUT.parent/'moving_peak_certificate_v1'
TEMPLATE = archive.ROOT/'reproducibility/moving_peak_certificate'
DEFAULT = archive.LOCAL/'moving_peak_certificate.zip'


def aggregate(cv, ols, target, valid, query, radii):
    ids, group, sizes = np.unique(query, return_inverse=True, return_counts=True)
    if len(ids) != len(radii) or cv.shape[1] != verify.HORIZON:
        raise ValueError('Query budgets do not align to the original 12-step roster')
    y = np.where(valid[..., None], target, 0.)
    c, o, d = (np.linalg.norm(a-b, axis=-1) for a, b in ((cv, y), (ols, y), (cv, ols)))
    w = 1./sizes[group]
    error = np.where(valid, c, 0.)
    censored = np.where(valid, 0., d)
    lengths = lambda caps: np.diff(np.sort(caps, axis=1), axis=1, prepend=0.)
    left, right = lengths(error), lengths(error+censored)
    rows = []
    for k, (n, b) in enumerate(zip(sizes, radii, strict=True)):
        at = group == k
        rows.append(dict(actors=int(n), budget=float(b), available=left[at].sum(0).tolist(),
                         charged=right[at].sum(0).tolist()))
    # Retain the query partition but not its order, frame or identifiers.
    rows.sort(key=lambda q: (q['actors'], q['budget'], q['available'], q['charged']))
    return dict(queries=rows, cv_error_mass=float(w @ error.sum(1)),
                known_ols4_advantage=float(w @ np.where(valid, c-o, 0.).sum(1)),
                missing_disagreement=float(w @ censored.sum(1)))


def export(*, aggregate_view=aggregate):
    reg, cells, _ = audit.register()
    files = {k: audit.OUT/k for k in ('readout.json', 'break_even.json')}
    parent, boundary = (old.read(p) for p in files.values())
    parent_sha = old.sha(files['readout.json'])
    if (parent['registration'] != reg or boundary['parent']['sha256'] != parent_sha
            or verify.POSITIVE_PP != audit.model.POSITIVE_PP or any(
                not p['original_2pct'] or not p['hindsight_only'] or p['a_labels_opened']
                or p['confirmation_access'] or p['independent_confirmation'] for p in (parent, boundary))):
        raise ValueError('Require the original exposed allocation and break-even study')
    lookup = lambda r: (r['identity']['fold'], r['identity']['seed'], r['identity']['recording'])
    original = {lookup(r): r for r in parent['rows']}
    reference = {lookup(r): r for r in boundary['rows']}
    base = audit.previous.base
    data = base.fit_population(lambda **_: None)
    linear = audit.previous.prior.study.motion.bank(data['history'])[:, :2]
    rows, ids, sources, expected = [], {}, {}, []
    for cell in cells:
        i = cell['identity']
        ix, h = base.subset(data, i['held'])
        with np.load(old.checked(cell['paths']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            anchor = z['half_anchor']
        held = linear[ix]
        for n in np.unique(h['recording']):
            at = h['recording'] == n
            source, = np.unique(h['sites'][at])
            key = (i['fold'], i['seed'], str(n))
            p, b = original[key], reference[key]
            if (p['identity']['source'] != source or p['identity'] != b['identity']):
                raise ValueError('Bound does not match the recorded source')
            cv, ols = held[at].transpose(1, 0, 2, 3)
            y, valid, query = (h[k][at] for k in ('target', 'valid', 'query'))
            _, _, budget = audit.budgets(cv, anchor[at], query)
            row = dict(recording=ids.setdefault(str(n), len(ids)), source=sources.setdefault(str(source), len(sources)),
                       seed=i['seed'], **aggregate_view(cv, ols, y, valid, query, budget))
            rows.append(row)
            expected.append((p, b))
    if len(rows) != len(original) or len(rows) != len(reference):
        raise ValueError('Missing frozen recording-seed view')
    return dict(views=rows), expected, {k: old.sha(p) for k, p in files.items()}


def check(result, expected):
    def same(x, y):
        return x is y if x is None or y is None else isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)

    for r, (p, b) in zip(result['per_view'], expected, strict=True):
        terms = [(r['peak_budget'], p['budget']['peak'])]
        for scope in ('query', 'record'):
            terms.extend((r[k][scope], v) for k, v in zip(('cv', 'ols4'), p['arbitrary'][scope], strict=True))
        for a, k in (('available', 'known_only'), ('charged', 'with_missing_debit')):
            terms.extend((r[field][a], b[field][k]) for field in ('required_peak', 'required_multiple'))
        if not all(same(a, z) for a, z in terms):
            raise ValueError('Independent certificate differs from the original numerical bound')
    if {r['excluded'] for r in result['upper_bounds'].values()} != {67}:
        raise ValueError('Manuscript OLS4 exclusion count changed')


def build(output):
    payload, expected, hashes = export()
    files = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    files['evidence.json'] = (json.dumps(payload, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if any(archive.PRIVATE_MARKERS.search(v) or b'moving_source_component_' in v for v in files.values()):
        raise ValueError('Private identifiers in certificate')
    files['manifest.json'] = archive.main.as_json({k: archive.main.digest(v) for k, v in files.items()})
    with TemporaryDirectory(prefix='moving-peak-') as tmp:
        root = Path(tmp)
        a, b = root/'a.zip', root/'b.zip'
        archive.encode(a, files)
        archive.encode(b, files)
        data = a.read_bytes()
        if data != b.read_bytes():
            raise ValueError('Certificate archive is not deterministic')
        clean = root/'verify'
        with ZipFile(a) as z:
            z.extractall(clean)
        report = json.loads(subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=clean,
                            check=True, capture_output=True, text=True, timeout=45).stdout)
    check(report, expected)
    report.pop('per_view')
    if output.exists() and output.read_bytes() != data:
        raise FileExistsError('Refusing to replace a different certificate')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with output.with_suffix('.pending.zip').open('xb') as f:
            f.write(data)
        output.with_suffix('.pending.zip').replace(output)
    result = dict(**report, archive_sha256=archive.main.digest(data), archive_bytes=len(data),
                  source_hashes=hashes, result_source='fresh_dual_replay_of_frozen_development_inputs',
                  offline_stdlib=True, deterministic_rebuild=True, original_2pct_unchanged=True,
                  independent_confirmation=False, external_outcomes_read=False, submission_ready=False)
    OUT.mkdir(parents=True, exist_ok=True)
    old.once(OUT/'receipt.json', old.stable(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT)
    a = p.parse_args()
    audit.torch.set_num_threads(4)
    audit.torch.set_num_interop_threads(1)
    build(a.output)
