"""Compact aggregate readouts without changing any scientific JSON bytes."""
import hashlib
import io
import json
from pathlib import Path
import zipfile


def digest(content):
    return hashlib.sha256(content).hexdigest()


def references(root, public):
    root, public = Path(root).resolve(), Path(public).resolve()
    complete = public/'complete.json'
    refs = json.loads(complete.read_text())['groups']; expected = {}
    for ref in refs:
        path = (root/ref['path']).resolve()
        if path.parent != public/'groups' or path.suffix != '.json':
            raise ValueError('Only named aggregate group JSON files belong in a metric bundle')
        name = str(path.relative_to(public))
        if name in expected:
            raise ValueError('Unique aggregate group paths required')
        expected[name] = ref['sha256']
    if not expected:
        raise ValueError('Completed aggregate readout required')
    return complete, expected


def verify(root, public, *, restore=False):
    public = Path(public).resolve(); complete, expected = references(root, public)
    archive = public/'metric_groups.zip'; manifest = json.loads((public/'metric_bundle.json').read_text())
    content = archive.read_bytes()
    if (manifest['archive_sha256'] != digest(content)
            or manifest['complete_sha256'] != digest(complete.read_bytes())):
        raise ValueError('Frozen metric bundle identity differs')
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        if sorted(z.namelist()) != sorted(expected):
            raise ValueError('Bundle must contain exactly the registered aggregate files')
        for name, sha in expected.items():
            value = z.read(name)
            if digest(value) != sha:
                raise ValueError('Aggregate JSON bytes changed')
            path = public/name
            if path.exists() and digest(path.read_bytes()) != sha:
                raise ValueError('Existing local aggregate differs; do not overwrite it')
            if restore and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    return dict(groups=len(expected), archive_bytes=len(content), complete_sha256=manifest['complete_sha256'],
                archive_sha256=manifest['archive_sha256'], bytes_verified=True,
                scientific_recalculation=False, raw_data_or_weights_included=False)


def pack(root, public):
    public = Path(public).resolve(); complete, expected = references(root, public)
    if (public/'metric_groups.zip').exists() or (public/'metric_bundle.json').exists():
        return verify(root, public)
    buf = io.BytesIO(); total = 0
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, sha in sorted(expected.items()):
            content = (public/name).read_bytes()
            if digest(content) != sha:
                raise ValueError('Registered aggregate file changed before packaging')
            total += len(content)
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED; entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            z.writestr(entry, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    content = buf.getvalue()
    (public/'metric_groups.zip').write_bytes(content)
    (public/'metric_bundle.json').write_text(json.dumps(dict(format='zip_deflate_aggregate_metrics_only',
        archive_sha256=digest(content), complete_sha256=digest(complete.read_bytes()),
        groups=len(expected), uncompressed_bytes=total, archive_bytes=len(content),
        numerical_JSON_bytes_unchanged=True, raw_data_or_weights_included=False), indent=2)+'\n')
    return verify(root, public)
