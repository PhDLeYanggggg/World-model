"""Replay the supplementary statistics using only the standard library."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import runpy
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile


MEMBERS = {
    'README.md', 'verify.py', 'reading_guide.pdf', 'technical_appendix.pdf', 'songdo_confirmation.pdf',
    'reviewer_evidence.zip', 'moving_peak_certificate.zip', 'songdo_replay.zip', 'moving_completion.json',
    'moving_completion.py', 'wepdtof_completion.py',
}


def verify_members(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Supplement manifest does not match its expected roster')
    if any(not (root/k).is_file() or (root/k).is_symlink()
           or sha256((root/k).read_bytes()).hexdigest() != v for k, v in hashes.items()):
        raise ValueError('Supplement member differs from the manifest')


def unpack(path, root):
    with ZipFile(path) as z:
        entries = z.infolist()
        names = {e.filename for e in entries}
        if len(names) != len(entries) or not {'verify.py', 'manifest.json'} <= names:
            raise ValueError('Unexpected replay archive')
        for e in entries:
            p = PurePosixPath(e.filename)
            if (p.is_absolute() or '..' in p.parts or p.as_posix() != e.filename
                    or '\\' in e.filename or e.is_dir() or stat.S_ISLNK(e.external_attr >> 16)):
                raise ValueError('Unexpected path in replay archive')
        z.extractall(root)


def run_replay(root):
    p = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=root,
                       capture_output=True, text=True, timeout=120, check=True)
    return json.loads(p.stdout)


def verify(root):
    verify_members(root)
    with TemporaryDirectory() as temp:
        temp = Path(temp)
        for k in ('reviewer_evidence', 'moving_peak_certificate', 'songdo_replay'):
            unpack(root/(k+'.zip'), temp/k)
        base, old_peak, songdo = (run_replay(temp/k) for k in ('reviewer_evidence', 'moving_peak_certificate', 'songdo_replay'))

        peak_path = temp/'moving_peak_certificate/evidence.json'
        extra = json.loads((root/'moving_completion.json').read_text())
        raw = peak_path.read_bytes()
        if extra['original_evidence_sha256'] != sha256(raw).hexdigest():
            raise ValueError('Completion data must extend the same original Moving certificate')
        moving = runpy.run_path(str(root/'moving_completion.py'))
        peak = SimpleNamespace(**runpy.run_path(str(temp/'moving_peak_certificate/verify.py')))
        views = moving['attach'](json.loads(raw), extra)
        completed = moving['report'](views, old_peak, peak)
        m = base['headline']['moving']
        if (m['views'] != completed['views'] or m['recordings'] != completed['recordings']
                or m['ols4_excluded'] != completed['completion_envelope']['record']['unfavorable_completion']['excluded_views']):
            raise ValueError('Moving headline and completion certificate disagree')

        external_path = temp/'reviewer_evidence/external'
        external = runpy.run_path(str(external_path/'verify.py'))
        wep = runpy.run_path(str(root/'wepdtof_completion.py'))
        wepdtof = wep['replay'](json.loads((external_path/'evidence.json').read_text()), base['external'], external['exact_interval'])
    return dict(original=base, moving=completed, wepdtof=wepdtof, songdo=songdo)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
