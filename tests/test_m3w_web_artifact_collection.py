import stat
import zipfile

import pytest

from scripts import import_m3w_easy_harm_web_artifacts as importer


def test_archive_accepts_only_exact_lightweight_members(tmp_path):
    path = tmp_path / 'fits.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('./', '')
        archive.writestr('fit.json', '{}')
    assert importer.archive_files(path, {'fit.json'}) == {'fit.json': b'{}'}


@pytest.mark.parametrize('case', ['extra', 'missing', 'duplicate', 'traversal', 'symlink', 'oversize'])
def test_unsafe_archive_rejected(tmp_path, case):
    path = tmp_path / 'fits.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        if case != 'missing':
            name = '../fit.json' if case == 'traversal' else 'fit.json'
            info = zipfile.ZipInfo(name)
            if case == 'symlink':
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, 'x' * (2 * 2**20 + 1) if case == 'oversize' else '{}')
        if case == 'extra':
            archive.writestr('checkpoint.pt', 'unwanted')
        if case == 'duplicate':
            with pytest.warns(UserWarning):
                archive.writestr('fit.json', '{}')
    with pytest.raises(ValueError):
        importer.archive_files(path, {'fit.json'})


def test_install_checks_conflicts_before_any_write(tmp_path):
    new = tmp_path / 'new.json'
    old = tmp_path / 'old.json'
    old.write_bytes(b'old')
    with pytest.raises(ValueError):
        importer.install({new: b'new', old: b'changed'})
    assert not new.exists()
    assert old.read_bytes() == b'old'


def test_idempotent_byte_preserving_collection_is_not_scheduler_admission(tmp_path):
    path = tmp_path / 'training_freeze.json'
    raw = b'{"neural_fits":144}\n'
    importer.install({path: raw})
    importer.install({path: raw})
    assert path.read_bytes() == raw
    assert not (tmp_path / 'train_collection.json').exists()


def test_install_rejects_symlink(tmp_path):
    original = tmp_path / 'original.json'
    original.write_bytes(b'{}')
    link = tmp_path / 'link.json'
    link.symlink_to(original)
    with pytest.raises(ValueError):
        importer.install({link: b'{}'})
