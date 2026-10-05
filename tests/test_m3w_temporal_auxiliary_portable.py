import hashlib
import ast
from pathlib import Path

import pytest
from scripts import train_m3w_temporal_auxiliary_portable as port
from test_m3w_temporal_auxiliary import data, settings


def test_packet_round_trip_and_optimizer_match(tmp_path):
    args = data()
    decoded = port.unpack(port.packet(args))
    port.run.api.core.exact(args, decoded)
    states = []
    for name, values in [('local', args), ('packet', decoded)]:
        states.append(port.run.api.fit(*values, arm='temporal', settings=settings(), seed=17,
            identity={'test': 'same'}, path=tmp_path/(name+'.pt.gz'), heartbeat=lambda **kw: None))
    for key in states[0]:
        if key != 'seconds': port.run.api.core.exact(states[0][key], states[1][key])


def test_transport_identity_and_corruption_fail_closed(tmp_path):
    content = port.packet(data()); (tmp_path/'inputs').mkdir()
    path = tmp_path/'inputs/g.npz'; path.write_bytes(content)
    ref = dict(group='g', bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), identity={'group': 'g'})
    manifest = {'heads': [ref]}
    assert next(port.verified_iterator(tmp_path, manifest))[0] == ref['identity']
    ref['identity'] = {'group': 'wrong'}
    with pytest.raises(ValueError, match='identity'): next(port.verified_iterator(tmp_path, manifest))
    path.write_bytes(content[:-1])
    with pytest.raises(ValueError, match='checksum'): next(port.verified_iterator(tmp_path, manifest))


def test_personal_quota_preserves_original_reserve(tmp_path):
    cfg = dict(disk_reserve_bytes=100, checkpoint_cap_bytes=20, atomic_checkpoint_headroom_bytes=2)
    assert not port.quota_storage(tmp_path, tmp_path, cfg, quota_free=121)['allowed']
    assert port.quota_storage(tmp_path, tmp_path, cfg, quota_free=122)['allowed']
    (tmp_path/'checkpoint.pt.gz').write_bytes(b'a'*21)
    with pytest.raises(OSError, match='cap'): port.quota_storage(tmp_path, tmp_path, cfg, quota_free=1000)


def test_embedded_transport_and_scheduler_programs_parse_and_fail_closed():
    from scripts import manage_m3w_temporal_auxiliary_create as manager
    tree = ast.parse(Path(manager.__file__).read_text())
    programs = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                and any(isinstance(t, ast.Name) and t.id in ('code', 'RECEIVER') for t in n.targets)]
    assert len(programs) == 6
    for program in programs: ast.parse(program)
    submission = next(x for x in programs if "['sbatch'" in x)
    assert submission.index("with intent.open('x')") < submission.index("['sbatch'")
    assert 'squeue' in submission and 'sacct' in submission
    assert 'unknown_timeout_do_not_resubmit' in submission
    assert manager.INPUT_CAP == 4*2**30


def test_packet_bytes_are_stable_and_future_targets_are_separate():
    args = data()
    assert port.packet(args) == port.packet(args)
    decoded = list(port.unpack(port.packet(args)))
    assert decoded[0].shape == args[0].shape
    decoded[3] = decoded[3]+5
    assert port.packet(tuple(decoded)) != port.packet(args)
    port.run.api.core.exact(decoded[0], args[0])
