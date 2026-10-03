"""Transport-only recovery of the unchanged, registered cost experiment."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_cost_harm_newton as frozen
from scripts.m3w_bounded_packet_write import with_keepalive, write_payload

PUBLIC = frozen.PUBLIC
RECEIPT = PUBLIC/'transport_recovery_20261003.json'


class RecoveryStream(frozen.runner.Stream):
    def __init__(self):
        super().__init__()
        self.command = with_keepalive(self.command)

    def send(self, name, payload):
        expected = dict(group=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        fd = self.process.stdin.fileno()
        write_payload(fd, (json.dumps(expected)+'\n').encode())
        write_payload(fd, payload)
        actual = self.receive()
        assert actual == expected
        return actual


def registration():
    assert frozen.registration() == json.loads((PUBLIC/'registration.json').read_text())
    paths = [Path(__file__), ROOT/'scripts/m3w_bounded_packet_write.py',
             ROOT/'tests/test_m3w_bounded_packet_write.py', PUBLIC/'transport_recovery.md']
    return dict(bindings={str(p.relative_to(ROOT)): frozen.runner.sha(p) for p in paths},
        scientific_registration_sha256=frozen.runner.sha(PUBLIC/'registration.json'),
        scientific_model_changed=False, threshold_changed=False,
        outbound_write_timeout_seconds=60, keepalive_seconds=15, keepalive_count=2,
        checkpoint_bytes_and_hash_protocol_unchanged=True, independent_roles_read=False)


def main():
    if sys.argv[1:] == ['--register-transport']:
        frozen.runner.once(RECEIPT, registration())
        print('Registered transport-only recovery')
        return
    if sys.argv[1:] != ['run', '--resume']:
        raise SystemExit('Use --register-transport or run --resume')
    assert registration() == json.loads(RECEIPT.read_text())
    frozen.runner.parent.base.inter.committed(RECEIPT)
    frozen.runner.Stream = RecoveryStream
    frozen.main()


if __name__ == '__main__':
    main()
