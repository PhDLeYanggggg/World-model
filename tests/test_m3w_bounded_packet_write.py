import os
import threading
import time

import pytest

from scripts.m3w_bounded_packet_write import with_keepalive, write_payload


def test_stalled_pipe_times_out_and_restores_fd_mode():
    read, write = os.pipe()
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match='Checkpoint write timed out'):
            write_payload(write, b'x'*(2**20), timeout=.05)
        assert time.monotonic()-start < 2
        assert os.get_blocking(write)
    finally:
        os.close(read)
        os.close(write)


def test_partial_writes_preserve_exact_payload():
    read, write = os.pipe()
    payload = bytes(range(256))*4096
    chunks = []

    def receiver():
        while True:
            chunk = os.read(read, 1991)
            if not chunk:
                break
            chunks.append(chunk)

    thread = threading.Thread(target=receiver, daemon=True)
    thread.start()
    try:
        write_payload(write, payload, timeout=3.)
        assert os.get_blocking(write)
    finally:
        os.close(write)
        thread.join(timeout=3.)
        os.close(read)
    assert not thread.is_alive()
    assert b''.join(chunks) == payload


def test_closed_receiver_fails_explicitly():
    read, write = os.pipe()
    os.close(read)
    try:
        with pytest.raises(BrokenPipeError):
            write_payload(write, b'checkpoint')
        assert os.get_blocking(write)
    finally:
        os.close(write)


def test_keepalive_preserves_original_authentication_and_destination():
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
               '-i', '/example/key', 'user@example', 'unchanged receiver command']
    result = with_keepalive(command)
    assert result == command[:1]+['-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']+command[1:]
    assert command[1] == '-o'
    with pytest.raises(ValueError):
        with_keepalive(['ssh', 'user@example'])
    with pytest.raises(ValueError):
        with_keepalive(result)
