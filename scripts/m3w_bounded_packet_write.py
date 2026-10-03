"""Bound only checkpoint transport, without changing scientific payloads."""
import os
import select
import time


def write_payload(fd, payload, timeout=60.):
    if timeout <= 0:
        raise ValueError('Positive transfer timeout required')
    deadline = time.monotonic()+timeout
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    view = memoryview(payload)
    try:
        while view:
            left = deadline-time.monotonic()
            if left <= 0 or not select.select([], [fd], [], left)[1]:
                raise TimeoutError('Checkpoint write timed out; preserve completed packets and resume')
            try:
                written = os.write(fd, view)
            except BlockingIOError:
                continue
            if not written:
                raise BrokenPipeError('Checkpoint transport closed')
            view = view[written:]
    finally:
        os.set_blocking(fd, blocking)


def with_keepalive(command):
    if not isinstance(command, list) or os.path.basename(command[0]) != 'ssh':
        raise ValueError('An explicit SSH argument vector is required')
    if 'BatchMode=yes' not in command or 'StrictHostKeyChecking=yes' not in command:
        raise ValueError('Keep existing authentication and host verification')
    if any(str(v).startswith(('ServerAliveInterval=', 'ServerAliveCountMax=')) for v in command):
        raise ValueError('Do not override an existing keepalive policy')
    return command[:1]+['-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']+command[1:]
