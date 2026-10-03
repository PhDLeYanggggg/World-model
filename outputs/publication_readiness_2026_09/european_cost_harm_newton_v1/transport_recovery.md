# Checkpoint Transport Recovery, 3 October 2026

The first full run (PID 49738) completed 36 heads and then stopped making
progress. The 3 October observation found 20h21m elapsed but only 31m37s CPU
time. The last heartbeat was 2 October 18:28:59 UTC. A native process sample
located its main thread in FileIO write; the checkpoint SSH sender was waiting
in pselect. A fresh SSH connection succeeded. The owned remote directory had
36 complete checkpoints, 81,730,626 bytes, and no temporary files.

Only the three confirmed child checkpoint transports were terminated. The
original training process exited 1 with BrokenPipeError at PacketStream.send,
confirming the blocked operation was checkpoint writing, not an active Torch
training kernel. A network/host interruption is plausible, but the underlying
network cause is not proven. No scheduler or simulation job was touched.

The original writer bounded acknowledgement reads but not outbound pipe writes.
The recovery wrapper adds a 60-second nonblocking write deadline and SSH
ServerAliveInterval=15 / ServerAliveCountMax=2 to its checkpoint sender. It
preserves authentication, strict host verification, destination, byte payloads,
hash checks, atomic remote checkpoint writes and existing reader timeouts. It
does not change the registered fit, data, objective, inference, thresholds or
scientific results. A separate operational receipt binds this wrapper and tests.

Resume verifies all 36 completed reports and remote hashes. The incomplete 37th
head must be refitted. No sample or head is dropped; the final target remains
72. The long blocked elapsed period must remain in the operational record and
must not be represented as active training time. Per-invocation 12-hour bounds
remain unchanged; a transport timeout exits rather than silently waiting past
the cap. This repair does not promise immunity to all network failures.

Regression checks reproduce a stalled real pipe, partial writes, receiver exit
and preservation of existing authentication arguments. They verify transport,
not research success. Independent scientific data roles remain closed.
