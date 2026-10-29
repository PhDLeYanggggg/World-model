# Pilot Interruption: Hash-Format Mismatch, Not a Scientific Result

PID74429 exited1 at the frozen input-hash assertion, after reading the first
four TRAIN packets and before retrieval or scientific scoring. No pilot pass,
full-run result or model update exists forv1. The original registration/code is
preserved by commit4cd2fafc and its input audit remains valid.

Reproduction on the same float32 array [1,2] gives different SHA256 values:
packet contract7c79d1f66cbb0ceee099a45e63dece584dbc8019ff68469071a5c5f1d4e664eb;
analogue-output contractc47c025396c0375461730e37fd1c797fce6ed3c3c94013a2955720cf15e50e55.
Packet hashing serializes(shape,dtype.str); analogue receipts serialize dtype
then shape. The wrapper incorrectly used the latter for frozen input and query
hashes. A mismatch alone was therefore not evidence of data drift.

Version2 repairs these two API boundaries, retaining analogue-output hashes
for analogue receipts. Explicit packet/query regression tests reject the wrong
format. No numeric algorithm, donor pool, selection, threshold or scientific
criterion changes. Original input audit is reused with its hash, not called a
new audit. Corrected registration is committed before another real pilot.
Failure is preserved, not hidden by deleting receipts or disabling assertions.
