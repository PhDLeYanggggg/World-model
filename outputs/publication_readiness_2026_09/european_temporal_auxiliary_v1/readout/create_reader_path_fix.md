# CREATE mounted-root containment repair

The read-only October 5 path diagnosis verified that the authorized `/users/`
M3W experiment resolves into its Ceph home volume. Comparing a resolved file
against the unresolved directory incorrectly rejects this legitimate mapping.
The first whole-input rehash stopped at this containment check; it was not a
training, checksum or scientific-model failure.

Reader revision 2 first verifies the exact authorized logical experiment path,
then resolves that root before checking its descendants. Checkpoint/code files
which resolve outside the owned experiment remain rejected. Owner, hash, size,
manifest, complete-training and committed-freeze requirements are unchanged.

The synthetic mounted-root regression fails for collection and streaming in
revision 1. Revision 2 must accept the owned alias and reject a checkpoint
symlink into an unrelated directory. The original registration is retained;
the new transport-only registration links its hash. The scientific readout,
training code, split, risk budget and selection rules are unchanged.

No real checkpoint collection or validation evaluation is authorized by a
filesystem repair. A completed, verified 216-head training freeze is still
required. This is engineering evidence, not a safer-policy result.
