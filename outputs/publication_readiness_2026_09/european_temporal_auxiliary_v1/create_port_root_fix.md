# CREATE training-root execution repair

The October 5 read-only filesystem diagnosis proves that `/users/k24101830/m3w`
maps to a Ceph directory. The portable training entry point resolved its actual
root but compared it against an unresolved expected parent. This necessarily
rejects the legitimate experiment before its first optimizer update. A focused
regression reproduces that defect; resolving both sides repairs it while still
rejecting foreign roots. The original optimizer and registered science do not
change.

Execution revision 3 preserves the registered job rather than submitting a
duplicate. Only pilot 37790290 is eligible, and only while authoritative Slurm
state is PENDING and no experiment checkpoint or completed pilot exists. A
separate bounded repair first holds that pending job, verifies the held state,
archives the prior runner and both input manifests, then atomically changes
the runner and its manifest binding. Packet bytes, identities and all other
code bindings remain the same. Nothing in another project is modified.

The repair is explicit and recoverable. On partial failure the job stays held;
it must not be released until all registered bytes are verified. Repeated repair
recognizes its exact receipt and does not repeat scheduler actions. A separate
release verifies the repaired bytes and resumes the same queued job. Original
registrations, transfer receipts, source bytes and job ID are preserved. This
is not a cancellation, new training result or a scientific protocol revision.

The earlier packet-transfer manifest hashes remain historical. Revision 3 has
new manifest hashes because one execution-code binding changes. No numerical
TRAIN packet is changed or re-exported. Full training still requires the real
pilot's loss/checkpoint/resume/resource checks; all 216 heads must finish before
the unchanged seven-arm readout. No validation/independent roles, deployment,
Stage5C, SMC or physical-unit claim is opened by this repair.
