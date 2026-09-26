# Compute Location and CREATE Observation

## Local Training

The real 100-update pilot ran in the native-arm64 PyTorch environment, with
four compute threads, one inter-op thread and no DataLoader workers. Its
extrapolation was 239.14 summed fitting seconds for 576,000 updates. This
excludes feature extraction, data loading, diagnostics and verification;
it is not an end-to-end runtime promise. The full local job retains optimizer,
sampler and model state, emits progress every 200 updates, and keeps a 10 GiB
disk reserve. The pilot updates belong to the first head's fixed budget.

## CREATE

A previously approved, read-only account-queue query completed successfully
at 2026-09-26 01:27:15 UTC. It observed one running and two pending jobs from
another project. No M3W training or remote output was verified by this query.
No remote files were changed and no jobs were submitted. Existing jobs were
left untouched. The raw receipt remains private; its SHA-256 is
`393a55af117f4e8649e9692158259cad382542fcbb259e8a5e8f6a1b5333029a`.

The simulation-project handoff did not establish a verified remote M3W
directory or transferable job IDs. This bounded experiment fits local
resources, so there is no justification to guess a remote path or disturb
the other project. Future large training should use the verified handoff and
scheduler when data location, memory or measured runtime warrants it.

This observation is fresh_run, not proof of model quality or GPU availability.
The final compute receipt records measured summed fitting time separately.
Source-development image-pixel/annotation-step evidence only; Stage5C and
SMC remain off, with no metric, seconds, human-gold or foundation claim.
