# Claude Handoff Receipt

The user stopped Codex research on 2026-11-10 and requested a transfer of
completed work and remaining tasks. This update only organizes and synchronizes
existing artifacts. It runs no training, scientific evaluation, test suite,
remote job query, manuscript edit, or portal action.

Start with [CLAUDE.md](../../../CLAUDE.md) and the complete
[project handoff](../../../CLAUDE_HANDOFF.md). The handoff maps the repository,
local data and checkpoints, external studies, prior attempts, current negative
results, paper/Word files, CREATE locations, constraints, and unfinished work.

The latest completed experiment is
[`opendd_advantage_factorial_v1`](../opendd_advantage_factorial_v1/README.md).
It trained 546 models for 260,400 updates before the pause. Its validation-selected
policy has 48 full-grid passes versus 55 for keeping the incumbent. The primary
fails; there is no new deployment or independent external confirmation.

`manifest.json` records 2,835 local files and 2,707 artifact references checked
at handoff. The original runner reference is verified from its registered Git
revision, and its authorized replacement is verified against the selection
amendment. This preserves the historical registration rather than silently
rewriting it. These are integrity checks, not a fresh scientific replication.

The latest local cache contains 210 producer weights, 336 selector weights,
420 teacher caches, 441 packet caches, 63 inner-cell records, and 21 frozen
choice records. Only hashes and paths, not these private payloads, are included
in the handoff manifest. Cloning GitHub alone does not recover the local data.

The author's Word and v219 paper/supplement/replay hashes match the previous
protected snapshot. Existing staged work is preserved. Some entries in the
default Git index are older than HEAD and appear as staged reversals; do not
commit that index wholesale. Use a scoped index and inspect the current diff.

The manifest is a snapshot of the local working tree, including the preexisting
uncommitted README/state content. The scoped Git commit intentionally carries
only this handoff's changes to those shared files. Thus their whole-file local
hash need not match a clean checkout. It is not evidence that the clean checkout
contains all unrelated local edits.

The files are available to Claude Code in the shared workspace. No direct
Claude conversation message or read acknowledgement has been verified.
