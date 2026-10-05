# Transport filename compatibility fix

The initial pilot transfer stopped before any input file or optimizer update.
The receiver allowed alphanumeric characters and underscores but the registered
source identity contains `eu-locality-008`. Its missing hyphen allowance caused
the receiver assertion and the sender's broken pipe. A subsequent read-only query
confirmed the owned input directory had no files and no training was submitted.

Execution revision 2 adds only the hyphen to this filename allowlist. The
regression test fails against revision 1 on the actual registered identity and
rejects path traversal, slashes, shell separators and embedded newlines. Numeric
packets, optimizer code, TRAIN partition, loss and evaluation rules are unchanged.

The original execution registration and remote owner marker are retained. The
revised local transport manager is separately hash-registered. It is not needed
by the remote numerical entrypoint, so it is excluded from the executable code
manifest instead of overwriting an immutable remote file. Its original unused
remote copy is left intact. All actual imported optimizer code remains pinned.

This is an engineering failure and repair, not a model-training result.
