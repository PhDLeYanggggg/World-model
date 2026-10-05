# Preserve the Original Submission Manifest Link

The first revision-4 archival attempt stopped at its manifest assertion before
any file mutation or resubmission. The original job intent correctly retains the
pre-root-repair manifest hash. Requiring it to equal the amended manifest was an
incorrect recovery precondition, not a changed scientific input.

Execution revision 5 verifies the existing remote revision-3 repair receipt
against the committed local receipt. It requires original intent hash = recorded
before hash, and current manifest hash = recorded after hash. Both current input
manifests and every numerical code binding must still verify. Arbitrary changes
to either the intent or amendment fail before mutation. Prior records are not
rewritten. The login-shell change remains the only batch-execution change.

The pre-fix regression reproduces the assertion with a faithfully amended
manifest. It and rejection tests pass after this narrow fix. No optimizer,
model, data, split, loss, checkpoint budget, risk limit or evaluation rule changes.
Revision-4 registration and failed attempt remain historical. This is execution
recovery, not model improvement or completed scientific training.
