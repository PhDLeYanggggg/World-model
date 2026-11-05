# Replay Scope

The diagnostic replay reproduces every stored gradient-geometry summary and
before/after loss, sign-regret and count statistic for all nine checkpoint views
and 27 disposable updates. The original checkpoint files and model objects
remain unchanged. Full gradient tensors and disposable model weight tensors
were not archived, so the verification flag's short name should not be read
as a stored tensor-by-tensor comparison. It is an exact replay of the recorded
diagnostic summaries, not independent scientific confirmation.

No diagnostic outcome, hypothesis decision, model or safety rule changes.
The separate real-training experiments do compare checkpoint weight tensors,
optimizer states, sample draws and predictions for their exact-refit checks.
