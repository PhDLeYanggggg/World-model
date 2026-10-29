# Why This Version Exists

The fit receipt parameter census, not an outcome comparison, exposed a defect.
`parent.initialize(anchor, isolated_risk).base` returns a frozen base.
`deepcopy(base.primary.encoder)` preserves the False requires_grad flags.
The output layer trained normally, so finite losses, exact resume and identical
arm parameter counts did not catch the missing encoder updates.

A targeted regression requiring copied-encoder gradients fails on the original
implementation. The repair sets requires_grad=True on that disjoint copy only;
the original base parameters remain frozen. New guards independently check
the12,322trainable count and actual post-training encoder weight changes.
The exact-resume and frozen benefit/reference tests remain required.

The unintended v1 run completed36fits/72,000updates, PID58797 exit0. It is
preserved under european_query_context_v1 with its own registration, checkpoints
and training receipts. Its held readout is not_run. The historical code is
available at registration commit a3c8699d. The v1 protocol is not silently edited
to make the defect look intentional. v2 is a newly registered restart from the
original anchor, with no results from v1 used to choose a model or threshold.

This verifies a local trainability defect, not the scientific cause of previous
M3W failures. No claim is made that enabling these gradients will improve held
utility or risk. Both statistical results, if later evaluated, must retain
their distinct implementation provenance.
