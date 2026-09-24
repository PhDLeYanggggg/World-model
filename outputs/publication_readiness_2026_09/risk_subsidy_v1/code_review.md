# Pre-Readout Bounded Code Review

2026-09-24. A separate native subagent, with the inherited model and reasoning
setting, performed read-only review while the main agent implemented the study.
It did not train, create policy outputs, access confirmation data or use an
external model API. This is independent code scrutiny within the same research
workflow, not independent scientific replication or confirmation.

## Findings and Repairs Before New Policy Decisions

1. The diagnostic's other-source outcome counts could be mistaken for actual
   training support. V2 explicitly labels exposed-source outcome support and
   retains the limitation. No metric value changed.
2. Exact past-CV classification needed an explicit annotation lattice and
   range contract. Bounded half-pixel coordinates and integer past frames are
   now checked without fitted tolerances. Unsupported inputs fail.
3. The diagnostic now binds parent decision manifest and experiment identity.
   The initial source/report are retained locally as pre-review artifacts;
   V2 reproduces every case value and aggregate count.
4. A two-agent boundary example demonstrated that optimizing rounded transformed
   coefficients could miss a feasible original-inequality optimum. Small-query
   optimization now enumerates the canonical original inequality; larger
   nontrivial solver outputs are not claimed canonical-optimal. Exact all-
   supported solutions remain valid because all eligible gains are positive.
5. A replay could recreate a missing decision array or complete manifest.
   Replay now requires all original evidence and hashes before computation.
6. First identity creation preceded the single-writer lock. It now occurs under
   the lock, and verify mode cannot create a missing identity.

The reviewer reproduced the numerical and replay defects, then confirmed their
targeted repairs. Final review found no remaining blocker within this scope.
63 related main-agent tests pass; the review additionally ran focused subsets
and small state checks. Do not sum these overlapping test counts.

## Remaining Limits

Canonical optimum is unverified for large nontrivial numerical problems even
when the numerical solver reports optimal. Feasible nonempty solutions, failed
empty fallbacks, reference uncertainty and count mismatches are separate fields.
Selected-only risk allocation still permits budget sharing among selected
agents. Clipped expected net harm is not expected positive harm. Input causality
is relative to released annotations, not verified real-time annotation provenance.

These are exposed-source comparisons. Neither code review nor finite numerical
tests establish calibrated risk, independent generalization, safe deployment,
a world-model contribution or submission readiness.
