# OpenDD: train on the alternatives within one recording

Test two changes to the frozen one-exchange setup on exposed OpenDD
development data: training against other options in the same legal
slate, and supervising that ranking with the full-grid certificate.
All controls share the same options, eight-step causal features, three
seeds, donor batches, architecture and 400 AdamW updates. Compare:

1. Pointwise BCE with the original dual-pass target.
2. The same BCE plus a listwise loss with the original target.
3. The same combined loss with the original target intersected with
   the full-grid certificate.

The listwise term assigns equal mass to the passing alternatives and
normalizes scores within a sampled slate. Homogeneous slates contribute
only BCE. Sample 32 slates per step with equal probability over donor
locations and then over their slates. Use the archived nomination and
sampling design (up to 32 options per slate). The proposal producer and
cutoffs for a donor exclude that donor and the outer held location. Only
the six non-held locations provide training examples. There is no
held-data early stopping or threshold fitting.

Freeze the 21 cells' held choices before scoring. At readout, all new
rankers intervene exactly where the archived aligned ranker intervenes:
101 recording-seed views and 780 total edits on the 606-view roster.
They pick the highest score within the same legal menu. Keep the original
movement caps and 2% checks. Compare also with the frozen incumbent,
archived aligned choices, and exact uniform expectation on this mask.
The primary outcome is the full-grid pass rate. Report original and
material (>0.10 pp against both references) passes, and paired 3,000-draw
intervals across the seven locations after averaging seeds. An improvement
needs a positive lower endpoint against both the same-mask uniform arm and
the equal-training original-target listwise arm to support the stronger
target. No result authorizes deployment or constitutes independent testing.
