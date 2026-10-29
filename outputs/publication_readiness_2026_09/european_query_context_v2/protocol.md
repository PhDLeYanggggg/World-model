# Trainable Query-Context Encoder: Pre-Readout Repair

The information comparison,36fit roster, three head seeds,2,000updates/head,
all eight strong controls, original TRAIN roles,2%risk budget, no threshold
search and advancement screen are unchanged from
[the v1 protocol](../european_query_context_v1/protocol.md).

One implementation defect is repaired before any new held outcome readout:
the parent constructor freezes its base encoder before the new branch copies
it. The copied risk encoder therefore also inherited requires_grad=False.
v1 trained only130output parameters, not the intended12,322parameter risk
branch. All36 v1 fits and72,000updates are retained, explicitly marked as an
unintended frozen-encoder diagnostic without held evaluation or promotion.
Nothing is deleted, renamed as fresh, or selected using its held performance.

v2 explicitly enables gradients on the copied risk encoder only. The original
base remains frozen. Both arms have12,322trainable and25,471total parameters.
An exact formula guard rejects the wrong trainable count before optimization;
the real pilot and every completed fit must show actual encoder weight changes.
The new regression test fails on v1 and passes only after this repair.

Both corrected arms restart from the same original shared-utility anchor, not
the unintended v1 outputs. Same samples, RNG, total updates, loss, optimizer,
masking, inference and bootstrap settings. The whole36-head family freezes
before held readout; no reading independent roles, testing-prompt input or
held-informed pooling change. Commit this amendment before numerical work.

Native arm64 CPU4/inter-op1/workers0, exact50+50resume/direct100pilot,100update
atomic checkpoints/heartbeat,12h pause-and-resume boundary,10GiBfree reserve,
256MiBcheckpoint cap. Refer to the original protocol for the known limitations
of mean pooling, Deep Sets citation and all outcome/support reporting rules.

This is a repaired cost-learning experiment, not new trajectory training,
independent confirmation, safe deployment or submission readiness. Keep
obs8/pred12 rawstride12, detector-silver, image-local language. No metric,
seconds, physical safety, human gold, true3D or foundation claim.
Stage5C and SMC remain off.
