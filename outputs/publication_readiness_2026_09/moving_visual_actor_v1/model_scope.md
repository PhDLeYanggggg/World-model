# Model Scope And Prior Work

The original [EqMotion paper](https://arxiv.org/pdf/2303.10876),Sections4.1-4.5,
separates geometric features from pattern features and uses learned interaction
reasoning. Section5.4distinguishes deterministic prediction from best-of20
evaluation. Those established mechanisms belong to Xu etal.,not M3W. The paper
sections and locally pinned author implementation were checked on6November.

## What Is Adapted Here

The local adapter fixes one head before fitting,keeps only complete aligned
past neighbors within the registered nearest-eight support and restores input
conditioning scale before loss and scoring. It does not reproduce the paper's
training recipe,metric ETH-UCY benchmark or best-of20numbers. Its task is the
registered native8/12image-coordinate MovingDroneCrowd++ component comparison.

This experiment adds a small,past-image-conditioned node-feature offset to the
same core and trains the full trajectory network. It does not add a trajectory
residual decoder or predict future camera poses. The offset is bounded in latent
feature space;that does not bound coordinate error or certify physical safety.

The image/context vector includes native image bounds and measurement features.
It is not proven invariant under every Euclidean transform. Therefore the
modified whole model must not inherit the original equivariance theorem merely
because its core is EqMotion. That restriction is a project inference from the
new input path,not a negative claim about the published architecture.

## Evidence Boundary

Zero initialization and frozen raw-control context preserve the old initial
function and RNG. A full old-control refit tests actual numerical equivalence;
small synthetic parity alone is not a real-data result. Quality and motion
actors share nominal and trainable capacity,draws,objective and budget. Six
source-held designs do not constitute six independent datasets or three seeds.

Past sparse tracking/affine measurement and simple neural conditioning are
conventional tools,not standalone novelty. A successful actor component would
only justify a separately registered relative-harm/joint-policy experiment.
The paper still requires a useful decision contribution,strong matched controls,
the original risk/support requirements,independent evidence and honest limits.
