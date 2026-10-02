# Positive Conditional Harm Head: Development Only

**Intended use:** an experimental cost estimator for deciding whether an
already-frozen trajectory candidate should replace its motion reference.
Not a trajectory decoder, calibrated probability, risk guarantee or deployable
world model. The registered advance screen fails; existing deployment unchanged.

**Inputs:** the original380 causal features and forecast-disagreement envelope,
plus seven verified past-quality features. No future endpoint, future detector
quality, central velocity, test goal or validation-fitted normalization.

**Learned parameters:** two seven-feature conditional slopes per populated leaf
of each frozen forest.72 source heads, three fixed seeds,912,895 repeated tree
leaves. No new tree routing or neural encoder training. Normalizers use only
known TRAIN rows with original query weights; unknown labels remain unknown.

**Outputs:** total/easy harm updates; raw benefit/reference unchanged. Common
feasibility projection may change benefit. Guarded actions require original
positive utility, support and predicted2% total/easy harm conditions. These
predicted conditions are not claims about realized risk.

**Training:** fixed mean-normalized conditional deviance and regularization1.
Source whole-recording70/30 partitions, no hyperparameter or threshold search.
Exact refits/checkpointed inference checked. See `protocol.md` and `method_note.md`.

**Evidence:**12 already-exposed European development localities, obs8/pred12
rawstride12, image-local detector-silver. Three seeds are not independent scenes.
Nominal3000-locality bootstrap, independent-role selection/calibration/test closed.
Historical Stage35/37/43/44 results remain exploratory and are not promoted here.

**Negative results:** signed cost MSE worsens against both controls;11 easy-risk
upper violations remain, including two known-label violations. Matched-count
support below original. Positive mean preservation does not solve uncertainty
about unknown future outcomes or unseen scenes.

**Storage:** owned CREATE immutable checkpoints, manifest hashes in Git. No raw
video/image, feature/cache or weight arrays in the committed evidence package.
Native arm64 CPU, four configured threads, no multiprocessing loader. Total
full run22.04minutes, peak10.087GB, no new Slurm jobs.

**Unavailable claims:** independent generalization, metric/seconds, human-gold,
physical safety, true3D, foundation scale, submission readiness. Stage5C/SMC off.
