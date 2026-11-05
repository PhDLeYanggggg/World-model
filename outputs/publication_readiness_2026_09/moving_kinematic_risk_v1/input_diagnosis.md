# Why This Is A Representation Repair, Not A New Threshold

Static trace of the current causal input, verified against the real builder:

| Zero-based columns | Contents |
|---|---|
| 0:24 | Eight ego x/y/time values |
| 24:32 | Eight past validity flags |
| 32:224 | Eight neighbors, eight past x/y/time values each |
| 224:288 | Neighbor past validity flags |
| 288:306 | Floor/candidate/disagreement mean, spread and predicted endpoint |
| 306:330 | Causal floor rollout |
| 330:354 | Causal candidate rollout |
| 354 | Log past-only scale |
| 355:379 | Causal CV rollout |
| 379 | Causal floor-switch flag |
| 380:428 | Previously tested shared-motion48; masked in both current arms |

Predicted endpoints here are model predictions derived from observed inputs,
not target endpoints. Actual future targets remain in the separate supervision
and evaluation interface. Raw neighbor validity, not normalized validity, gates
the new feature calculations. Invalid neighbors contribute nothing.

The current critic sees this flattened history through a single64-unit linear
layer, SiLU and query-mean context. It therefore has the coordinates but must
learn products, norms, backward differences and relative-motion relationships.
The new64-field basis makes selected relationships explicit without new data.
Its scale comes from the already admitted past-only coordinate normalization;
extra standardization is fitted only on the critic's own sources.

The previous matched shared-motion48 experiment failed its held utility and
risk tests. It is not silently re-enabled. The previous affine and nonlinear
readout failures likewise remain negative. A proposed feature-basis advantage
is a hypothesis until the matched neural fits and held-source readout finish.

Engineering preflight: ten targeted tests pass. The synthetic checks include
an analytically known straight path, a stationary approaching neighbor with
closest approach at two native steps, masked-neighbor invariance, zero motion,
query isolation, same warm predictions, real optimizer weight changes,
interrupted/uninterrupted equality and rejection of future endpoint inputs.
These checks are not evidence of real-world improvement or physical safety.

No newly acquired scene image, goal, video representation or trajectory model
is introduced. Current scope remains native obs8/pred12 image-coordinate
SOURCE development, not verified metric/seconds, true3D or foundation evidence.
