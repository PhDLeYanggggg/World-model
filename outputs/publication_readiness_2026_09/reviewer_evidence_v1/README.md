# Offline reviewer evidence

I combined the headline, external-evaluation and matched-budget
replays into one self-contained archive. An independent extraction
replays all three with the standard library and cross-checks the
WEPDTOF count against the main paper. The new OpenDD replay computes
exact empirical site-bootstrap intervals from anonymous location
means, so the conclusions do not depend on the original 3,000 draws.

| OpenDD development contrast | Conservative gain (pp) | Exact empirical 95% interval (pp) |
| --- | ---: | ---: |
| Joint vs additive | -0.005331 | [-0.009074, -0.002399] |
| Additive + support vs additive | -0.000690 | [-0.001079, -0.000410] |
| Additive + support vs additive + shuffled | -0.001171 | [-0.001721, -0.000683] |

The intervals enumerate all ordered seven-location resamples through
their multiplicities. They are not exact population intervals, and
they are not a substitute for independent confirmation. The observed
contrast of joint vs additive still spans zero. The headline replay
reproduces 330/606 hindsight and 57/606 causal OpenDD passes, 67/72
nonpositive relaxed OLS4 bounds on MovingDroneCrowd++, and 0/45
WEPDTOF dual passes. It preserves the unresolved OpenDD hindsight view.

## Artifact and check

Local file: `data/stage_cvpr2027_experiments/submission_admin/reviewer_evidence.zip`.
The SHA-256 and exact replay values are in [receipt.json](receipt.json).
The two builds matched byte for byte. The clean-directory verifier
needs no repository, no trained weights and no third-party packages.
The archive does not contain trajectories, source identifiers,
author records, private paths or external URLs. A direct marker
scan is not proof that public results cannot be linked to their authors.

```sh
.venv-pytorch/bin/python scripts/package_m3w_reviewer_evidence.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_reviewer_evidence.py tests/test_m3w_opendd_support_rerank.py tests/test_m3w_opendd_pair_disagreement.py
# In the extracted reviewer directory:
python -I -B -S verify.py
```

The builder reuses the three original ZIPs after verifying their
hashes; their build commands are in the
[headline](../main_claims_replay_v1/README.md),
[budget](../budget_addendum_v1/README.md) and
[external](../wepdtof_scalar_replay_v1/README.md) reports.
Twelve focused tests passed. The existing main and supplementary
PDFs are unchanged. This is a scalar verification companion, not
end-to-end model reproduction or new empirical evidence. Nothing was
uploaded. Songdo and reserved DroneCrowd labels remain unopened.

The [CVPR 2027 author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
allow supplementary code and require self-contained review material.
This archive is a local candidate for that purpose; no OpenReview
upload field or submission status is inferred from its creation.
