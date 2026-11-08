# Working Decision Supplement v62

The main manuscript stays at v50. This supplement adds the donor-held
selected-action calibration study and exact same-bank, same-cap hindsight
audit. Neither result qualifies the controller or opens confirmation.
The approved DroneCrowd split remains 15 calibration groups and 30
independent confirmation groups under the original 2% rule.

```sh
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v62.py \
  --source-zip data/stage_cvpr2027_experiments/evidence_manuscript_v62/supplement_source.zip
```

The builder verifies the input hashes and copies the previous supplement
before inserting the bound readout. Local direct and archive-source
builds both have 17 pages and identical 85 DPI page renders. The new
paragraph and bibliography ending were visually checked. The archive
and PDF are local build outputs, not tracked files.

- PDF SHA-256: `77e8c29ac0c4718a7ad238149befd92c4a06d06254d51ceb87e5a67614a9d121`
- Source ZIP SHA-256: `30123b3fa26d880cec7ea244774e3fbf395c743b427e861296797d55915c2d0e`
