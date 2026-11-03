# Candidate-Panel Inspection

I inspected all ten generated within-source panels on 3 November 2026:
`within_candidate_01.png` through `within_candidate_10.png`. They contain the
two strong multi-view pairs and the first eight lexically ordered ambiguous
pairs. This is model-assisted visual self-audit, not human gold, a complete
review of all 31 supported pairs or geographic certification.

The immutable `within_pair_contact_sources.json` binds exact figure hashes,
view slots, source JPEG hashes and actual image frames. Its SHA-256 is
`6f9bc64fc752fa1fb833062da483ecb62adad7c71dc15b022e4fe9ad9e55ac7e`.
All figures stay private and are not model inputs.

The two strong pairs are scene22 clips 5/6 and scene7 clips 2/4. The displayed
crosswalk/vendor roofs and market-front structures respectively are visually
consistent across the distinct view pairs. The eight ambiguous examples cover
scene10 clips 1/2, scene12 clips 1/2, scene14 clips 1/2 and 2/3, scene16 clips
2/3, 2/4 and 3/4, and scene17 clips 1/2. Several show clearly repeated plaza,
building or street structures despite changed scale and angle. Some ambiguous
pairs have only one displayed support row; the blank row is not a second match.
No weak relation is removed or upgraded by this inspection. All these pairs
already fall within retained publisher groups.

There are no supported cross-source pairs, so `pair_contact_sources.json`
correctly contains zero generated panels. SHA-256:
`e1c2646b20c7ad935a1f2ab51db4cffa501230a1fd0eaacce04b09ae37d180b9`.
Zero cross-source panels inspected is not an independent human judgment that
the sources never overlap. It is an empty candidate set under this matcher.

Of 213 pairs sharing a publisher scene label, only 31 have image support under
the frozen sparse-view criteria. Publisher labels are not verified shared-view
ground truth, so this is not an estimated false-negative rate. Together with
the oblique, night, water and repetitive-background observations, it prevents
treating negative matching as proof of geographic independence. All 44 new
publisher groups and the original 45 protected groups remain constraints; the
complete graph has 89 conservative components, not 89 verified physical sites.

No forecast error, model selection, training, role assignment or threshold
tuning occurs in this review. DUT imagery is outside this completed roster.
Protected confirmation and calibration roles, the original 2% contract,
Stage5C-off and SMC-off restrictions remain unchanged.
