# Primary-source reading and acquisition scope

Read on 2026-09-24. This is a focused source-admission review, not a full systematic
review. No secondary repository or search-summary claim authorizes data use.

## HT21 / CroHD

- [Publisher archive](https://motchallenge.net/data/Head_Tracking_21/): download
  section, sequence tables and dataset description read. The labels-only link
  resolves to the acquired archive. The benchmark's online service is closed;
  published test-set counts are not downloadable GT. Numeric counts in our
  report are independently measured from local bytes.
- [Sundararaman et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.pdf):
  dataset description, annotation procedure and scene illustration inspected.
  Tracking scenes recur across the supplied splits; static labels reflect
  whole-recording motion. This prevents treating either split names or static
  classes as causal forecasting evidence.
- [Supplement](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Sundararaman_Tracking_Pedestrian_Heads_CVPR_2021_supplemental.pdf):
  annotation subsection inspected through available indexed text; direct refresh
  also returned HTTP403. The described keyframe interpolation is not a supplied
  per-row provenance mask. No keyframes are inferred by modulo arithmetic.
- [TrackEval dataset reader](https://raw.githubusercontent.com/JonathonLuiten/TrackEval/master/trackeval/datasets/head_tracking_challenge.py):
  class mapping and pedestrian filtering inspected, not executed or vendored.
  Mapping: 1 pedestrian, 2 static, 3 ignore, 4 person-on-vehicle. Its tracking
  filtering is not adopted as a forecasting-population definition. The URL is a
  mutable reference, not a pinned software dependency.

The source INI files independently specify camera-motion flags and 25fps. Only
INI/label bytes, not video clocks or world-coordinate transforms, were verified.
Neither a paper license nor an evaluator code license is assumed to license the
dataset. No resolved dataset-use declaration was established in this bounded
review; acquisition remains quarantined and redistribution excluded.

## CrowdTraj lead

[Wijaya, Henderson and Mahmoud, arXiv:2609.07685v1](https://arxiv.org/html/2609.07685v1),
posted 2026-09-07: abstract, data collection, annotation, coordinate section and
scene table inspected. Its five recording conditions span three named locations;
Duri recordings must not automatically be counted as independent physical sites.
The declared homographies would require inspection of the released files before
supporting metric claims in this project.

No verifiable official annotation-download endpoint was found in the inspected
paper links and bounded official-source search. This is a search limitation,
not proof that no release exists. Acquisition and predictive evaluation are
`not_run`; no author was contacted, and no third-party mirror substituted.
The article's license is not substituted for dataset terms.

## Preserved roles

SDD source scenes remain development-exposed. DUT remains diagnostic after its
completed predictive readout. DroneCrowd's entire source remains reserved and
closed for potential confirmation. HT21 is `quarantined_unassigned`, with source
audit permitted and no scientific training/calibration/test role assigned.
