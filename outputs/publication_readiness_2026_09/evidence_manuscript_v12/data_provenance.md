# Data Provenance And Unit Wording

The working paper studies offline forecast replacement from existing
research releases. The project did not recruit people, collect new
drone recordings, or attempt to identify individuals. The anonymous
replay distributes aggregate statistics, not images, individual tracks,
or checkpoints.

| Release | What the primary source establishes | How this paper uses it |
|---|---|---|
| [MovingDroneCrowd++, Fan et al., arXiv v3](https://arxiv.org/html/2601.12500v3#S3) | The release annotates head boxes with clip-local identities. Its authors report downsampling by factors of 3, 6, or 9 and temporary annotation gaps during occlusion. | Use released head-box centers and contiguous history in annotation-index space. Keep incomplete futures in the evaluation roster. The 34-recording native A and 24-recording own-B studies are exposed development analyses, not the publisher's counting or tracking scores. |
| [DroneCrowd, Wen et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Wen_Detection_Tracking_and_Counting_Meets_Drones_in_Crowds_A_Benchmark_CVPR_2021_paper.pdf) | The authors describe 112 recorded clips, head trajectories, and their collection and privacy checks. Their 25 FPS capture statement is about that release, not the annotation timebase of MovingDroneCrowd++. | Apply the registered whole-group calibration and confirmation roles rather than the publisher's counting split. Report forecast steps and image pixels, not metric or matched-duration comparisons. |

The [MovingDroneCrowd++ data card](https://huggingface.co/datasets/fyw1999/MovingDroneCrowd)
declares CC BY-NC 4.0; the
[intake record](../movingdronecrowd_intake_v1/conclusions.md) pins the
actual archive revision and selected member checks. The primary
articles do not establish individual consent for this reuse; the
publisher's privacy description is not an independent privacy audit
or ethics determination for this project. Do not claim consent,
approval, exemption, or redistribution rights beyond the checked
source. The final data statement should be confirmed by the author
against the [venue's human-data guidance](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines).

The [cross-archive source screen](../dronecrowd_cross_archive_v1/README.md)
found no match meeting its predeclared three-view criteria across
the 48 unique exposed A/B recordings and 112 DroneCrowd recordings.
It does not establish that every group is a distinct physical site.
The [approved role manifest](../dronecrowd_approved_roles_v2/README.md)
fixes the 15/30 group separation. This provenance note neither opens
labels nor reports any external outcome.
