# Stage42 Experiment Tables

| experiment | source | all | t50 | t100 | hard | easy | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stage42-B protected endpoint/external validation | fresh_run | 0.2103 | 0.1365 | 0.1469 | 0.2038 | -0.1451 | protected composite-tail endpoint dynamics; external source-fold eval |
| Stage42-C protected full-waypoint dynamics ADE | fresh_run | 0.1858 | 0.1480 | 0.2286 | 0.1952 | -0.0000 | actual reconstructed future waypoint labels; positive on ETH_UCY and TrajNet |
| Stage42-C protected full-waypoint dynamics FDE | fresh_run | 0.1938 | 0.2158 | n/a | n/a | n/a | full-waypoint FDE summary |
| Stage42-E best deployable safety-floor policy | fresh_run | 0.2103 | 0.1365 | 0.1469 | 0.2038 | 0.0000 | teacher_floor_required_for_current_deployment |

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

| evidence | source | all | t50 | hard | easy | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Stage42-AB no-aux full-waypoint ablation | `fresh_run` | -0.002339 | -0.037443 | -0.002564 | 0.000000 | no-aux variant is negative on t50; auxiliary heads have small t50 support but mixed all/hard evidence |
<!-- STAGE42_AC_REFRESH:END -->
